import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ouro_pipeline.evaluation import evaluation_root
from scripts.parameters import MAX_NEW_TOKENS, PRIMARY_METRICS, TASKS, parse_args, training_config
from scripts.runtime import container_command
from scripts.experiment import main


class ExperimentEntryTests(unittest.TestCase):
    def test_all_methods_share_eight_tasks_and_fixed_epochs(self):
        for method in ["full", "projected", "osft"]:
            args = parse_args(["train", "--method", method, "--output-dir", "runs/new"])
            self.assertEqual(args.tasks, TASKS)
            self.assertEqual(args.eval_split, "test")
            self.assertFalse(args.early_stopping)
            self.assertEqual(args.early_stopping_max_evals, 6)
            config = training_config(args)
            self.assertEqual(config["primary_metrics"]["20Minuten"], "sari")
            self.assertNotIn("depth_schedule", config)

    def test_unsupported_methods_are_rejected_before_execution(self):
        for method in ["grouped", "depth_lora"]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["train", "--method", method, "--output-dir", "runs/unused"])

    def test_full_saved_protocol_does_not_gain_projected_depth_settings(self):
        args = parse_args(["train", "--method", "full", "--output-dir", "runs/unused"])
        config = training_config(args)
        self.assertNotIn("basis_depth_mode", config)
        self.assertEqual(config["route_depths"], [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(config["checkpoint_scope"], "final")

    def test_dry_run_never_launches_a_container_or_creates_outputs(self):
        with tempfile.TemporaryDirectory() as temp, patch("scripts.experiment.run_in_container") as launch:
            output = Path(temp) / "must_not_exist"
            with contextlib.redirect_stdout(io.StringIO()) as buffer:
                main(["train", "--method", "projected", "--output-dir", str(output), "--dry-run"])
            config = json.loads(buffer.getvalue())
            self.assertEqual(config["basis_depth_mode"], "all")
            self.assertIsNone(config["route_depths"])
            self.assertEqual(config["checkpoint_scope"], "all")
            launch.assert_not_called()
            self.assertFalse(output.exists())

    def test_split_change_requires_isolated_output_and_resets_completed_evaluations(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            cp = source / "checkpoints/02_after_ScienceQA"
            cp.mkdir(parents=True)
            original = {"config": {"tasks": ["FOMC", "ScienceQA"], "eval_split": "eval",
                                   "eval_limit": 128, "model": "models/Ouro", "data_root": "data"},
                        "evaluations": {"02_after_ScienceQA": {"FOMC": {"accuracy": 1}}}}
            raw = json.dumps(original)
            (source / "summary.json").write_text(raw)
            args = parse_args(["evaluate", "--run-dir", str(source), "--split", "test"])
            with self.assertRaises(ValueError):
                evaluation_root(args)
            args.output_dir = Path(temp) / "test"
            root = evaluation_root(args)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["config"]["eval_split"], "test")
            self.assertIsNone(summary["config"]["eval_limit"])
            self.assertEqual(summary["evaluations"], {})
            self.assertEqual((root / "checkpoints/02_after_ScienceQA").resolve(), cp)
            self.assertEqual((source / "summary.json").read_text(), raw)
            self.assertEqual(evaluation_root(args), root)

    def test_container_launch_uses_python_directly_and_preserves_vllm_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            image = root / "containers/nvidia-vllm-26.02-py3.sif"
            image.parent.mkdir(parents=True)
            image.touch()
            with patch("scripts.runtime.REPO_ROOT", root), \
                 patch("scripts.runtime.CODE_ROOT", root), \
                 patch("scripts.runtime.shutil.which", return_value="/usr/bin/apptainer"):
                command, env = container_command("vllm", ["worker", "--run-dir", "runs/test"])
            self.assertNotIn("bash", command)
            self.assertFalse(any(arg.endswith(".sh") for arg in command))
            self.assertIn(str(root / "scripts/experiment.py"), command)
            self.assertEqual(env["APPTAINERENV_OURO_RUNTIME"], "vllm")
            self.assertEqual(env["APPTAINERENV_CUDA_VISIBLE_DEVICES"], "0")

    def test_sari_is_normalized_before_macro_and_forgetting(self):
        from ouro_pipeline.training import continual_aggregates, continual_metrics

        evaluations = {"01_after_20Minuten": {"20Minuten": {"sari": 40}},
                       "02_after_FOMC": {"20Minuten": {"sari": 32}, "FOMC": {"accuracy": 0.8}}}
        matrix, forgetting = continual_metrics(evaluations, ["20Minuten", "FOMC"], PRIMARY_METRICS)
        aggregate = continual_aggregates(matrix, forgetting, ["20Minuten", "FOMC"])
        self.assertAlmostEqual(aggregate["final_macro_average"], 0.56)
        self.assertAlmostEqual(aggregate["backward_transfer"], -0.08)


if __name__ == "__main__":
    unittest.main()
