import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ouro_pipeline.evaluation import evaluation_root
from scripts.parameters import REPO_ROOT, parse_args
from scripts.pbs_launch import arguments


class PBSConfigurationTests(unittest.TestCase):
    def test_depth_controls_keep_content_sampling_budget_fixed_across_ranks(self):
        with tempfile.TemporaryDirectory() as temp:
            for variant in ("first", "last", "joint_local"):
                for rank in (64, 128, 512):
                    output = Path(temp) / f"{variant}{rank}"
                    env = {**os.environ, "PBS_DRY_RUN": "1", "PBS_O_WORKDIR": str(REPO_ROOT),
                           "VARIANT": variant, "RANK": str(rank), "SEED": "42", "RUN_DIR": str(output)}
                    env.pop("CODE_DIR", None)
                    result = subprocess.run(["bash", str(REPO_ROOT / "pbs/train_depth_controls.pbs")],
                                            env=env, check=True, capture_output=True, text=True)
                    config = json.loads(result.stdout)
                    self.assertEqual(config["basis_max_rank"], rank)
                    self.assertEqual((config["stat_samples"], config["stat_vectors_per_call"],
                                      config["stat_vectors_per_layer"]), (64, 16, 1024))
                    self.assertEqual(config["eval_split"], "test")
                    self.assertIsNone(config["eval_limit"])
                    self.assertEqual(config["checkpoint_scope"], "all")
                    if variant == "joint_local":
                        self.assertEqual(config["basis_strategy"], "joint_local")
                        self.assertEqual(config["basis_depth_mode"], "all")
                        self.assertEqual(config["history_rank"], rank)
                    self.assertFalse(output.exists())

    def test_multi_seed_preview_uses_independent_paths_without_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "study"
            env = {**os.environ, "PBS_DRY_RUN": "1", "PBS_O_WORKDIR": str(REPO_ROOT),
                   "RUN_DIR": str(root), "METHOD": "projected", "SEED": "99",
                   "SEEDS": "42:43:44:45:46"}
            env.pop("OSFT_STUDY_ROOT", None)
            env.pop("OLORA_STUDY_ROOT", None)
            result = subprocess.run(["bash", str(REPO_ROOT / "pbs/train.pbs")],
                                    env=env, check=True, capture_output=True, text=True)
            remaining, records = result.stdout.strip(), []
            while remaining:
                record, end = json.JSONDecoder().raw_decode(remaining)
                records.append(record)
                remaining = remaining[end:].strip()
            self.assertEqual([r["seed"] for r in records], [42, 43, 44, 45, 46])
            self.assertEqual([Path(r["output_dir"]) for r in records],
                             [root / f"seed{seed}" for seed in range(42, 47)])
            self.assertTrue(all(r["dry_run"] and r["checkpoint_scope"] == "all"
                                for r in records))
            self.assertFalse(root.exists())

    def test_multi_seed_preflight_and_failure_do_not_start_later_seeds(self):
        source = (REPO_ROOT / "pbs/train.pbs").read_text().split("<<'PYCONFIG'\n", 1)[1]
        source = source.rsplit("\nPYCONFIG", 1)[0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "study"
            env = {"PBS_DRY_RUN": "0", "RUN_DIR": str(root), "METHOD": "projected",
                   "SEEDS": "42:43:44:45:46"}
            with patch.dict(os.environ, env, clear=True):
                (root / "seed46").mkdir(parents=True)
                with patch("scripts.pbs_launch.launch") as launch:
                    with self.assertRaises(FileExistsError):
                        exec(compile(source, "train.pbs", "exec"), {})
                    launch.assert_not_called()
                (root / "seed46").rmdir()
                with patch("scripts.pbs_launch.launch", side_effect=[None, RuntimeError("failed")]) as launch:
                    with self.assertRaisesRegex(RuntimeError, "failed"):
                        exec(compile(source, "train.pbs", "exec"), {})
                    self.assertEqual([call.args[1]["seed"] for call in launch.call_args_list], [42, 43])
                for seeds in ("42:42", "", "42:-1", "42:bad"):
                    with self.subTest(seeds=seeds), patch.dict(os.environ, {"SEEDS": seeds}):
                        with patch("scripts.pbs_launch.launch") as launch:
                            with self.assertRaises(ValueError):
                                exec(compile(source, "train.pbs", "exec"), {})
                            launch.assert_not_called()
                with patch.dict(os.environ, {"RUN_DIR": str(REPO_ROOT / "result/new_study")}):
                    with patch("scripts.pbs_launch.launch") as launch:
                        with self.assertRaisesRegex(ValueError, "not result"):
                            exec(compile(source, "train.pbs", "exec"), {})
                        launch.assert_not_called()

    def test_every_pbs_can_preview_its_actual_settings_without_execution(self):
        expected = {"train": ["train"], "base": ["base"], "evaluate": ["evaluate"],
                    "test_seeds": ["test-seeds"], "validate": ["validate", "validate"]}
        with tempfile.TemporaryDirectory() as temp:
            output = str(Path(temp) / "must_not_exist")
            env = {**os.environ, "PBS_DRY_RUN": "1", "PBS_O_WORKDIR": str(REPO_ROOT),
                   "RUN_DIR": output, "EVAL_DIR": output + "_eval", "METHOD": "full",
                   "SEED": "43", "TEST_SEED": "44", "SPLIT": "test", "CHECKPOINT_SCOPE": "final"}
            for name, commands in expected.items():
                path = REPO_ROOT / "pbs" / f"{name}.pbs"
                subprocess.run(["bash", "-n", str(path)], check=True)
                result = subprocess.run(["bash", str(path)], env=env, check=True,
                                        capture_output=True, text=True)
                text = result.stdout.strip()
                records = []
                while text:
                    record, end = json.JSONDecoder().raw_decode(text)
                    records.append(record)
                    text = text[end:].strip()
                self.assertEqual([r["command"] for r in records], commands)
                self.assertTrue(all(r["dry_run"] for r in records))
                if name == "train":
                    self.assertEqual(records[0]["method"], "full")
                    self.assertEqual(records[0]["learning_rate"], 1e-5)
                    self.assertEqual(records[0]["seed"], 43)
                if name == "test_seeds":
                    self.assertEqual(records[0]["seed"], 44)
                    self.assertTrue(records[0]["source_run"].endswith("seed44"))
            self.assertFalse(Path(output).exists())
            self.assertFalse(Path(output + "_eval").exists())

    def test_pbs_values_preserve_booleans_lists_paths_and_generation_caps(self):
        settings = {"output_dir": "runs/path with spaces", "tasks": ["FOMC"],
                    "early_stopping": False, "truncate_long_prompts": False, "train_only": True,
                    "route_depths": [0], "loops": 2,
                    "temperature": 0.3, "max_new_tokens_by_task": {"FOMC": 17},
                    "learning_rate": 2e-5, "dry_run": False}
        with patch.dict(os.environ, {"PBS_DRY_RUN": "0"}):
            args = parse_args(arguments("train", settings))
        self.assertFalse(args.early_stopping)
        self.assertFalse(args.truncate_long_prompts)
        self.assertTrue(args.train_only)
        self.assertFalse(args.dry_run)
        self.assertEqual(str(args.output_dir), settings["output_dir"])
        self.assertEqual(args.max_new_tokens_by_task["FOMC"], 17)
        self.assertEqual(args.temperature, 0.3)
        self.assertEqual(args.loops, 2)

    def test_evaluation_overrides_reach_saved_protocol_and_cannot_mix_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "output"
            (source / "checkpoints/01_after_FOMC").mkdir(parents=True)
            original = {"config": {"tasks": ["FOMC"], "eval_tasks": ["FOMC"], "loops": 4,
                                   "eval_split": "eval", "max_length": 4096, "max_new_tokens": 100,
                                   "depth_schedule": [4], "model": "model", "data_root": "data"}}
            raw = json.dumps(original)
            (source / "summary.json").write_text(raw)
            argv = ["evaluate", "--run-dir", str(source), "--output-dir", str(output),
                    "--split", "test", "--eval-loops", "2", "--max-length", "256",
                    "--eval-limit", "3", "--max-new-tokens-by-task", "FOMC=17",
                    "--temperature", "0.3", "--no-truncate-long-prompts"]
            args = parse_args(argv)
            evaluation_root(args)
            summary = json.loads((output / "summary.json").read_text())
            config = summary["config"]
            self.assertEqual(config["eval_loops"], 2)
            self.assertNotIn("depth_schedule", config)
            self.assertEqual(config["max_length"], 256)
            self.assertEqual(config["eval_limit"], 3)
            self.assertIsNone(config["max_new_tokens"])
            self.assertEqual(config["max_new_tokens_by_task"]["FOMC"], 17)
            self.assertFalse(config["truncate_long_prompts"])
            self.assertEqual(config["temperature"], 0.3)
            self.assertEqual(evaluation_root(args), output)
            args.temperature = 0.7
            with self.assertRaises(ValueError):
                evaluation_root(args)
            self.assertEqual((source / "summary.json").read_text(), raw)

    def test_invalid_generation_values_fail_before_execution(self):
        for extra in (["--temperature", "nan"], ["--temperature", "-1"],
                      ["--max-new-tokens-by-task", "FOMC=0"],
                      ["--max-new-tokens-by-task", "unknown=8"],
                      ["--max-new-tokens-by-task", "FOMC=8", "FOMC=16"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["base", "--output-dir", "unused", *extra])
