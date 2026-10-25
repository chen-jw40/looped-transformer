import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import test_runs as runner


class DirectFinetuneTestTests(unittest.TestCase):
    def test_test_results_are_isolated_resumable_and_verified_against_actual_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "runs/20261002_full_unconstrained_k4_vllm"
            checkpoint = source / "checkpoints" / runner.FINAL_CHECKPOINT
            checkpoint.mkdir(parents=True)
            for name in ["config.json", "modeling_ouro.py", "weights.safetensors"]:
                (checkpoint / name).write_text("{}")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "weights.safetensors"}}))
            config = {"seed": 42, "tasks": runner.TASKS, "loops": 4, "eval_loops": 4,
                      "max_length": 4096, "model": "models/Ouro", "data_root": "data",
                      "eval_split": "eval", "eval_limit": 128, "wandb_mode": "online"}
            raw = json.dumps({"config": config, "method": "unconstrained full fine-tuning",
                              "training": [], "evaluations": {"old": {"FOMC": {"accuracy": 1}}}}).encode()
            (source / "summary.json").write_bytes(raw)
            for task in runner.TASKS:
                data = root / "data" / task
                data.mkdir(parents=True)
                (data / "test.json").write_text(json.dumps([{"prompt": "TEST", "answer": "A"}]))
                (data / "eval.json").write_text(json.dumps([{"prompt": "EVAL", "answer": "B"}]))
            with patch.object(runner, "REPO_ROOT", root):
                output, summary = runner.prepare(42, root / "test_output")
                self.assertEqual(summary["config"]["eval_split"], "test")
                self.assertIsNone(summary["config"]["eval_limit"])
                self.assertEqual(summary["config"]["wandb_mode"], "disabled")
                self.assertEqual(summary["evaluations"], {})
                self.assertEqual((output / "checkpoints" / runner.FINAL_CHECKPOINT).resolve(), checkpoint)
                runner.prepare(42, root / "test_output")
                options = runner.parse_args(["test-seeds", "--seed", "42", "--loop-depth", "2",
                                             "--max-length", "256", "--temperature", "0.4",
                                             "--max-new-tokens-by-task", "FOMC=17"])
                with self.assertRaises(ValueError):
                    runner.prepare(42, root / "test_output", options)
                _, custom = runner.prepare(42, root / "custom_output", options)
                self.assertEqual(custom["test_protocol"]["loops"], 2)
                self.assertEqual(custom["config"]["eval_loops"], 2)
                self.assertEqual(custom["config"]["max_length"], 256)
                self.assertEqual(custom["config"]["temperature"], 0.4)
                self.assertEqual(custom["config"]["max_new_tokens_by_task"]["FOMC"], 17)
                results = {}
                for task in runner.TASKS:
                    results[task] = {"samples": 1, "accuracy": 0.5, "rouge-L": 0.5,
                                     "similarity": 50, "sari": 40}
                    path = output / "predictions" / runner.FINAL_CHECKPOINT / f"{task}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"index": 0, "prompt": "TEST", "reference": "A"}) + "\n")
                summary["evaluations"][runner.FINAL_CHECKPOINT] = results
                runner.save(summary, output)
                runner.finalize(output)
                completed = json.loads((output / "summary.json").read_text())
                self.assertAlmostEqual(completed["test_scores"]["macro_score_100"], 48.75)
                self.assertEqual(completed["test_scores"]["tasks"]["20Minuten"]["metric"], "sari")
                self.assertEqual((source / "summary.json").read_bytes(), raw)
                path = output / "predictions" / runner.FINAL_CHECKPOINT / "FOMC.jsonl"
                path.write_text(json.dumps({"index": 0, "prompt": "EVAL", "reference": "B"}) + "\n")
                with self.assertRaises(ValueError):
                    runner.finalize(output)


if __name__ == "__main__":
    unittest.main()
