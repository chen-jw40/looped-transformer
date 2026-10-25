import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def encode(self, value, add_special_tokens=False):
        return [3] * len(value)

    def decode(self, token_ids, **kwargs):
        return "A"


class FakeLLM:
    last_kwargs = None
    last_sampling = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeLLM.last_kwargs = kwargs

    def generate(self, prompts, sampling, **kwargs):
        FakeLLM.last_sampling = sampling
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[4])])
                for _ in prompts]


class VllmWorkerTests(unittest.TestCase):
    def test_full_checkpoint_writes_predictions_metrics_and_gpu_budget(self):
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = lambda **kwargs: kwargs
        fake_vllm.__version__ = "test"
        fake_metrics = types.ModuleType("ouro_pipeline.metrics")
        fake_metrics.evaluate_task = lambda task, predictions, references, sources: {
            "accuracy": sum(a == b for a, b in zip(predictions, references)) / len(references)
        }
        replacements = {"vllm": fake_vllm,
                        "ouro_pipeline.metrics": fake_metrics}
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, replacements):
            from scripts.parameters import parse_args
            worker = importlib.import_module("ouro_pipeline.vllm_worker")
            root = Path(temp)
            checkpoint = root / "checkpoints" / "01_after_FOMC"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}")
            data = root / "data" / "FOMC"
            data.mkdir(parents=True)
            (data / "eval.json").write_text(json.dumps([{"prompt": "Q", "answer": "A"}]))
            summary = {"config": {"tasks": ["FOMC"], "model": "unused", "data_root": str(root / "data"),
                                  "eval_split": "eval", "eval_limit": 1, "seed": 42,
                                  "max_length": 20, "max_new_tokens": None,
                                  "max_new_tokens_by_task": {"FOMC": 17}, "temperature": 0.3,
                                  "truncate_long_prompts": True}, "evaluations": {}}
            (root / "summary.json").write_text(json.dumps(summary))
            with patch.object(worker.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()), \
                 patch.object(worker, "gpu_budget", return_value=(64, 0.85, 95.0, 96.0)):
                worker.run(parse_args(["worker","--run-dir", str(root), "--checkpoint", "01_after_FOMC",
                             "--loop-depth", "4", "--tasks", "FOMC"]))
            result = json.loads((root / "summary.json").read_text())
            self.assertEqual(result["evaluations"]["01_after_FOMC"]["FOMC"]["accuracy"], 1.0)
            self.assertEqual(result["evaluation_runtime"]["01_after_FOMC"]["4"]["batch_size"], 64)
            self.assertEqual(FakeLLM.last_sampling["temperature"], 0.3)
            self.assertEqual(FakeLLM.last_sampling["max_tokens"], 17)
            self.assertEqual(FakeLLM.last_kwargs["max_model_len"], 37)
            self.assertEqual(len((root / "predictions" / "01_after_FOMC" / "FOMC.jsonl").read_text().splitlines()), 1)
            full_checkpoint = root / "checkpoints" / "02_after_FOMC"
            full_checkpoint.mkdir()
            (full_checkpoint / "config.json").write_text("{}")
            with patch.object(worker.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()), \
                 patch.object(worker, "gpu_budget", return_value=(64, 0.85, 95.0, 96.0)):
                worker.run(parse_args(["worker","--run-dir", str(root), "--checkpoint", "02_after_FOMC",
                             "--loop-depth", "4", "--tasks", "FOMC"]))
            self.assertEqual(FakeLLM.last_kwargs["model"], str(full_checkpoint))
            self.assertTrue(FakeLLM.last_kwargs["hf_overrides"]["use_cache"])
            self.assertNotIn("enable_lora", FakeLLM.last_kwargs)
            sys.modules.pop("ouro_pipeline.vllm_worker", None)
            sys.modules.pop("ouro_pipeline.base_evaluation", None)


if __name__ == "__main__":
    unittest.main()
