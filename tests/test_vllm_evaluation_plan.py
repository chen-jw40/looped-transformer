import tempfile
import unittest
from pathlib import Path

from ouro_pipeline.evaluation import checkpoint_plan, tasks_by_depth
from ouro_pipeline.vllm_budget import choose_batch_size


class VllmEvaluationPlanTests(unittest.TestCase):
    def test_full_checkpoints_evaluate_only_seen_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "checkpoints" / "00_initial").mkdir(parents=True)
            (root / "checkpoints" / "01_after_FOMC").mkdir(parents=True)
            (root / "checkpoints" / "02_after_ScienceQA").mkdir()
            summary = {"config": {"tasks": ["FOMC", "ScienceQA"],
                                  "eval_tasks": ["FOMC", "ScienceQA"],
                                  "eval_loops": 4, "loops": 4}, "evaluations": {}}
            self.assertEqual(checkpoint_plan(summary, root),
                             [("01_after_FOMC", 1), ("02_after_ScienceQA", 2)])
            grouped, selected = tasks_by_depth(summary, root, "01_after_FOMC", 1)
            self.assertEqual(selected, ["FOMC"])
            self.assertEqual(grouped, {4: ["FOMC"]})

    def test_batch_size_uses_available_memory(self):
        self.assertEqual(choose_batch_size(96), 64)
        self.assertEqual(choose_batch_size(80), 48)
        self.assertEqual(choose_batch_size(50), 32)
        self.assertEqual(choose_batch_size(30), 16)


if __name__ == "__main__":
    unittest.main()
