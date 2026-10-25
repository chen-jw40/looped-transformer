import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from ouro_pipeline.early_stopping import EarlyStoppingState, TaskEarlyStopping, validation_steps


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return [3] * len(text)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8))
        self.total_ut_steps = 4

    def forward(self, input_ids, **kwargs):
        logits = self.weight.expand(*input_ids.shape, 8)
        return SimpleNamespace(logits=logits)


class EarlyStoppingTests(unittest.TestCase):
    def test_training_stops_at_validation_boundary_and_returns_best_weights(self):
        from ouro_pipeline import training

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "FOMC").mkdir()
            records = [{"prompt": "Q", "answer": "A"} for _ in range(8)]
            (root / "FOMC/eval.json").write_text(json.dumps(records))
            args = SimpleNamespace(output_dir=root, data_root=root, loops=4, max_length=20,
                                   train_limit=8, epochs=2, micro_batch_size=1,
                                   gradient_accumulation=2, learning_rate=0.1, seed=42,
                                   truncate_long_prompts=True, early_stopping=True,
                                   early_stopping_steps=2, early_stopping_max_evals=6,
                                   early_stopping_samples=2, early_stopping_seed=42,
                                   early_stopping_patience=1, early_stopping_min_delta=0.001)
            model = TinyModel()
            snapshots = []

            def loss(stopper):
                snapshots.append(model.weight.detach().clone())
                return 1.0 if len(snapshots) == 1 else 2.0

            with patch.object(training, "move", side_effect=lambda batch, device: batch), \
                 patch.object(TaskEarlyStopping, "validation_loss", loss):
                report = training.train_task(model, Tokenizer(), records, args, "FOMC")
            self.assertEqual(report["optimizer_steps"], 4)
            self.assertEqual(report["micro_steps"], 8)
            self.assertTrue(report["early_stopping"]["stopped_early"])
            self.assertEqual(report["early_stopping"]["best_optimizer_step"], 2)
            self.assertFalse(torch.equal(snapshots[0], snapshots[1]))
            self.assertTrue(torch.equal(model.weight, snapshots[0]))

    def test_check_schedule_caps_evaluation_and_always_includes_final_update(self):
        self.assertEqual(validation_steps(313, 150, 6), [150, 300, 313])
        self.assertEqual(validation_steps(20, 150, 6), [20])
        for total in [1, 312, 313, 939, 10000]:
            schedule = validation_steps(total, 150, 6)
            self.assertLessEqual(len(schedule), 6)
            self.assertEqual(schedule[-1], total)

    def test_patience_resets_on_improvement_and_counts_checks_not_steps(self):
        state = EarlyStoppingState(patience=2, min_delta=0.01)
        self.assertEqual(state.observe(1.0, 150), (True, False))
        self.assertEqual(state.observe(0.995, 300), (False, False))
        self.assertEqual(state.observe(0.9, 450), (True, False))
        self.assertEqual(state.observe(1.1, 600), (False, False))
        self.assertEqual(state.observe(1.2, 750), (False, True))
        self.assertEqual(state.best_step, 450)
        with self.assertRaises(FloatingPointError):
            state.observe(float("nan"), 900)

    def test_eval_only_token_loss_and_restoration_without_changing_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "FOMC"
            data.mkdir()
            (data / "eval.json").write_text(json.dumps([
                {"prompt": "Q", "answer": "A"}, {"prompt": "Q2", "answer": "AA"}]))
            # There is deliberately no test split: selection must use eval only.
            args = SimpleNamespace(output_dir=root, data_root=root, loops=4,
                                   max_length=20, truncate_long_prompts=True,
                                   early_stopping_steps=1, early_stopping_max_evals=3,
                                   early_stopping_samples=128, early_stopping_seed=42,
                                   early_stopping_patience=2, early_stopping_min_delta=0.001)
            model = TinyModel()
            stopper = TaskEarlyStopping(model, Tokenizer(), args, "FOMC", 3, None)
            model.train()
            rng = torch.get_rng_state().clone()
            self.assertAlmostEqual(stopper.validation_loss(), math.log(8), places=6)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse(stopper.check(1))
            with torch.no_grad():
                model.weight[0] = 5
            self.assertFalse(stopper.check(2))
            self.assertTrue(stopper.check(3))
            report = stopper.restore()
            self.assertTrue(torch.equal(model.weight, torch.zeros(8)))
            self.assertEqual(report["best_optimizer_step"], 1)
            self.assertTrue(report["stopped_early"])
            self.assertFalse(stopper.path.exists())


if __name__ == "__main__":
    unittest.main()
