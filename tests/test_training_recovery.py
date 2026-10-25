import copy
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ouro_pipeline import training, recovery
from ouro_pipeline.projected_full import HistoricalFullWeightProjection
from scripts.parameters import parse_args
from tests.test_early_stopping import Tokenizer


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(8, 8)
        self.dropout = torch.nn.Dropout(.25)
        self.total_ut_steps = 4

    def forward(self, input_ids, **kwargs):
        x = torch.nn.functional.one_hot(input_ids, 8).float()
        return SimpleNamespace(logits=self.lm_head(self.dropout(x)))


@pytest.mark.parametrize("projected", [False, True])
def test_optimizer_recovery_matches_uninterrupted_training_across_epochs(tmp_path, projected):
    torch.manual_seed(123)
    initial = Model()
    records = [{"prompt": "Q" * (i % 3 + 1), "answer": "A" * (i % 4 + 1)} for i in range(23)]
    args = parse_args(["train", "--output-dir", str(tmp_path), "--tasks", "FOMC",
                       "--epochs", "3", "--max-length", "24", "--gradient-accumulation", "2"])
    def guard(model):
        return HistoricalFullWeightProjection(model, {}, joint_bases={"lm_head": torch.eye(8)[:, :2]}) if projected else None
    reference = copy.deepcopy(initial)
    reference_guard = guard(reference)
    with patch.object(training, "move", side_effect=lambda batch, _: batch), \
         patch.dict(os.environ, {"OURO_RECOVERY_ENABLED": "0"}):
        torch.manual_seed(91)
        expected = training.train_task(reference, Tokenizer(), records, args, "FOMC", update_projection=reference_guard)
    with patch.object(training, "move", side_effect=lambda batch, _: batch), \
         patch.dict(os.environ, {"OURO_RECOVERY_ENABLED": "1"}):
        # Interrupt inside an epoch, exactly at its end, then in the next epoch.
        for expected_step in (1, 2, 3):
            model = copy.deepcopy(initial)
            projection = guard(model)
            torch.manual_seed(91 if expected_step == 1 else 999)
            with patch.object(recovery, "deadline_reached", return_value=True), pytest.raises(recovery.TrainingPaused):
                training.train_task(model, Tokenizer(), records, args, "FOMC", update_projection=projection)
            saved = torch.load(recovery.checkpoint_path(args, "FOMC"), weights_only=False)
            assert saved["progress"]["optimizer_steps"] == expected_step
        model = copy.deepcopy(initial)
        projection = guard(model)
        with patch.object(recovery, "deadline_reached", return_value=False):
            actual = training.train_task(model, Tokenizer(), records, args, "FOMC", update_projection=projection)
    assert actual == expected
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)
    if projected:
        assert projection.diagnostics() == reference_guard.diagnostics()


def test_recovery_refuses_changed_protocol(tmp_path):
    args = parse_args(["train", "--output-dir", str(tmp_path), "--tasks", "FOMC"])
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters())
    recovery.save(args, "FOMC", model, optimizer, None, {})
    args.learning_rate *= 2
    with patch.dict(os.environ, {"OURO_RECOVERY_ENABLED": "1"}), pytest.raises(ValueError, match="configuration"):
        recovery.restore(args, "FOMC", model, optimizer, None)
