import json
import os
import subprocess
from unittest.mock import patch

import pytest

from scripts.parameters import REPO_ROOT, TASKS, TRACE_EPOCHS, parse_args, task_epochs, training_config


def test_epoch_mapping_follows_names_and_explicit_overrides():
    base = ["train", "--output-dir", "runs/unused", "--tasks", "MeetingBank", "FOMC"]
    args = parse_args(base)
    assert args.epochs_by_task == {"MeetingBank": 7, "FOMC": 3}
    assert args.micro_batch_size == 8 and args.gradient_accumulation == 16
    assert args.early_stopping is False
    args = parse_args(base + ["--epochs", "2"])
    assert task_epochs(args, "MeetingBank") == task_epochs(args, "FOMC") == 2
    assert "epochs_by_task" not in training_config(args)  # Legacy uniform resume schema.
    args = parse_args(base + ["--epochs", "2", "--epochs-by-task", "FOMC=4"])
    assert args.epochs_by_task == {"MeetingBank": 2, "FOMC": 4}
    for override in ("FOMC=0", "FOMC=1.5", "ScienceQA=3"):
        with pytest.raises(SystemExit):
            parse_args(base + ["--epochs-by-task", override])


def test_training_runs_all_task_epochs_without_validation_or_restore(tmp_path):
    from ouro_pipeline import training
    from tests.test_early_stopping import TinyModel, Tokenizer
    records = [{"prompt": "Q", "answer": "A"} for _ in range(16)]
    args = parse_args(["train", "--output-dir", str(tmp_path), "--tasks", "FOMC", "Py150",
                       "--max-length", "20", "--gradient-accumulation", "2"])
    with patch.object(training, "move", side_effect=lambda batch, _: batch), \
         patch("ouro_pipeline.early_stopping.TaskEarlyStopping") as stopper:
        for task, epochs in (("FOMC", 3), ("Py150", 5)):
            report = training.train_task(TinyModel(), Tokenizer(), records, args, task)
            assert report["epochs_requested"] == epochs
            assert report["micro_steps"] == 2 * epochs
            assert report["optimizer_steps"] == epochs
            assert "early_stopping" not in report
        stopper.assert_not_called()


@pytest.mark.parametrize("entry", ["train", "train_adaptive", "train_last128",
                                    "train_depth_controls", "train_depth_coupled"])
def test_formal_pbs_resolves_shared_budget(entry, tmp_path):
    env = {**os.environ, "PBS_DRY_RUN": "1", "PBS_O_WORKDIR": str(REPO_ROOT),
           "RUN_DIR": str(tmp_path / "unused"), "SEED": "42", "RANK": "256"}
    for key in ("CODE_DIR", "SEEDS", "SMOKE", "PROJECTED_STUDY_ROOT", "COUPLED_STUDY_ROOT"):
        env.pop(key, None)
    # Coupled is outside this rank study; its existing allowed ranks are unchanged.
    if entry == "train_depth_coupled":
        env["RANK"] = "128"
    result = subprocess.run(["bash", str(REPO_ROOT / "pbs" / f"{entry}.pbs")],
                            env=env, capture_output=True, text=True, check=True)
    args = json.loads(result.stdout)
    assert args["micro_batch_size"] == 8
    assert args["gradient_accumulation"] == 16
    assert args["epochs_by_task"] == {task: TRACE_EPOCHS[task] for task in TASKS}
    assert args["early_stopping"] is False
    assert not (tmp_path / "unused").exists()
