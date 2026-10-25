from contextlib import ExitStack, nullcontext
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from ouro_pipeline import full_training, recovery, training
from ouro_pipeline.ewc import ContinualEWC
from ouro_pipeline.lora import ContinualLoRA
from ouro_pipeline.olora import ContinualOLoRA
from ouro_pipeline.osft import StageOSFT
from scripts.parameters import parse_args
from tests.test_early_stopping import Tokenizer
from tests.test_osft import tiny_ouro


@pytest.mark.parametrize("method", ["osft", "ewc", "olora", "lora"])
def test_baseline_optimizer_resume_preserves_updates_and_regularization(tmp_path, method):
    torch.manual_seed(31)
    initial = tiny_ouro()
    args = parse_args(["train", "--method", method, "--output-dir", str(tmp_path),
                       "--tasks", "FOMC", "--epochs", "2", "--max-length", "20",
                       "--micro-batch-size", "2", "--gradient-accumulation", "1"])
    records = [{"prompt": "Q", "answer": "A"}, {"prompt": "QQ", "answer": "AA"}] * 2

    def build():
        torch.manual_seed(51)
        model = copy.deepcopy(initial)
        projection = StageOSFT(model, 2, 8) if method == "osft" else None
        regularizer = None
        if method == "ewc":
            regularizer = ContinualEWC(model)
            regularizer.consolidate({name: torch.ones_like(p) for name, p in model.named_parameters()})
        elif method in {"lora", "olora"}:
            controller = (ContinualLoRA if method == "lora" else ContinualOLoRA)(model, rank=2)
            if method == "olora":
                with torch.no_grad():
                    for module in controller.modules.values():
                        module.B.normal_(std=.01)
                controller.consolidate()
                controller.begin_task(2)
                regularizer = controller
        return model, projection, regularizer

    reference, projection, regularizer = build()
    with patch.object(training, "move", side_effect=lambda batch, _: batch), \
         patch.dict(os.environ, {"OURO_RECOVERY_ENABLED": "0"}):
        torch.manual_seed(81)
        expected = training.train_task(reference, Tokenizer(), records, args, "FOMC",
                                       update_projection=projection, regularizer=regularizer)
    interrupted, projection, regularizer = build()
    with patch.object(training, "move", side_effect=lambda batch, _: batch), \
         patch.dict(os.environ, {"OURO_RECOVERY_ENABLED": "1"}), \
         patch.object(recovery, "deadline_reached", return_value=True), pytest.raises(recovery.TrainingPaused):
        torch.manual_seed(81)
        training.train_task(interrupted, Tokenizer(), records, args, "FOMC", update_projection=projection, regularizer=regularizer)
    restored, projection, regularizer = build()
    with patch.object(training, "move", side_effect=lambda batch, _: batch), \
         patch.dict(os.environ, {"OURO_RECOVERY_ENABLED": "1"}), \
         patch.object(recovery, "deadline_reached", return_value=False):
        actual = training.train_task(restored, Tokenizer(), records, args, "FOMC", update_projection=projection, regularizer=regularizer)
    assert actual == expected
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["osft", "ewc", "olora", "lora"])
def test_two_task_baseline_resume_restores_history_base_and_rng(tmp_path, method):
    torch.manual_seed(73)
    initial = tiny_ouro()
    for task in ("FOMC", "ScienceQA"):
        folder = tmp_path / "data" / task
        folder.mkdir(parents=True)
        for split in ("train", "eval", "test"):
            (folder / f"{split}.json").write_text(json.dumps([{"prompt": "Q", "answer": "A"}] * 2))
    tokenizer = Tokenizer()
    tokenizer.save_pretrained = lambda path: (Path(path) / "tokenizer_config.json").write_text("{}")
    base = ["train", "--method", method, "--data-root", str(tmp_path / "data"),
            "--tasks", "FOMC", "ScienceQA", "--epochs", "1", "--max-length", "20",
            "--train-limit", "2", "--micro-batch-size", "2", "--gradient-accumulation", "1",
            "--ewc-fisher-samples", "2", "--olora-rank", "2", "--lora-rank", "2"]
    real_fork_rng = torch.random.fork_rng
    real_train = training.train_task

    def load_model(path, *args, **kwargs):
        if str(path) == "models/Ouro-1.4B":
            return copy.deepcopy(initial)
        return AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float32)

    with ExitStack() as stack:
        for name, value in [("is_available", True), ("is_bf16_supported", True), ("current_device", 0),
                            ("get_device_name", "CPU mock"), ("max_memory_allocated", 0), ("max_memory_reserved", 0)]:
            stack.enter_context(patch.object(torch.cuda, name, return_value=value))
        for name in ("reset_peak_memory_stats", "synchronize"):
            stack.enter_context(patch.object(torch.cuda, name))
        stack.enter_context(patch.object(full_training, "load_ouro", side_effect=load_model))
        stack.enter_context(patch.object(full_training, "load_tokenizer", return_value=tokenizer))
        stack.enter_context(patch.object(training, "move", side_effect=lambda batch, _: batch))
        stack.enter_context(patch("torch.autocast", side_effect=lambda *args, **kwargs: nullcontext()))
        stack.enter_context(patch("torch.random.fork_rng", side_effect=lambda **kwargs: real_fork_rng(devices=[])))
        stack.enter_context(patch.dict(os.environ, {"OURO_RECOVERY_ENABLED": "1"}))
        reference = tmp_path / "reference"
        full_training.run(parse_args(base + ["--output-dir", str(reference)]))
        partial = tmp_path / "partial"

        def pause(*args, **kwargs):
            if args[4] == "ScienceQA":
                raise recovery.TrainingPaused("test task-boundary pause")
            return real_train(*args, **kwargs)

        with patch.object(training, "train_task", side_effect=pause), pytest.raises(recovery.TrainingPaused):
            full_training.run(parse_args(base + ["--output-dir", str(partial)]))
        full_training.run(parse_args(base + ["--output-dir", str(partial), "--resume"]))
    left = load_file(reference / "checkpoints/02_after_ScienceQA/model.safetensors")
    right = load_file(partial / "checkpoints/02_after_ScienceQA/model.safetensors")
    assert left.keys() == right.keys()
    for name in left:
        # A dense HF reload can change tensor strides and FP32 reduction order.
        torch.testing.assert_close(left[name], right[name], rtol=1e-6, atol=1e-9)
