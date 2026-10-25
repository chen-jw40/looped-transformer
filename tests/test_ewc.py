from unittest.mock import patch

import pytest
import torch

from ouro_pipeline.ewc import ContinualEWC, _Quadratic
from ouro_pipeline.modeling import final_exit_loss
from scripts.parameters import parse_args, training_config
from tests.test_early_stopping import TinyModel, Tokenizer
from tests.test_osft import tiny_ouro


def test_compact_history_gradient_matches_all_task_quadratics():
    model = torch.nn.Linear(3, 2, bias=False).double()
    controller = ContinualEWC(model, strength=400)
    histories = []
    for offset in (1., 3., -2.):
        with torch.no_grad():
            model.weight.fill_(offset)
        fisher = torch.arange(6, dtype=torch.double).reshape(2, 3) + 1
        histories.append((fisher.clone(), model.weight.detach().clone()))
        controller.consolidate({"weight": fisher})
    with torch.no_grad():
        model.weight.fill_(.7)
    penalty = controller.regularization_loss()
    compact_grad, = torch.autograd.grad(penalty, model.weight)
    explicit = sum(200 * (f * (model.weight - anchor).square()).sum() for f, anchor in histories)
    expected_grad, = torch.autograd.grad(explicit, model.weight)
    torch.testing.assert_close(compact_grad, expected_grad)
    assert controller.tasks == 3
    assert torch.autograd.gradcheck(_Quadratic.apply, (model.weight, histories[0][0], histories[0][1]))


def test_fisher_is_mean_of_example_squares_and_preserves_rng_weights_and_modes(tmp_path):
    model = TinyModel()
    controller = ContinualEWC(model)
    args = parse_args(["train", "--method", "ewc", "--output-dir", str(tmp_path),
                       "--max-length", "20", "--ewc-fisher-samples", "2"])
    records = [{"prompt": "Q", "answer": "A"}, {"prompt": "Q", "answer": "AA"}]
    from ouro_pipeline.data import AnswerOnlyCollator
    collate = AnswerOnlyCollator(Tokenizer(), args.max_length)
    gradients = []
    for record in records:
        model.zero_grad()
        final_exit_loss(model, collate([record]), 4)[0].backward()
        gradients.append(model.weight.grad.clone())
    model.eval()
    before = model.weight.detach().clone()
    rng = torch.get_rng_state().clone()
    report = controller.collect(Tokenizer(), records, args, "FOMC")
    torch.testing.assert_close(controller.precision["weight"], torch.stack(gradients).square().mean(0))
    assert not torch.allclose(controller.precision["weight"], torch.stack(gradients).mean(0).square())
    assert not model.training and model.weight.grad is None
    assert torch.equal(rng, torch.get_rng_state()) and torch.equal(before, model.weight)
    assert report["samples"] == 2 and report["split"] == "train"
    assert controller.regularization_loss().item() == 0
    with torch.no_grad():
        model.weight.add_(1)
    assert controller.regularization_loss().item() > 0
    controller.save(tmp_path / "ewc/state.pt")
    saved = torch.load(tmp_path / "ewc/state.pt", weights_only=True)
    assert saved["tasks"] == 1 and saved["precision"]["weight"].device.type == "cpu"


def test_actual_ouro_shared_loop_gradients_and_next_task_regularization(tmp_path):
    model = tiny_ouro()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    controller = ContinualEWC(model)
    args = parse_args(["train", "--method", "ewc", "--output-dir", str(tmp_path),
                       "--max-length", "20", "--ewc-fisher-samples", "2"])
    keys = set(model.state_dict())
    records = [{"prompt": "Q", "answer": "A"}] * 2
    report = controller.collect(Tokenizer(), records, args, "FOMC")
    assert report["precision_trace"] > 0 and set(model.state_dict()) == keys
    assert len(controller.precision) == len(list(model.named_parameters()))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(.01)
    batch = {"input_ids": torch.tensor([[1, 3, 4, 5, 2]]),
             "attention_mask": torch.ones(1, 5, dtype=torch.long),
             "labels": torch.tensor([[-100, -100, 4, 5, 2]])}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = final_exit_loss(model, batch, 4)
    penalty = controller.regularization_loss()
    assert penalty.item() > 0
    (loss + penalty).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_full_training_driver_collects_best_weights_and_saves_every_stage(tmp_path):
    import ouro_pipeline.full_training as driver
    model = TinyModel()
    model.gradient_checkpointing_enable = lambda **kwargs: None
    model.save_pretrained = lambda path, **kwargs: torch.save(model.state_dict(), path / "weights.pt")
    tokenizer = Tokenizer()
    tokenizer.save_pretrained = lambda path: None
    args = parse_args(["train", "--method", "ewc", "--output-dir", str(tmp_path / "run"),
                       "--data-root", str(tmp_path), "--tasks", "FOMC", "ScienceQA", "--max-length", "20"])
    import json
    for task in args.tasks:
        (tmp_path / task).mkdir()
        for split in ("train", "eval", "test"):
            (tmp_path / task / f"{split}.json").write_text(json.dumps([{"prompt": "Q", "answer": "A"}]))

    def train(*pos, regularizer, **kwargs):
        assert isinstance(regularizer, ContinualEWC)
        if regularizer.tasks:
            assert regularizer.precision["weight"].sum() > 0
        with torch.no_grad():
            model.weight[3].add_(.1)
        return {"optimizer_steps": 1}

    with patch.object(driver.torch.cuda, "is_available", return_value=True), \
         patch.object(driver.torch.cuda, "is_bf16_supported", return_value=True), \
         patch.object(driver.torch.cuda, "get_device_name", return_value="CPU mock"), \
         patch.object(driver.torch.cuda, "reset_peak_memory_stats"), \
         patch.object(driver.torch.cuda, "synchronize"), \
         patch.object(driver.torch.cuda, "max_memory_allocated", return_value=0), \
         patch.object(driver.torch.cuda, "max_memory_reserved", return_value=0), \
         patch.object(driver, "load_ouro", return_value=model), \
         patch.object(driver, "load_tokenizer", return_value=tokenizer), \
         patch("ouro_pipeline.training.train_task", side_effect=train):
        driver.run(args)
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["training_status"] == "complete"
    assert [r["ewc_fisher"]["historical_tasks"] for r in summary["training"]] == [1, 2]
    assert all(torch.load(s["ewc_state"], weights_only=True)["tasks"] == i
               for i, s in enumerate(summary["stages"], 1))


def test_protocol_and_invalid_settings():
    args = parse_args(["train", "--method", "ewc", "--output-dir", "runs/test"])
    config = training_config(args)
    assert args.checkpoint_scope == "all" and args.ewc_lambda == 400 and args.ewc_fisher_samples == 64
    assert "route_depths" not in config and config["ewc_lambda"] == 400
    for extra in (["--ewc-lambda", "nan"], ["--ewc-lambda", "-1"], ["--ewc-fisher-samples", "0"]):
        with pytest.raises(SystemExit):
            parse_args(["train", "--method", "ewc", "--output-dir", "runs/test", *extra])


@pytest.mark.parametrize("mismatch", [False, True])
def test_ewc_statistics_preserve_provenance_and_reject_different_strengths(tmp_path, mismatch):
    import json
    from tests.test_study_statistics import make_study
    from scripts.study_statistics import run
    make_study(tmp_path)
    (tmp_path / "osft").rename(tmp_path / "ewc")
    for seed in (42, 43):
        path = tmp_path / "ewc" / f"seed{seed}" / "summary.json"
        summary = json.loads(path.read_text())
        summary["config"].update(method="ewc", ewc_lambda=400 if not mismatch or seed == 42 else 200,
                                 ewc_fisher_samples=64)
        summary["ewc_protocol"] = {"variant": "multitask", "reference_commit": "pinned"}
        path.write_text(json.dumps(summary))
    args = parse_args(["summarize-study", "--output-root", str(tmp_path),
                       "--seeds", "42", "43", "--methods", "ewc"])
    if mismatch:
        with pytest.raises(ValueError, match="EWC protocols differ"):
            run(args)
        assert not (tmp_path / "statistics.json").exists()
    else:
        run(args)
        payload = json.loads((tmp_path / "statistics.json").read_text())
        assert payload["std_ddof"] == 1
        assert all(s["ewc_protocol"]["reference_commit"] == "pinned" for s in payload["sources"])
