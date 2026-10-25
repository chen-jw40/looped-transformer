from contextlib import nullcontext
import json
from unittest.mock import patch

import pytest
import torch
from torch import nn

from ouro_pipeline.projected_full import HistoricalFullWeightProjection
from ouro_pipeline.subspace import BasisStore, FullDepthActivationCollector


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(5, 5, bias=False)
        self.self_attn.o_proj = nn.Linear(5, 5, bias=False)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(5, 5, bias=False)

    def forward(self, x, current_ut):
        x = self.self_attn.q_proj(x)
        x = self.self_attn.o_proj(x)
        return self.mlp.down_proj(x)


class TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyDecoder()])
        self.norm = nn.LayerNorm(5)
        self.early_exit_gate = nn.Linear(5, 1)

    def forward(self, x):
        for depth in range(2):
            x = self.layers[0](x, current_ut=depth)
            x = self.norm(x)
            x = x + self.early_exit_gate(x)
        return x


class TinyOuro(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyCore()
        self.lm_head = nn.Linear(5, 7, bias=False)

    def forward(self, x):
        return self.lm_head(self.model(x))


def test_collects_only_selected_depth_for_every_linear():
    torch.manual_seed(3)
    model = TinyOuro()
    with FullDepthActivationCollector(model, 12, 3, target_depth=1) as collector:
        model(torch.randn(2, 4, 5))
        samples = collector.tensors()
    assert set(samples) == {
        "layers.0.self_attn.q_proj@ut1", "layers.0.self_attn.o_proj@ut1",
        "layers.0.mlp.down_proj@ut1", "lm_head@ut1", "early_exit_gate@ut1",
    }
    assert all(value.shape == (3, 5) for value in samples.values())
    basis = BasisStore()
    basis.update(samples, max_rank=2, residual_tolerance=0.1)
    assert "lm_head@ut1" in basis.bases
    assert collector.handles == []


@pytest.mark.parametrize("take, expected", [(1, [4]), (2, [2, 8]), (16, [1, 2, 3, 4, 8, 9])])
def test_content_sampling_excludes_special_and_padding_and_covers_strata(take, expected):
    model = TinyOuro()
    # Identical BOS/EOS IDs and unequal sequence lengths, with an interior special token.
    ids = torch.tensor([[0, 11, 12, 13, 14, 0], [0, 99, 21, 22, 0, 88]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]])
    values = torch.arange(12 * 5).reshape(2, 6, 5).float()
    rng = torch.random.get_rng_state()
    with FullDepthActivationCollector(model, 64, take) as collector:
        collector.set_batch(ids, mask, {0, 99, None})
        model(values)
        samples = collector.tensors()
    torch.testing.assert_close(samples["layers.0.self_attn.q_proj@ut0"], values.reshape(-1, 5)[expected])
    assert all(rows.shape == (len(expected), 5) for rows in samples.values())
    assert {key.rsplit("@", 1)[-1] for key in samples} == {"ut0", "ut1"}
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_content_sampling_respects_budget_and_refreshes_mask_between_batches():
    model = TinyOuro()
    values = torch.arange(6 * 5).reshape(1, 6, 5).float()
    with FullDepthActivationCollector(model, 3, 2) as collector:
        collector.set_batch(torch.tensor([[0, 11, 12, 13, 14, 0]]), torch.ones(1, 6), {0})
        collector._record("test", values)
        collector.set_batch(torch.tensor([[0, 0, 21, 22, 0, 0]]), torch.ones(1, 6), {0})
        collector._record("test", values + 100)
        collector._record("test", values + 200)
        torch.testing.assert_close(collector.tensors()["test"],
                                   torch.stack([values[0, 2], values[0, 4], values[0, 3] + 100]))


def test_content_sampling_empty_tokens_and_shape_mismatch():
    with FullDepthActivationCollector(TinyOuro(), 10, 2) as collector:
        collector.set_batch(torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4), {0})
        collector._record("empty", torch.randn(1, 4, 5))
        assert collector.tensors() == {}
        with pytest.raises(ValueError, match="activation positions"):
            collector._record("bad", torch.randn(1, 3, 5))
        with pytest.raises(ValueError, match="matching"):
            collector.set_batch(torch.zeros(1, 4), torch.ones(1, 3), {0})


def test_projected_full_weights_survive_adamw_steps():
    torch.manual_seed(5)
    model = TinyOuro()
    u, _ = torch.linalg.qr(torch.randn(5, 2))
    keys = ("layers.0.self_attn.q_proj", "layers.0.self_attn.o_proj", "lm_head", "early_exit_gate")
    bases = {f"{key}@ut1": u for key in keys}
    bases["layers.0.mlp.down_proj@ut0"] = u
    guard = HistoricalFullWeightProjection(model, bases)
    assert len(guard.constraints) == len(keys) + 1
    reference = {key: weight.detach().clone() @ protected_u
                 for key, weight, protected_u, *_ in guard.constraints}
    gate_bias = model.model.early_exit_gate.bias.detach().clone()
    initial_q = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    initial_mlp = model.model.layers[0].mlp.down_proj.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.1)
    for _ in range(4):
        optimizer.zero_grad()
        model(torch.randn(2, 3, 5)).square().mean().backward()
        optimizer.step()
        guard.after_step()
        for key, weight, protected_u, *_ in guard.constraints:
            torch.testing.assert_close(weight @ protected_u, reference[key], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model.model.early_exit_gate.bias, gate_bias)
    assert not torch.equal(model.model.layers[0].self_attn.q_proj.weight, initial_q)
    assert not torch.equal(model.model.layers[0].mlp.down_proj.weight, initial_mlp)
    assert guard.diagnostics()["optimizer_steps_checked"] == 4


def test_empty_history_and_o_proj_scope():
    model = TinyOuro()
    empty = HistoricalFullWeightProjection(model, {})
    assert not empty.constraints
    empty.after_step()
    u = torch.linalg.qr(torch.randn(5, 1))[0]
    bases = {
        "layers.0.self_attn.q_proj@ut0": u,
        "layers.0.self_attn.o_proj@ut0": u,
    }
    only_o = HistoricalFullWeightProjection(model, bases, scope="o-proj")
    assert [key for key, *_ in only_o.constraints] == ["layers.0.self_attn.o_proj"]


def test_large_finite_projection_residual_is_reported_without_aborting():
    torch.manual_seed(0)
    model = TinyOuro()
    weight = model.model.layers[0].self_attn.o_proj.weight
    with torch.no_grad():
        weight.zero_()
    u = torch.linalg.qr(torch.randn(5, 2))[0]
    guard = HistoricalFullWeightProjection(
        model, {"layers.0.self_attn.o_proj@ut0": u}, scope="o-proj"
    )
    with torch.no_grad():
        weight.copy_(torch.randn_like(weight) * 1e8)
    guard.after_step()
    assert guard.diagnostics()["max_WU_history_scaled_error"] > 1e-4
    assert guard.diagnostics()["optimizer_steps_checked"] == 1


def test_route_switch_still_protects_every_old_task_depth():
    torch.manual_seed(11)
    model = TinyOuro()
    u = torch.linalg.qr(torch.randn(5, 2))[0]
    key = "layers.0.self_attn.o_proj"
    bases = {f"{key}@ut0": u[:, :1], f"{key}@ut1": u[:, 1:2]}
    guard = HistoricalFullWeightProjection(model, bases, scope="o-proj")
    assert guard.diagnostics()["history_union_ranks"][key] == 2
    assert guard.diagnostics()["historical_basis_depths"] == [0, 1]
    weight = model.model.layers[0].self_attn.o_proj.weight
    before = weight.detach() @ u
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.randn(2, 3, 5)).square().mean().backward()
        optimizer.step()
        guard.after_step()
        torch.testing.assert_close(weight @ u, before, atol=1e-5, rtol=1e-5)


def test_projected_cli_defaults_to_all_depths_and_all_checkpoints():
    from scripts.parameters import parse_args

    args = parse_args(["train", "--method", "projected", "--output-dir", "/tmp/projected_full_test", "--tasks", "FOMC", "ScienceQA"])
    assert args.basis_depth_mode == "all"
    assert args.route_depths is None
    assert args.checkpoint_scope == "all"


def test_legacy_task_depths_are_explicit_and_validated():
    from scripts.parameters import parse_args

    base = ["train", "--method", "projected", "--output-dir", "runs/unused",
            "--tasks", "FOMC", "ScienceQA", "--loops", "2"]
    legacy = parse_args([*base, "--basis-depth-mode", "task"])
    assert legacy.route_depths == [0, 0]
    explicit = parse_args([*base, "--route-depths", "0", "1"])
    assert explicit.basis_depth_mode == "task" and explicit.route_depths == [0, 1]
    for extra in (["--basis-depth-mode", "all", "--route-depths", "0", "1"],
                  ["--route-depths", "0", "2"], ["--route-depths", "0"]):
        with pytest.raises(SystemExit):
            parse_args([*base, *extra])


@pytest.mark.parametrize("mode,strategy", [("all", "append"), ("task", "append"), ("last", "append"),
                                          ("all", "joint_greedy"), ("all", "joint_pca"), ("last", "joint_pca"),
                                          ("all", "joint_balanced"), ("all", "joint_local"), ("last", "joint_local")])
def test_two_task_training_collects_and_preserves_history_on_real_ouro(tmp_path, mode, strategy):
    """CPU integration: real training, collector, cumulative bases and AdamW constraint."""
    from ouro_pipeline import full_training
    from ouro_pipeline.projected_full import iter_full_linears
    from scripts.parameters import parse_args
    from tests.test_early_stopping import Tokenizer
    from tests.test_osft import tiny_ouro
    from ouro_pipeline.adaptive_subspace import AdaptiveBasisStore

    model = tiny_ouro()
    tokenizer = Tokenizer()
    records = [{"prompt": "Q", "answer": "A"}, {"prompt": "QQ", "answer": "AA"}]
    for task in ["FOMC", "ScienceQA"]:
        folder = tmp_path / "data" / task
        folder.mkdir(parents=True)
        for split in ["train", "eval", "test"]:
            (folder / f"{split}.json").write_text(json.dumps(records))
    output = tmp_path / "run"
    args = parse_args(["train", "--method", "projected", "--output-dir", str(output),
                       "--data-root", str(tmp_path / "data"), "--tasks", "FOMC", "ScienceQA",
                       "--basis-depth-mode", mode, "--loops", "4", "--no-early-stopping",
                       "--train-limit", "2", "--stat-samples", "2", "--gradient-accumulation", "1",
                       "--stat-vectors-per-layer", "3", "--stat-vectors-per-call", "2",
                       "--basis-max-rank", "2", "--basis-residual-tolerance", "0",
                       "--basis-strategy", strategy, "--history-rank", "3", "--basis-svd-method", "exact"])
    cpu_fork_rng = torch.random.fork_rng
    with patch.object(full_training, "load_ouro", return_value=model), \
         patch.object(full_training, "load_tokenizer", return_value=tokenizer), \
         patch.object(tokenizer, "save_pretrained", create=True), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.is_bf16_supported", return_value=True), \
         patch("torch.cuda.current_device", return_value=0), \
         patch("torch.autocast", side_effect=lambda **kwargs: nullcontext()), \
         patch("torch.random.fork_rng", side_effect=lambda **kwargs: cpu_fork_rng(devices=[])), \
         patch("ouro_pipeline.training.move", side_effect=lambda batch, device: batch):
        full_training.run(args)
    summary = json.loads((output / "summary.json").read_text())
    depths = list(range(4)) if mode == "all" else [3] if mode == "last" else [0]
    expected = {f"{key}@ut{d}" for key, _ in iter_full_linears(model) for d in depths}
    assert len(expected) == 9 * len(depths)
    assert summary["training_status"] == "complete"
    assert summary["projection_protocol"]["basis_depth_mode"] == mode
    for stage in summary["stages"]:
        assert stage["basis_source_depths"] == depths
        update_keys = expected if strategy == "append" else {key.split("@ut")[0] for key in expected}
        assert {u["layer"] for u in stage["basis_updates"]} == update_keys
        assert all(u["sampled_vectors"] == 3 * (1 if strategy == "append" else len(depths))
                   for u in stage["basis_updates"])
        assert all(count == 3 for count in stage["sampled_vectors_by_depth"].values())
        assert all(u["new_rank"] <= 2 for u in stage["basis_updates"])
    store_type = BasisStore if strategy == "append" else AdaptiveBasisStore
    stores = [store_type.load(stage["basis"]) for stage in summary["stages"]]
    before, after = [store.bases for store in stores]
    assert set(before) == set(after) == update_keys
    if strategy == "append":
        for key in before:
            torch.testing.assert_close(after[key] @ (after[key].T @ before[key]), before[key],
                                       rtol=1e-5, atol=1e-5)
    else:
        assert set(stores[-1].history) == expected
        assert all(h["task_count"] == 2 for h in stores[-1].history.values())
        assert summary["projection_protocol"]["basis_strategy"] == strategy
        if strategy == "joint_local":
            assert summary["projection_protocol"]["selection_gain_scope"] == "source_depth_only"
            assert summary["projection_protocol"]["candidate_orthogonalization"] == "global across all depths"
            assert all(u["selection_gain_scope"] == "source_depth_only"
                       for stage in summary["stages"] for u in stage["basis_updates"])
    diagnostics = summary["training"][1]["projection_diagnostics"]
    assert diagnostics["historical_basis_depths"] == (depths if strategy == "append" else [])
    assert diagnostics["protected_linear_weights"] == 9
    assert diagnostics["max_WU_history_scaled_error"] < 1e-5
    assert all(rank <= 2 * len(depths) for rank in diagnostics["history_union_ranks"].values())


def test_last_depth_mode_rejects_task_routes():
    from scripts.parameters import parse_args
    with pytest.raises(SystemExit):
        parse_args(["train", "--method", "projected", "--output-dir", "runs/unused",
                    "--tasks", "FOMC", "--basis-depth-mode", "last", "--route-depths", "0"])


def test_collect_history_rejects_missing_depth_and_removes_hooks():
    from ouro_pipeline.full_training import collect_history
    from scripts.parameters import parse_args
    from tests.test_early_stopping import Tokenizer
    from tests.test_osft import tiny_ouro

    model = tiny_ouro()
    args = parse_args(["train", "--method", "projected", "--output-dir", "runs/unused"])
    tensors = FullDepthActivationCollector.tensors

    def omit_one_depth(collector):
        return {key: value for key, value in tensors(collector).items() if not key.endswith("@ut2")}

    with patch("torch.autocast", side_effect=lambda **kwargs: nullcontext()), \
         patch("ouro_pipeline.training.move", side_effect=lambda batch, device: batch), \
         patch.object(FullDepthActivationCollector, "tensors", omit_one_depth):
        with pytest.raises(RuntimeError, match="missing history activations.*@ut2"):
            collect_history(model, Tokenizer(), [{"prompt": "Q", "answer": "A"}], args, "FOMC", None)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
