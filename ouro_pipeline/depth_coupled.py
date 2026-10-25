"""Input-side sketches of gradients summed across repeated physical layers."""
from __future__ import annotations

from contextlib import contextmanager
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from ouro_pipeline.projected_full import iter_full_linears


@contextmanager
def calibration_mode(model):
    """Keep checkpointing active, disable dropout, and restore every module mode."""
    modes = [(module, module.training) for module in model.modules()]
    attention = [(module, module.attention_dropout) for module in model.modules()
                 if hasattr(module, "attention_dropout")]
    try:
        model.train()
        for module, _ in modes:
            if isinstance(module, nn.modules.dropout._DropoutNd):
                module.eval()
        for module, _ in attention:
            module.attention_dropout = 0.0
        yield
    finally:
        for module, value in attention:
            module.attention_dropout = value
        for module, mode in modes:
            module.training = mode


class DepthCoupledCollector:
    """One shared Rademacher probe per sampled position and physical module.

    Forward hooks keep only sampled input copies. Tensor hooks read the original
    graph's output adjoints; forward recording is disabled during backward so
    non-reentrant checkpoint recomputation cannot duplicate or relabel records.
    Heads retain final-depth activation protection: the LM head is called once,
    and the exit gate does not contribute to this pipeline's final-exit loss.
    """

    def __init__(self, model, loops, max_vectors, vectors_per_call, scope="all-linear",
                 seed=42, independent=False):
        if loops < 1 or min(max_vectors, vectors_per_call) < 1:
            raise ValueError("Positive loop and sampling limits required")
        self.loops, self.max_vectors, self.vectors_per_call = loops, max_vectors, vectors_per_call
        self.independent = independent
        self.handles, self.tensor_handles = [], []
        self.depths, self.chunks, self.counts, self.energies = {}, {}, {}, {}
        self.recording = False
        self.selected = dict(iter_full_linears(model, scope))
        self.decoder_keys = {k for k in self.selected if k.startswith("layers.")}
        self.head_keys = set(self.selected) - self.decoder_keys
        if not self.decoder_keys:
            raise ValueError("No recurrent linear modules selected")
        device = next(model.parameters()).device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        for index, decoder in enumerate(model.model.layers):
            def mark(_module, _args, kwargs, index=index):
                if self.recording:
                    self.depths[index] = int(kwargs["current_ut"])
            self.handles.append(decoder.register_forward_pre_hook(mark, with_kwargs=True))
            for name, module in decoder.named_modules():
                key = f"layers.{index}.{name}"
                if key not in self.decoder_keys:
                    continue
                def record(_module, args, output, key=key, index=index):
                    if self.recording:
                        self._record(key, self.depths[index], args[0], output)
                self.handles.append(module.register_forward_hook(record))
        if self.head_keys:
            def head_inputs(_module, _args, output):
                if self.recording and self.depths[len(model.model.layers) - 1] == self.loops - 1:
                    for key in self.head_keys:
                        indices = self._indices(key, output)
                        if indices.numel():
                            self.head_rows[key] = output.detach().reshape(-1, output.shape[-1]).index_select(0, indices).float().cpu()
            self.handles.append(model.model.norm.register_forward_hook(head_inputs))

    def begin(self, attention_mask):
        if self.tensor_handles:
            raise RuntimeError("Previous sample has not been finalized")
        self.valid = attention_mask.detach().reshape(-1).nonzero().flatten()
        self.pending, self.probes, self.forward_seen, self.backward_seen, self.head_rows = {}, {}, {}, {}, {}
        self.diagonal = {}
        self.recording = True

    def _indices(self, key, value):
        if value.numel() // value.shape[-1] != self.mask_size:
            raise ValueError(f"{key}: module positions do not match attention mask")
        take = min(self.max_vectors - self.counts.get(key, 0), self.vectors_per_call, self.valid.numel())
        positions = torch.linspace(0, self.valid.numel() - 1, take, device=value.device).long() if take else self.valid[:0]
        return self.valid.index_select(0, positions)

    def start(self, attention_mask):
        self.mask_size = attention_mask.numel()
        self.begin(attention_mask)

    def _record(self, key, depth, inputs, output):
        indices = self._indices(key, inputs)
        if not indices.numel():
            return
        seen = self.forward_seen.setdefault(key, set())
        if depth in seen or not 0 <= depth < self.loops:
            raise RuntimeError(f"{key}: duplicate or invalid forward depth {depth}")
        seen.add(depth)
        if not output.requires_grad:
            raise RuntimeError("Depth coupling requires a recorded graph (non-reentrant checkpointing)")
        x = inputs.detach().reshape(-1, inputs.shape[-1]).index_select(0, indices).float()
        probe_key = (key, depth if self.independent else 0)
        if probe_key not in self.probes:
            self.probes[probe_key] = torch.randint(0, 2, (len(indices), output.shape[-1]),
                device=output.device, generator=self.generator, dtype=torch.int8).float().mul_(2).sub_(1)
        probe = self.probes[probe_key]
        if key not in self.pending:
            self.pending[key] = torch.zeros_like(x)
            self.diagonal[key] = x.new_zeros(())

        def adjoint(grad):
            with torch.no_grad():
                seen = self.backward_seen.setdefault(key, set())
                if depth in seen:
                    raise RuntimeError(f"{key}: duplicate backward depth {depth}")
                seen.add(depth)
                b = grad.detach().reshape(-1, grad.shape[-1]).index_select(0, indices).float()
                contribution = (probe * b).sum(-1, keepdim=True) * x
                self.pending[key].add_(contribution)
                self.diagonal[key].add_(contribution.square().sum())
            return None
        self.tensor_handles.append(output.register_hook(adjoint))

    def backward(self, loss):
        self.recording = False
        loss.backward()

    def finish(self):
        self.recording = False
        for key, rows in self.pending.items():
            expected = set(range(self.loops))
            if self.forward_seen.get(key) != expected or self.backward_seen.get(key) != expected:
                raise RuntimeError(f"{key}: incomplete depth pairing")
            if not torch.isfinite(rows).all():
                raise FloatingPointError(f"{key}: non-finite coupled sketch")
            totals = self.energies.setdefault(key, {"coupled_energy": 0.0, "diagonal_probe_energy": 0.0})
            totals["coupled_energy"] += float(rows.square().sum())
            totals["diagonal_probe_energy"] += float(self.diagonal[key])
        for key, rows in {**self.pending, **self.head_rows}.items():
            self.chunks.setdefault(key, []).append(rows.cpu())
            self.counts[key] = self.counts.get(key, 0) + rows.shape[0]
        for handle in self.tensor_handles:
            handle.remove()
        self.tensor_handles.clear()
        self.pending.clear()
        self.probes.clear()
        self.head_rows.clear()

    def tensors(self):
        missing = self.selected.keys() - self.chunks.keys()
        if missing:
            raise RuntimeError(f"Missing coupled history modules: {sorted(missing)}")
        return {key: torch.cat(parts) for key, parts in self.chunks.items()}

    def close(self):
        self.recording = False
        for handle in self.handles + self.tensor_handles:
            handle.remove()
        self.handles.clear()
        self.tensor_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def collect_coupled_history(model, tokenizer, records, args, task, stage):
    from ouro_pipeline.data import AnswerOnlyCollator, PromptAnswerDataset, fixed_subset
    from ouro_pipeline.modeling import final_exit_loss, set_loop_count
    from ouro_pipeline.training import prompt_truncation

    started = time.perf_counter()
    # Match the existing activation calibration sample selection exactly.
    selected = fixed_subset(records, args.stat_samples, args.seed)
    loader = DataLoader(PromptAnswerDataset(selected), batch_size=1, shuffle=False,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=AnswerOnlyCollator(tokenizer, args.max_length,
            prompt_truncation=prompt_truncation(args, task)))
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    set_loop_count(model, args.loops)
    tokens = 0
    try:
        with torch.random.fork_rng(devices=devices), calibration_mode(model), torch.enable_grad(), DepthCoupledCollector(
                model, args.loops, args.stat_vectors_per_layer, args.stat_vectors_per_call,
                args.projection_scope, args.seed + stage,
                independent=args.basis_statistic == "depth_diagonal") as collector:
            for index, batch in enumerate(loader, 1):
                model.zero_grad(set_to_none=True)
                batch = {k: v.to(device) for k, v in batch.items()}
                count = int((batch["labels"][:, 1:] != -100).sum())
                if count == 0:
                    raise ValueError(f"{task}: calibration sample has no answer tokens")
                tokens += count
                collector.start(batch["attention_mask"])
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss, logits = final_exit_loss(model, batch, args.loops)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{task}: non-finite calibration loss")
                collector.backward(loss)
                collector.finish()
                del loss, logits, batch
                if index == 1 or index % 16 == 0 or index == len(selected):
                    print(f"{task}: depth-coupled calibration={index}/{len(selected)}", flush=True)
            tensors = collector.tensors()
            diagnostics = {"status": "complete", "samples": len(selected), "answer_tokens": tokens,
                "split": "train", "timing": "after restoring best eval-CE weights",
                "probe_seed": args.seed + stage, "probes_per_position": 1,
                "extra_calibration_backward_passes": len(selected),
                "sampled_vectors_per_module": dict(collector.counts),
                "probe_energy_by_module": collector.energies,
                "head_statistic": "final-depth activations", "seconds": time.perf_counter() - started}
    finally:
        model.zero_grad(set_to_none=True)
    return tensors, len(selected), diagnostics
