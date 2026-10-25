"""Multi-task diagonal empirical EWC with compact, exact historical gradients."""
from __future__ import annotations

from contextlib import contextmanager
import time

import torch
from torch.utils.data import DataLoader

from ouro_pipeline.data import AnswerOnlyCollator, PromptAnswerDataset, fixed_subset
from ouro_pipeline.modeling import final_exit_loss, set_loop_count

PAPER_URL = "https://arxiv.org/abs/1612.00796"
REFERENCE_COMMIT = "462e39f616134f4f819efeb3baea8638c03c7db4"
REFERENCE_URL = f"https://github.com/BeyonderXX/TRACE/blob/{REFERENCE_COMMIT}/model/Regular/EWC.py"


class _Quadratic(torch.autograd.Function):
    # Avoid retaining full-sized differences for all 1.4B parameters in the graph.
    @staticmethod
    def forward(ctx, parameter, precision, center):
        ctx.save_for_backward(parameter, precision, center)
        delta = parameter - center
        return (precision * delta.square()).sum() * .5

    @staticmethod
    def backward(ctx, grad_output):
        parameter, precision, center = ctx.saved_tensors
        return grad_output * precision * (parameter - center), None, None


@contextmanager
def _fisher_mode(model):
    # Train mode keeps Transformers activation checkpointing active. Disable
    # dropout so Fisher is measured at deterministic, selected task weights.
    modes = [(module, module.training) for module in model.modules()]
    attention = [(module, module.attention_dropout) for module in model.modules()
                 if hasattr(module, "attention_dropout")]
    try:
        model.train()
        for module, _ in modes:
            if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                module.eval()
        for module, _ in attention:
            module.attention_dropout = 0.0
        yield
    finally:
        for module, value in attention:
            module.attention_dropout = value
        for module, mode in modes:
            module.training = mode


class ContinualEWC:
    def __init__(self, model, strength=400.0):
        self.model, self.strength = model, strength
        self.precision, self.center = {}, {}
        self.tasks = 0

    def regularization_loss(self):
        loss = next(self.model.parameters()).new_zeros(())
        if self.strength == 0:
            return loss
        for name, parameter in self.model.named_parameters():
            if name in self.precision:
                loss = loss + self.strength * _Quadratic.apply(
                    parameter, self.precision[name], self.center[name])
        return loss

    @torch.no_grad()
    def consolidate(self, fisher):
        parameters = dict(self.model.named_parameters())
        for name, importance in fisher.items():
            parameter = parameters[name].detach()
            if not torch.isfinite(importance).all() or (importance < 0).any():
                raise FloatingPointError(f"Invalid Fisher for {name}")
            if name not in self.precision:
                self.precision[name] = importance.detach().clone()
                self.center[name] = parameter.clone()
            else:
                total = self.precision[name] + importance
                fraction = importance / total.clamp_min(torch.finfo(total.dtype).tiny)
                self.center[name].add_(fraction * (parameter - self.center[name]))
                self.precision[name].copy_(total)
        self.tasks += 1

    def collect(self, tokenizer, records, args, task, autocast_dtype=None):
        from ouro_pipeline.training import prompt_truncation

        started = time.perf_counter()
        # Fisher uses only the selected training population, never eval/test.
        population = fixed_subset(records, args.train_limit, args.seed)
        selected = fixed_subset(population, args.ewc_fisher_samples, args.seed)
        if not selected:
            raise ValueError(f"{task}: empty Fisher population")
        parameters = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        fisher = {name: torch.zeros_like(p) for name, p in parameters.items()}
        device = next(self.model.parameters()).device
        devices = [device.index or 0] if device.type == "cuda" else []
        loader = DataLoader(PromptAnswerDataset(selected), batch_size=1, shuffle=False,
                            generator=torch.Generator().manual_seed(args.seed),
                            collate_fn=AnswerOnlyCollator(tokenizer, args.max_length,
                                prompt_truncation=prompt_truncation(args, task)))
        set_loop_count(self.model, args.loops)
        tokens = 0
        used = set()
        try:
            with torch.random.fork_rng(devices=devices), _fisher_mode(self.model), torch.enable_grad():
                for index, batch in enumerate(loader, 1):
                    self.model.zero_grad(set_to_none=True)
                    batch = {key: value.to(device) for key, value in batch.items()}
                    count = int((batch["labels"][:, 1:] != -100).sum())
                    if not count:
                        raise ValueError(f"{task}: Fisher sample has no answer tokens")
                    tokens += count
                    with torch.autocast(device.type, dtype=autocast_dtype or torch.bfloat16,
                                        enabled=autocast_dtype is not None):
                        loss, logits = final_exit_loss(self.model, batch, args.loops)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"{task}: non-finite Fisher loss")
                    loss.backward()
                    for name, p in parameters.items():
                        if p.grad is not None:
                            if not torch.isfinite(p.grad).all():
                                raise FloatingPointError(f"{task}: non-finite Fisher gradient: {name}")
                            fisher[name].addcmul_(p.grad, p.grad, value=1 / len(selected))
                            used.add(name)
                    del loss, logits, batch
                    if index == 1 or index % 16 == 0 or index == len(selected):
                        print(f"{task}: Fisher sample={index}/{len(selected)}", flush=True)
        finally:
            self.model.zero_grad(set_to_none=True)
        self.consolidate(fisher)
        del fisher
        return {"split": "train", "samples": len(selected), "population_samples": len(population),
                "sampling_seed": args.seed, "answer_tokens": tokens, "status": "complete",
                "seconds": time.perf_counter() - started, "historical_tasks": self.tasks,
                "parameters_with_gradient": len(used),
                "parameters_without_gradient": sorted(parameters.keys() - used),
                "precision_trace": sum(float(value.sum()) for value in self.precision.values()),
                "state_bytes": sum(value.numel() * value.element_size()
                                   for bank in (self.precision, self.center) for value in bank.values())}

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "multitask_diagonal_ewc_v1", "tasks": self.tasks,
                    "strength": self.strength,
                    "precision": {k: v.cpu() for k, v in self.precision.items()},
                    "center": {k: v.cpu() for k, v in self.center.items()}}, path)

    def load(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        parameters = dict(self.model.named_parameters())
        if (state.get("format") != "multitask_diagonal_ewc_v1" or state["strength"] != self.strength
                or state["precision"].keys() != state["center"].keys()
                or not state["precision"].keys() <= parameters.keys()):
            raise ValueError("Incompatible EWC history")
        for name in state["precision"]:
            if state["precision"][name].shape != parameters[name].shape or state["center"][name].shape != parameters[name].shape:
                raise ValueError("EWC parameter shape changed")
        self.precision = {name: value.to(parameters[name]) for name, value in state["precision"].items()}
        self.center = {name: value.to(parameters[name]) for name, value in state["center"].items()}
        self.tasks = state["tasks"]
