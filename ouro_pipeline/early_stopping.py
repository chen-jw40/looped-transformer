"""Sparse validation-loss checks and restoration of the best task weights."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path


def validation_steps(total_steps, minimum_interval, maximum_checks):
    """Include the final update while bounding the total number of checks."""
    if min(total_steps, minimum_interval, maximum_checks) < 1:
        raise ValueError("Step counts and validation budget must be positive")
    interval = max(minimum_interval, math.ceil(total_steps / maximum_checks))
    return sorted(set(range(interval, total_steps + 1, interval)) | {total_steps})


@dataclass
class EarlyStoppingState:
    patience: int
    min_delta: float
    best_loss: float = math.inf
    best_step: int | None = None
    bad_checks: int = 0
    history: list = field(default_factory=list)

    def observe(self, loss, step):
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        improved = loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss, self.best_step, self.bad_checks = loss, step, 0
        else:
            self.bad_checks += 1
        self.history.append({"optimizer_step": step, "loss": loss,
                             "improved": improved, "bad_checks": self.bad_checks})
        return improved, self.bad_checks >= self.patience


class TaskEarlyStopping:
    def __init__(self, model, tokenizer, args, task, total_steps, autocast_dtype):
        import torch
        from torch.utils.data import DataLoader
        from ouro_pipeline.data import AnswerOnlyCollator, PromptAnswerDataset, fixed_subset, read_split

        self.model, self.args, self.task = model, args, task
        self.autocast_dtype = autocast_dtype
        self.steps = validation_steps(total_steps, args.early_stopping_steps,
                                      args.early_stopping_max_evals)
        # Validation sampling is fixed across training seeds and never uses test.
        records = fixed_subset(read_split(args.data_root, task, "eval"),
                               args.early_stopping_samples, args.early_stopping_seed)
        if not records:
            raise ValueError(f"{task}: empty eval split")
        self.samples = len(records)
        self.loader = DataLoader(
            PromptAnswerDataset(records), batch_size=1, shuffle=False,
            generator=torch.Generator().manual_seed(args.early_stopping_seed),
            collate_fn=AnswerOnlyCollator(
                tokenizer, args.max_length,
                prompt_truncation=("left" if task == "Py150" else "right")
                if args.truncate_long_prompts else "error"),
        )
        self.state = EarlyStoppingState(args.early_stopping_patience,
                                        args.early_stopping_min_delta)
        self.path = Path(args.output_dir) / ".early_stopping" / f"{task}.pt"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stopped = False

    def validation_loss(self):
        import torch
        from ouro_pipeline.modeling import final_exit_loss

        device = next(self.model.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        was_training = self.model.training
        total_loss, total_tokens = 0.0, 0
        # Evaluation must not change the next training update's RNG sequence.
        try:
            self.model.eval()
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                for batch in self.loader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                    with torch.autocast(device_type=device.type,
                                        dtype=self.autocast_dtype or torch.bfloat16,
                                        enabled=self.autocast_dtype is not None and device.type == "cuda"):
                        loss, _ = final_exit_loss(self.model, batch, self.args.loops)
                    total_loss += float(loss.item()) * tokens
                    total_tokens += tokens
            if total_tokens == 0:
                raise ValueError("Validation contains no answer tokens")
            return total_loss / total_tokens
        finally:
            self.model.train(was_training)

    def check(self, step, wandb_run=None):
        import os
        import torch

        if step not in self.steps:
            return False
        loss = self.validation_loss()
        improved, stop = self.state.observe(loss, step)
        if improved:
            # Store on disk via CPU: no second full-weight model on the GPU.
            weights = {name: p.detach().cpu().clone() for name, p in self.model.named_parameters()
                       if p.requires_grad}
            temporary = self.path.with_suffix(".tmp")
            torch.save(weights, temporary)
            os.replace(temporary, self.path)
        print(f"{self.task}: eval_loss={loss:.6f} optimizer_step={step} "
              f"check={len(self.state.history)}/{len(self.steps)} "
              f"bad_checks={self.state.bad_checks} stop={stop}", flush=True)
        if wandb_run is not None:
            wandb_run.log({f"validation/loss/{self.task}": loss,
                           f"validation/best_loss/{self.task}": self.state.best_loss,
                           f"validation/checks/{self.task}": len(self.state.history)})
        self.stopped = stop
        return stop

    def restore(self):
        import torch

        if self.state.best_step is None:
            raise RuntimeError("No validation checkpoint was selected")
        weights = torch.load(self.path, map_location="cpu", weights_only=True)
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(weights[name])
        self.path.unlink()
        return {"split": "eval", "metric": "answer_token_cross_entropy",
                "samples": self.samples, "sampling_seed": self.args.early_stopping_seed,
                "planned_check_steps": self.steps, "history": self.state.history,
                "patience": self.state.patience, "min_delta": self.state.min_delta,
                "best_loss": self.state.best_loss, "best_optimizer_step": self.state.best_step,
                "stopped_early": self.stopped, "restored_best_weights": True}
