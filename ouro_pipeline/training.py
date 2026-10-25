#!/usr/bin/env python3
"""Shared training loop, sparse validation and continual-learning metrics."""
from __future__ import annotations

import json
import os
import random
import shlex
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ouro_pipeline.data import AnswerOnlyCollator, PromptAnswerDataset, fixed_subset, read_split, token_length_report
from ouro_pipeline.modeling import final_exit_loss, set_loop_count


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


from scripts.parameters import PRIMARY_METRICS, task_epochs

# Legacy summaries used ROUGE-L for 20Minuten; new configurations explicitly select SARI.
PRIMARY_METRIC = {**PRIMARY_METRICS, "20Minuten": "rouge-L"}


def write_summary(summary, output_root):
    """Atomically persist progress so an interrupted evaluation is resumable."""
    path = output_root / "summary.json"
    temporary = output_root / ".summary.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prompt_truncation(args, task):
    if not args.truncate_long_prompts:
        return "error"
    return "left" if task == "Py150" else "right"


def load_env_file(path):
    """Load missing environment variables from a local dotenv-style file."""
    if not path.is_file():
        return
    unassigned_values = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip().isidentifier():
            value = value.strip()
            try:
                parsed = shlex.split(value, comments=True)
                value = parsed[0] if parsed else ""
            except ValueError:
                value = value.strip("\"'")
            os.environ.setdefault(name.strip(), value)
        elif not separator:
            unassigned_values.append(line)
    # Also accept a .env containing only the API key, with no variable name.
    if len(unassigned_values) == 1:
        os.environ.setdefault("WANDB_API_KEY", unassigned_values[0])


def init_wandb(args, output_root):
    load_env_file(Path(args.env_file))
    if args.wandb_mode == "disabled":
        return None
    if args.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_MODE=online requested but WANDB_API_KEY is missing from environment/.env")
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or output_root.name,
        dir=str(output_root),
        mode=args.wandb_mode,
        config={key: value for key, value in vars(args).items() if not key.startswith("wandb_") and key != "env_file"},
    )


def gpu_memory_metrics(device):
    """Return current GPU memory and the PyTorch peak since the last reset."""
    gib = 1024 ** 3
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    return {
        "train/gpu/allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "train/gpu/reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "train/gpu/peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "train/gpu/peak_reserved_gib": peak_reserved_bytes / gib,
        "train/gpu/peak_reserved_pct": 100 * peak_reserved_bytes / total_bytes,
        # Device usage includes allocations from other processes on this GPU.
        "train/gpu/device_used_gib": (total_bytes - free_bytes) / gib,
        "train/gpu/device_total_gib": total_bytes / gib,
        "train/gpu/device_used_pct": 100 * (total_bytes - free_bytes) / total_bytes,
    }


def train_task(model, tokenizer, records, args, task, wandb_run=None, wandb_curves=None, update_projection=None, autocast_dtype=None, optimizer_step_offset=None, regularizer=None, gradient_observer=None):
    from ouro_pipeline import recovery
    recovering = recovery.enabled()
    if recovering and getattr(args, "early_stopping", False):
        raise ValueError("Optimizer recovery requires fixed epochs")
    if recovering and gradient_observer is not None:
        from ouro_pipeline.drsp import DRSPCollector
        if not isinstance(gradient_observer, DRSPCollector):
            raise ValueError("Optimizer recovery only supports the DRSP gradient observer")
    epochs = task_epochs(args, task)
    print(f"Training {task}: limit={args.train_limit}, epochs={epochs}", flush=True)
    set_loop_count(model, args.loops)
    selected = fixed_subset(records, args.train_limit, args.seed)
    report = token_length_report(selected, tokenizer, args.max_length)
    if report["over_max_length"] and not args.truncate_long_prompts:
        raise ValueError(f"{task}: {report['over_max_length']} training samples exceed max_length; report={report}")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        PromptAnswerDataset(selected),
        batch_size=args.micro_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=AnswerOnlyCollator(
            tokenizer,
            args.max_length,
            prompt_truncation=prompt_truncation(args, task),
        ),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=getattr(args, "weight_decay", 0.0),
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    regularization_losses = []
    task_curve = []
    micro_steps = 0
    optimizer_steps = 0
    global_step = (optimizer_step_offset if optimizer_step_offset is not None
                   else wandb_run.step if wandb_run is not None else 0)
    accumulated = 0
    restored = recovery.restore(args, task, model, optimizer, update_projection) if recovering else None
    start_epoch, skip_batches = 0, 0
    if restored is not None:
        start_epoch, skip_batches = restored["epoch"], restored["next_batch"]
        generator.set_state(restored["epoch_generator_state"])
        losses = restored["losses"]
        regularization_losses = restored.get("regularization_losses", [])
        micro_steps, optimizer_steps = restored["micro_steps"], restored["optimizer_steps"]
        global_step = restored["global_step"]
        print(f"Resumed {task}: optimizer_step={optimizer_steps}, epoch={start_epoch}, next_batch={skip_batches}", flush=True)
    memory_device = torch.cuda.current_device() if wandb_run is not None else None
    early_stopper = None
    if getattr(args, "early_stopping", False):
        from ouro_pipeline.early_stopping import TaskEarlyStopping
        total_steps = ((len(loader) + args.gradient_accumulation - 1)
                       // args.gradient_accumulation) * epochs
        early_stopper = TaskEarlyStopping(model, tokenizer, args, task, total_steps, autocast_dtype)
    stop_training = False
    if gradient_observer is not None:
        gradient_observer.configure(len(loader) * epochs)
        if restored is not None:
            gradient_observer.load_state_dict(restored["gradient_observer"])
    if memory_device is not None:
        torch.cuda.reset_peak_memory_stats(memory_device)
    for _epoch in range(start_epoch, epochs):
        epoch_generator_state = generator.get_state()
        for batch_index, batch in enumerate(loader):
            if _epoch == start_epoch and batch_index < skip_batches:
                continue
            batch = move(batch, "cuda")
            if gradient_observer is not None:
                gradient_observer.begin(micro_steps, batch, args.gradient_accumulation)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype or torch.bfloat16, enabled=autocast_dtype is not None):
                loss, _ = final_exit_loss(model, batch, args.loops)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{task}: non-finite loss at micro-step {micro_steps}")
            objective = loss
            if regularizer is not None:
                penalty = regularizer.regularization_loss()
                if not torch.isfinite(penalty):
                    raise FloatingPointError(f"{task}: non-finite regularization at micro-step {micro_steps}")
                objective = loss + penalty
                regularization_losses.append(float(penalty.detach()))
            if gradient_observer is not None:
                gradient_observer.before_backward()
            (objective / args.gradient_accumulation).backward()
            if gradient_observer is not None:
                gradient_observer.finish()
            losses.append(loss.item())
            micro_steps += 1
            accumulated += 1
            is_boundary = accumulated == args.gradient_accumulation
            is_last = batch_index == len(loader) - 1
            if is_boundary or is_last:
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"{task}: non-finite gradient in {name}")
                if update_projection is not None and hasattr(update_projection, "before_step"):
                    update_projection.before_step()
                optimizer.step()
                if update_projection is not None:
                    update_projection.after_step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                global_step += 1
                accumulated = 0
                if recovering and (optimizer_steps == 1 or optimizer_steps % 20 == 0 or recovery.deadline_reached()):
                    recovery.save(args, task, model, optimizer, update_projection, dict(
                        epoch=_epoch, next_batch=batch_index + 1, epoch_generator_state=epoch_generator_state,
                        micro_steps=micro_steps, optimizer_steps=optimizer_steps, global_step=global_step,
                        losses=losses, regularization_losses=regularization_losses,
                        **({"gradient_observer": gradient_observer.state_dict()}
                           if gradient_observer is not None else {})))
                    print(f"Recovery checkpoint saved: {task} optimizer_step={optimizer_steps}", flush=True)
                    if recovery.deadline_reached():
                        raise recovery.TrainingPaused("Allocation time budget reached; optimizer checkpoint committed")
                if wandb_run is not None:
                    metrics = {
                        f"train/loss/{task}": loss.item(),
                        "train/optimizer_step": global_step,
                        **gpu_memory_metrics(memory_device),
                    }
                    if optimizer_step_offset is None:
                        wandb_run.log(metrics, step=global_step)
                    else:
                        # Evaluation and plot logs also advance W&B's internal step.
                        # The explicit metric remains the true optimizer-step count.
                        wandb_run.log(metrics)
                    torch.cuda.reset_peak_memory_stats(memory_device)
                    task_curve.append(loss.item())
                if optimizer_steps == 1 or optimizer_steps % 10 == 0:
                    print(f"{task}: optimizer_step={optimizer_steps} loss={loss.item():.6f}", flush=True)
                if early_stopper is not None and early_stopper.check(optimizer_steps, wandb_run):
                    stop_training = True
                    break
        if stop_training:
            break
    early_stopping_report = early_stopper.restore() if early_stopper is not None else None
    if wandb_run is not None:
        if wandb_curves is not None:
            wandb_curves[task] = task_curve
            import wandb
            wandb_run.log({
                "training/loss_curves": wandb.plot.line_series(
                    xs=[list(range(1, len(values) + 1)) for values in wandb_curves.values()],
                    ys=list(wandb_curves.values()),
                    keys=list(wandb_curves),
                    title="Training loss by dataset",
                    xname="Optimizer step within dataset",
                )
            })
    return {
        "task": task,
        "samples": len(selected),
        "epochs_requested": epochs,
        "micro_steps": micro_steps,
        "optimizer_steps": optimizer_steps,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        **({"mean_regularization_loss": sum(regularization_losses) / len(regularization_losses)}
           if regularization_losses else {}),
        "lengths": report,
        "prompt_truncation": prompt_truncation(args, task),
        **({"early_stopping": early_stopping_report} if early_stopping_report is not None else {}),
    }


def continual_metrics(evaluations, tasks, primary_metrics=None):
    primary_metrics = {**PRIMARY_METRIC, **(primary_metrics or {})}
    checkpoint_names = list(evaluations)
    matrix = {}
    for checkpoint in checkpoint_names:
        matrix[checkpoint] = {}
        for task, result in evaluations[checkpoint].items():
            metric = primary_metrics[task]
            matrix[checkpoint][task] = {"metric": metric, "value": result[metric]}

    forgetting = {}
    for stage, task in enumerate(tasks, start=1):
        learned_checkpoint = f"{stage:02d}_after_{task}"
        if learned_checkpoint not in matrix or task not in matrix[learned_checkpoint]:
            continue
        later = checkpoint_names[checkpoint_names.index(learned_checkpoint):]
        values = [matrix[name][task]["value"] for name in later if task in matrix[name]]
        final_value = values[-1]
        forgetting[task] = {
            "metric": primary_metrics[task],
            "score_immediately_after_learning": values[0],
            "best_score_after_learning": max(values),
            "final_score": final_value,
            "final_forgetting_from_best": max(values) - final_value,
            "final_forgetting_from_acquisition": values[0] - final_value,
        }
    return matrix, forgetting


def continual_aggregates(matrix, forgetting, tasks):
    """Compute scale-normalized CL summaries while preserving native metrics."""
    metric_by_task = {task: result["metric"] for row in matrix.values() for task, result in row.items()}

    def normalized(task, value):
        return value / 100.0 if metric_by_task[task] in {"similarity", "sari"} else value

    checkpoint_average = {}
    for checkpoint, results in matrix.items():
        values = [normalized(task, result["value"]) for task, result in results.items()]
        checkpoint_average[checkpoint] = sum(values) / len(values) if values else None

    final_checkpoint = list(matrix)[-1]
    final_results = matrix[final_checkpoint]
    backward_terms = []
    for stage, task in enumerate(tasks[:-1], start=1):
        acquisition = f"{stage:02d}_after_{task}"
        if task in matrix.get(acquisition, {}) and task in final_results:
            backward_terms.append(
                normalized(task, final_results[task]["value"])
                - normalized(task, matrix[acquisition][task]["value"])
            )
    forgetting_terms = [
        normalized(task, result["final_forgetting_from_best"])
        for task, result in forgetting.items()
    ]
    return {
        "normalization": "all metrics in [0,1]; similarity and SARI divided by 100",
        "checkpoint_macro_average_available_tasks": checkpoint_average,
        "final_checkpoint": final_checkpoint,
        "final_macro_average": checkpoint_average[final_checkpoint],
        "backward_transfer": sum(backward_terms) / len(backward_terms) if backward_terms else None,
        "backward_transfer_task_count": len(backward_terms),
        "mean_final_forgetting_from_best": (
            sum(forgetting_terms) / len(forgetting_terms) if forgetting_terms else None
        ),
    }
