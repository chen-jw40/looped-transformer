"""Opt-in atomic optimizer recovery, plus task-boundary RNG for continual baselines."""
import os
from pathlib import Path
import random
import time

import torch


class TrainingPaused(RuntimeError):
    pass


def enabled():
    return os.environ.get("OURO_RECOVERY_ENABLED") == "1"


def checkpoint_path(args, task):
    return Path(args.output_dir) / "recovery" / f"{task}.pt"


def restore(args, task, model, optimizer, projection):
    path = checkpoint_path(args, task)
    if not enabled() or not path.exists():
        return None
    from scripts.parameters import training_config
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["config"] != training_config(args) or state["task"] != task:
        raise ValueError("Recovery checkpoint configuration differs")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    constraints = getattr(projection, "constraints", [])
    if [x[0] for x in constraints] != [x[0] for x in state["constraints"]]:
        raise ValueError("Recovery projection modules differ")
    for (_, _, q, target, _, bias_target), (_, saved_q, saved_target, saved_bias) in zip(constraints, state["constraints"]):
        q.copy_(saved_q)
        target.copy_(saved_target)
        if bias_target is not None:
            bias_target.copy_(saved_bias)
    if projection is not None:
        for key, value in state["projection_diagnostics"].items():
            setattr(projection, key, value)
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["progress"]


def save(args, task, model, optimizer, projection, progress):
    from scripts.parameters import training_config
    path = checkpoint_path(args, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(config=training_config(args), task=task, model=model.state_dict(),
        optimizer=optimizer.state_dict(), progress=progress, python_rng=random.getstate(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        constraints=[(name, q, target, bias_target) for name, _, q, target, _, bias_target in
                     getattr(projection, "constraints", [])],
        projection_diagnostics={key: getattr(projection, key) for key in
            ("steps", "max_relative_error", "max_absolute_error") if hasattr(projection, key)})
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def deadline_reached():
    return time.time() >= float(os.environ.get("OURO_RECOVERY_DEADLINE", "inf"))


def save_task_boundary(args, checkpoint):
    path = Path(args.output_dir) / "continuation" / f"{checkpoint}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def restore_task_boundary(args, checkpoint):
    state = torch.load(Path(args.output_dir) / "continuation" / f"{checkpoint}.pt",
                       map_location="cpu", weights_only=False)
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
