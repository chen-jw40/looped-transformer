"""Convert the explicit Python settings in PBS files into the unified CLI."""
from __future__ import annotations

import os

from scripts.experiment import main


def arguments(command, settings):
    argv = [command]
    for key, value in settings.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            elif key in {"early_stopping", "truncate_long_prompts"}:
                argv.append("--no-" + key.replace("_", "-"))
            elif key not in {"dry_run", "train_only", "prepare_only"}:
                raise ValueError(f"Unsupported boolean setting: {key}")
        elif isinstance(value, dict):
            argv.extend([flag, *(f"{name}={item}" for name, item in value.items())])
        elif isinstance(value, (list, tuple)):
            argv.extend([flag, *(str(item) for item in value)])
        else:
            argv.extend([flag, str(value)])
    if os.environ.get("PBS_DRY_RUN") == "1" and "--dry-run" not in argv:
        argv.append("--dry-run")
    return argv


def launch(command, settings):
    main(arguments(command, settings))
