"""Python-only launching of the pinned training and vLLM containers."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

from scripts.parameters import REPO_ROOT

CODE_ROOT = Path(__file__).resolve().parents[1]


def apptainer_executable():
    executable = shutil.which("apptainer")
    if executable:
        return executable
    site_executable = Path("/work/opt/local/aarch64/cores/apptainer/1.3.5/bin/apptainer")
    if site_executable.is_file() and os.access(site_executable, os.X_OK):
        return str(site_executable)
    # pbsdsh starts non-login shells, without the site's module initialization.
    modulecmd = shutil.which("modulecmd")
    if modulecmd:
        env = dict(os.environ)
        modulepaths = env.get("MODULEPATH", "").split(os.pathsep)
        site_modules = "/work/opt/local/modules/modulefiles/LNG/util"
        if Path(site_modules).is_dir() and site_modules not in modulepaths:
            modulepaths.append(site_modules)
        env["MODULEPATH"] = os.pathsep.join(filter(None, modulepaths))
        result = subprocess.run([modulecmd, "python", "load", "apptainer/1.3.5"],
                                env=env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Cannot load apptainer/1.3.5: {result.stderr.strip()}")
        exec(compile(result.stdout, "<modulecmd apptainer>", "exec"), {"os": os})
        executable = shutil.which("apptainer")
    if executable is None:
        raise RuntimeError("Install Apptainer or load the apptainer/1.3.5 environment module")
    return executable


def container_command(runtime, arguments):
    if runtime not in {"training", "vllm"}:
        raise ValueError(runtime)
    executable = apptainer_executable()
    env = dict(os.environ)
    env["APPTAINERENV_OURO_RUNTIME"] = runtime
    env["APPTAINERENV_PYTHONUNBUFFERED"] = "1"
    binds = ["--bind", f"{REPO_ROOT}:{REPO_ROOT}"]
    if runtime == "training":
        image = REPO_ROOT / "containers/nvidia-pytorch-25.01-py3.sif"
        python = str(REPO_ROOT / ".venv-ouro-gpu/bin/python")
        env["APPTAINERENV_PYTHONPATH"] = str(CODE_ROOT)
        env["APPTAINERENV_HF_MODULES_CACHE"] = str(REPO_ROOT / ".hf_modules")
        # Apply the same strict FP32 arithmetic to every training method.
        # NGC environment overrides can defeat PyTorch's allow_tf32 flag.
        env["APPTAINERENV_TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
        env["APPTAINERENV_NVIDIA_TF32_OVERRIDE"] = "0"
    else:
        image = REPO_ROOT / "containers/nvidia-vllm-26.02-py3.sif"
        python = "python3"
        cache = REPO_ROOT.parent / ".cache/vllm"
        cache.mkdir(parents=True, exist_ok=True)
        binds += ["--bind", f"{cache}:{cache}"]
        env.update(APPTAINERENV_VLLM_CACHE_ROOT=str(cache), APPTAINERENV_CUDA_VISIBLE_DEVICES="0",
                   APPTAINERENV_HF_HUB_OFFLINE="1",
                   APPTAINERENV_PYTHONPATH=f"{REPO_ROOT / '.vllm-metrics'}:{CODE_ROOT}",
                   APPTAINERENV_HF_MODULES_CACHE=str(REPO_ROOT / ".hf_modules_vllm"))
    if not image.is_file():
        raise FileNotFoundError(image)
    return [executable, "exec", "--nv", *binds, "--pwd", str(REPO_ROOT), str(image), python,
            str(CODE_ROOT / "scripts/experiment.py"), *arguments], env


def run_in_container(runtime, arguments):
    command, env = container_command(runtime, arguments)
    subprocess.run(command, env=env, cwd=REPO_ROOT, check=True)
