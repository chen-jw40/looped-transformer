"""Python replacements for the environment and pinned-input setup scripts."""
import hashlib
from pathlib import Path
import subprocess
import sys
import zipfile

from scripts.parameters import REPO_ROOT
from scripts.runtime import container_command


def run(args):
    if args.action == "cpu":
        subprocess.run([sys.executable, "-m", "venv", str(REPO_ROOT / ".venv-ouro")], check=True)
        python = str(REPO_ROOT / ".venv-ouro/bin/python")
        subprocess.run([python, "-m", "pip", "install", "-r", "requirements-local.txt"], check=True)
    elif args.action == "gpu":
        checksum_path = REPO_ROOT / "containers/SHA256SUMS"
        for line in checksum_path.read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            image = REPO_ROOT / "containers" / name.lstrip("*")
            digest = hashlib.sha256()
            with image.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Container checksum mismatch: {image}")
        command, env = container_command("training", [])
        prefix = command[:command.index(str(REPO_ROOT / ".venv-ouro-gpu/bin/python"))]
        prefix.remove("--nv")
        for arguments in [["python3", "-m", "venv", "--system-site-packages", ".venv-ouro-gpu"],
                          [".venv-ouro-gpu/bin/python", "-m", "pip", "install", "--index-url", "https://pypi.org/simple", "-r", "requirements-ouro.txt"]]:
            subprocess.run([*prefix, *arguments], env=env, cwd=REPO_ROOT, check=True)
    elif args.action == "assets":
        python = str(REPO_ROOT / ".venv-ouro/bin/python")
        if not (REPO_ROOT / "models/Ouro-1.4B/config.json").exists():
            subprocess.run([python, "-c", "from huggingface_hub import snapshot_download; "
                "snapshot_download('ByteDance/Ouro-1.4B', revision='574fa66cb8bf5abdc979642d01cf2b79b16bfab1', local_dir='models/Ouro-1.4B')"], check=True)
        trace = REPO_ROOT / "external/TRACE"
        if not (trace / ".git").exists():
            trace.parent.mkdir(exist_ok=True)
            subprocess.run(["git", "clone", "https://github.com/BeyonderXX/TRACE.git", str(trace)], check=True)
        revision = "462e39f616134f4f819efeb3baea8638c03c7db4"
        subprocess.run(["git", "-C", str(trace), "fetch", "origin", revision], check=True)
        subprocess.run(["git", "-C", str(trace), "checkout", "--detach", revision], check=True)
    else:
        archive = REPO_ROOT / "data/downloads/TRACE_data.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            python = str(REPO_ROOT / ".venv-ouro/bin/python")
            subprocess.run([python, "-m", "pip", "install", "gdown"], check=True)
            subprocess.run([str(REPO_ROOT / ".venv-ouro/bin/gdown"),
                            "https://drive.google.com/file/d/1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV/view",
                            "--output", str(archive)], check=True)
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError(f"Unsafe archive path: {name}")
            bundle.extractall(REPO_ROOT / "data/TRACE")
        print("TRACE archive SHA256:", hashlib.sha256(archive.read_bytes()).hexdigest())
