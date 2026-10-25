"""Five independent seeds in one five-node PBS allocation; no gradient synchronization."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def study_root(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root / "code_snapshot"))
    from scripts.parameters import REPO_ROOT
    if not root.is_relative_to((REPO_ROOT / "runs").resolve()):
        raise ValueError("Parallel seed studies must use runs/")
    manifest = json.loads((root / "submission.json").read_text())
    if [entry["seed"] for entry in manifest["jobs"]] != list(range(42, 47)):
        raise ValueError("Expected independent seeds42-46 in order")
    return root, manifest


def allocation_nodes():
    nodes = Path(os.environ["PBS_NODEFILE"]).read_text().splitlines()
    if len(nodes) != 5 or len({node.split(".")[0] for node in nodes}) != 5:
        raise ValueError("Expected five distinct nodes with one process slot each")
    return nodes


def dispatch(root):
    root, manifest = study_root(root)
    from scripts.runtime import apptainer_executable
    nodes = allocation_nodes()
    job_id = os.environ["PBS_JOBID"]
    directory = root / "allocations"
    directory.mkdir(exist_ok=True)
    allocation_path = directory / f"{job_id}.json"
    if allocation_path.exists():
        raise FileExistsError(allocation_path)
    allocation = dict(nodes=nodes, job_id=job_id, seeds=[entry["seed"] for entry in manifest["jobs"]],
                      started_at_utc=datetime.now(timezone.utc).isoformat())
    save(allocation_path, allocation)
    path = str(Path(apptainer_executable()).parent) + os.pathsep + os.environ.get("PATH", os.defpath)
    command = ["pbsdsh", "-c", "5", "--", "/usr/bin/env", f"PATH={path}",
               "PYTHONUNBUFFERED=1", "OMP_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=4",
               "/usr/bin/python3", str(Path(__file__).resolve()), "--worker", str(root),
               "--allocation", str(allocation_path)]
    result = subprocess.run(command)
    states = []
    for entry in manifest["jobs"]:
        status_path = root / "logs" / f"{entry['name']}.status.json"
        state = json.loads(status_path.read_text()) if status_path.exists() else {"status": "missing"}
        if state.get("job_id") != job_id:
            state = {"status": "missing_from_current_allocation", "seed": entry["seed"]}
        states.append(state)
    accepted = {"complete", "already_complete", "paused"}
    passed = result.returncode == 0 and all(state["status"] in accepted for state in states)
    status = "paused" if passed and any(state["status"] == "paused" for state in states) else "complete" if passed else "failed"
    save(directory / f"{job_id}.status.json", dict(status=status, pbsdsh_returncode=result.returncode, workers=states))
    if not passed:
        raise RuntimeError("One or more seed workers failed or did not report; see allocations/ and logs/")


def worker(root, allocation_path):
    root, manifest = study_root(root)
    allocation = json.loads(Path(allocation_path).read_text())
    host = socket.gethostname().split(".")[0]
    nodes = [node.split(".")[0] for node in allocation["nodes"]]
    if nodes.count(host) != 1:
        raise ValueError("Worker host is not uniquely assigned to this allocation")
    index = nodes.index(host)
    if allocation["seeds"] != [entry["seed"] for entry in manifest["jobs"]]:
        raise ValueError("Allocation seed mapping changed")
    entry = manifest["jobs"][index]
    env = dict(os.environ, PBS_JOBID=allocation["job_id"], CUDA_VISIBLE_DEVICES="0")
    # pbsdsh workers need no inherited PBS_NODEFILE or MPI rank environment.
    log = root / "logs" / f"{entry['name']}.{allocation['job_id']}.log"
    with log.open("x", buffering=1) as handle:
        result = subprocess.run(["/usr/bin/python3", str(root / "worker.py"), str(index)],
                                env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"seed{entry['seed']} exited {result.returncode}; see {log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dispatch", type=Path)
    mode.add_argument("--worker", type=Path)
    parser.add_argument("--allocation", type=Path)
    args = parser.parse_args()
    if args.dispatch is not None:
        dispatch(args.dispatch)
    else:
        if args.allocation is None:
            parser.error("Workers require --allocation")
        worker(args.worker, args.allocation)
