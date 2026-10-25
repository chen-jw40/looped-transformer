import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import parallel_seeds


def plan(root):
    (root / "logs").mkdir()
    return {"jobs": [{"seed": seed, "name": f"seed{seed}"} for seed in range(42, 47)]}


@pytest.mark.parametrize("bad", [False, True])
def test_five_seed_dispatch_checks_current_worker_status_and_resource_mapping(tmp_path, bad):
    manifest = plan(tmp_path)
    for entry in manifest["jobs"]:
        (tmp_path / "logs" / f"{entry['name']}.status.json").write_text(json.dumps(
            {"status": "complete", "job_id": "old" if bad and entry["seed"] == 45 else "new.opbs"}))
    with patch.object(parallel_seeds, "study_root", return_value=(tmp_path, manifest)), \
         patch.object(parallel_seeds, "allocation_nodes", return_value=[f"n{i}" for i in range(5)]), \
         patch("scripts.runtime.apptainer_executable", return_value="/site/bin/apptainer"), \
         patch.dict(os.environ, {"PBS_JOBID": "new.opbs"}), \
         patch.object(parallel_seeds.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
        if bad:
            with pytest.raises(RuntimeError, match="failed"):
                parallel_seeds.dispatch(tmp_path)
        else:
            parallel_seeds.dispatch(tmp_path)
        command = run.call_args.args[0]
        assert command[:5] == ["pbsdsh", "-c", "5", "--", "/usr/bin/env"]
        assert "--worker" in command and command[5].startswith("PATH=/site/bin:")
    allocation = json.loads((tmp_path / "allocations/new.opbs.json").read_text())
    assert allocation["seeds"] == list(range(42, 47))


def test_worker_maps_host_to_one_seed_without_distributed_training(tmp_path):
    manifest = plan(tmp_path)
    allocation = tmp_path / "allocation.json"
    allocation.write_text(json.dumps({"nodes": [f"n{i}.domain" for i in range(5)],
                                      "seeds": list(range(42, 47)), "job_id": "123.opbs"}))
    with patch.object(parallel_seeds, "study_root", return_value=(tmp_path, manifest)), \
         patch.object(parallel_seeds.socket, "gethostname", return_value="n3.domain"), \
         patch.object(parallel_seeds.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
        parallel_seeds.worker(tmp_path, allocation)
        assert run.call_args.args[0][-1] == "3"
        assert run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert run.call_args.kwargs["env"]["PBS_JOBID"] == "123.opbs"
    assert (tmp_path / "logs/seed45.123.opbs.log").exists()


def test_reject_duplicate_nodes_in_five_gpu_allocation(tmp_path):
    nodes = tmp_path / "nodes"
    nodes.write_text("n1\nn1\nn2\nn3\nn4\n")
    with patch.dict(os.environ, {"PBS_NODEFILE": str(nodes)}), pytest.raises(ValueError, match="distinct"):
        parallel_seeds.allocation_nodes()
