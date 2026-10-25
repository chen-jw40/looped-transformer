"""Regression coverage for the failed PBS startup, without scheduling jobs."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import drsp_study, runtime


@pytest.mark.parametrize("returncode,states", [
    (0, ["complete", "failed", "complete"]),
    (0, ["complete", "complete", None]),
    (1, ["complete"] * 3),
])
def test_dispatch_checks_worker_status_even_when_pbsdsh_returns_zero(tmp_path, returncode, states):
    for rank, status in zip((64, 128, 512), states):
        if status:
            drsp_study.write_json(tmp_path / f"worker_rank{rank}.json", {"status": status})
    with patch.object(drsp_study, "output_root", return_value=tmp_path), \
         patch.object(drsp_study, "allocation_nodes", return_value=["n1", "n2", "n3"]), \
         patch.object(runtime, "apptainer_executable", return_value="/site/bin/apptainer"), \
         patch.dict(os.environ, {"PBS_JOBID": "smoke.opbs"}), \
         patch.object(drsp_study.subprocess, "run", return_value=SimpleNamespace(returncode=returncode)), \
         patch.object(drsp_study, "summarize") as summarize:
        with pytest.raises(RuntimeError, match="workers failed"):
            drsp_study.dispatch(tmp_path)
        summarize.assert_not_called()
    assert json.loads((tmp_path / "dispatch_status.json").read_text())["status"] == "failed"


def test_smoke_dispatch_has_explicit_executable_path_and_no_formal_summary(tmp_path):
    for rank in (64, 128, 512):
        drsp_study.write_json(tmp_path / f"worker_rank{rank}.json", {"status": "complete"})
    with patch.object(drsp_study, "output_root", return_value=tmp_path), \
         patch.object(drsp_study, "allocation_nodes", return_value=["n1", "n2", "n3"]), \
         patch.object(runtime, "apptainer_executable", return_value="/site/bin/apptainer"), \
         patch.dict(os.environ, {"PBS_JOBID": "smoke.opbs"}), \
         patch.object(drsp_study.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, \
         patch.object(drsp_study, "summarize") as summarize:
        drsp_study.dispatch(tmp_path, smoke_only=True)
        command = run.call_args.args[0]
        assert command[:5] == ["pbsdsh", "-c", "3", "--", "/usr/bin/env"]
        assert command[5].startswith("PATH=/site/bin:")
        assert command[-1] == "--smoke-only"
        summarize.assert_not_called()
    allocation = json.loads((tmp_path / "allocation.json").read_text())
    assert allocation["nodes"] == ["n1", "n2", "n3"]
    assert allocation["job_id"] == "smoke.opbs"
    assert json.loads((tmp_path / "smoke_summary.json").read_text())["status"] == "passed"


def test_remote_worker_needs_no_pbs_nodefile_and_smoke_never_launches_formal_run(tmp_path):
    (tmp_path / "logs").mkdir()
    drsp_study.write_json(tmp_path / "allocation.json", {
        "nodes": ["n1", "n2", "n3"], "job_id": "smoke.opbs", "smoke_only": True})
    command = ["apptainer", "exec", str(drsp_study.CODE_ROOT / "scripts/experiment.py")]
    with patch.object(drsp_study, "output_root", return_value=tmp_path), \
         patch.dict(os.environ, {"PATH": os.defpath}, clear=True), \
         patch.object(drsp_study.socket, "gethostname", return_value="n2.domain"), \
         patch.object(drsp_study.os, "dup2"), \
         patch.object(drsp_study.os, "chdir"), \
         patch.object(drsp_study.subprocess, "run", return_value=SimpleNamespace(stdout="GH200")) as run, \
         patch.object(runtime, "container_command", return_value=(command, {})), \
         patch("scripts.pbs_launch.launch") as launch:
        drsp_study.worker(tmp_path, smoke_only=True)
        launch.assert_not_called()
        assert run.call_count == 2  # inventory and bounded GPU validation
    state = json.loads((tmp_path / "worker_rank128.json").read_text())
    assert state["job_id"] == "smoke.opbs" and state["status"] == "complete"


def test_snapshot_container_uses_frozen_code_for_training_and_evaluation(tmp_path):
    with patch.object(runtime, "CODE_ROOT", tmp_path), \
         patch.object(runtime, "apptainer_executable", return_value="/site/bin/apptainer"), \
         patch.object(Path, "is_file", return_value=True):
        for kind in ("training", "vllm"):
            command, env = runtime.container_command(kind, [])
            assert command[-1] == str(tmp_path / "scripts/experiment.py")
            assert str(tmp_path) in env["APPTAINERENV_PYTHONPATH"].split(":")


def test_result_and_existing_studies_are_not_smoke_destinations(tmp_path):
    from scripts.parameters import REPO_ROOT
    with pytest.raises(ValueError, match="runs/"):
        drsp_study.prepare_smoke(REPO_ROOT / "result" / "unauthorized")
    with patch.object(drsp_study, "output_root", return_value=tmp_path):
        with pytest.raises(FileExistsError):
            drsp_study.prepare_smoke(tmp_path)
