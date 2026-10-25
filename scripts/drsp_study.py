"""Three independent DRSP ranks in one PBS allocation, with frozen settings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import hashlib
import shutil
import socket
import subprocess
import sys
import traceback

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def allocation_nodes():
    nodes = Path(os.environ["PBS_NODEFILE"]).read_text().splitlines()
    if len(nodes) != 3 or len(set(n.split(".")[0] for n in nodes)) != 3:
        raise RuntimeError(f"Expected exactly 3 nodes with one process slot each; got {nodes}")
    return nodes


def output_root(root):
    from scripts.parameters import REPO_ROOT
    root = root.resolve()
    if not root.is_relative_to((REPO_ROOT / "runs").resolve()):
        raise ValueError("DRSP outputs must use runs/")
    return root


def prepare_smoke(root):
    """Freeze a separate, bounded smoke allocation; never edit a failed study."""
    from scripts.parameters import REPO_ROOT
    root = output_root(root)
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root / "code_snapshot"
    snapshot.mkdir()
    for name in ("scripts", "ouro_pipeline", "pbs"):
        shutil.copytree(CODE_ROOT / name, snapshot / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    parameters = snapshot / "scripts/parameters.py"
    parameters.write_text(parameters.read_text().replace(
        "REPO_ROOT = Path(__file__).resolve().parents[1]",
        f"REPO_ROOT = Path({str(REPO_ROOT)!r})", 1))
    (root / "logs").mkdir()
    (root / "configs").mkdir()
    for rank in (64, 128, 512):
        settings = dict(method="projected", basis_statistic="drsp", basis_depth_mode="all",
            basis_max_rank=rank, basis_svd_method="randomized", drsp_lambda=.5, drsp_sketch_width=8,
            stat_samples=2, output_dir=str(root / f"rank{rank}" / "seed42"),
            tasks=["FOMC", "ScienceQA"], seed=42, loops=4, eval_loops=4,
            train_limit=2, epochs=1, learning_rate=1e-5, weight_decay=0., micro_batch_size=1,
            gradient_accumulation=2, max_length=512, truncate_long_prompts=True,
            early_stopping=False, train_only=True, wandb_mode="disabled")
        write_json(root / "configs" / f"rank{rank}.json", {"settings": settings})
    write_json(root / "source_manifest.json", {"sha256": {
        str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(snapshot.rglob("*")) if p.is_file()}})
    write_json(root / "smoke_plan.json", {"kind": "smoke_only", "walltime": "00:10:00",
        "ranks": [64, 128, 512], "native_model_pipeline_rank": 64,
        "tasks": ["FOMC", "ScienceQA"], "train_samples_per_task": 2,
        "formal_experiment": False, "evaluation": "not run"})
    shutil.copy2(snapshot / "pbs/smoke_drsp.pbs", root / "smoke_drsp.pbs")
    print(root)


def dispatch(root, smoke_only=False):
    from scripts.runtime import apptainer_executable
    root = output_root(root)
    nodes = allocation_nodes()
    # Persist the allocation on shared storage. Remote PBS_NODEFILE and PBS_JOBID
    # are not guaranteed by pbsdsh (including task zero on the mother node).
    allocation = {"nodes": nodes, "job_id": os.environ["PBS_JOBID"], "smoke_only": smoke_only}
    allocation_path = root / "allocation.json"
    if allocation_path.exists():
        raise FileExistsError(allocation_path)
    executable = apptainer_executable()
    write_json(allocation_path, allocation)
    path = str(Path(executable).parent) + os.pathsep + os.environ.get("PATH", os.defpath)
    command = ["pbsdsh", "-c", "3", "--", "/usr/bin/env", f"PATH={path}",
               "/usr/bin/python3", str(CODE_ROOT / "scripts/drsp_study.py"),
               "--worker", str(root)]
    if smoke_only:
        command.append("--smoke-only")
    print(json.dumps({"allocation": allocation, "command": command}), flush=True)
    result = subprocess.run(command)
    states = {}
    for rank in (64, 128, 512):
        status = root / f"worker_rank{rank}.json"
        states[str(rank)] = json.loads(status.read_text()) if status.exists() else {"status": "missing"}
    passed = result.returncode == 0 and all(s["status"] == "complete" for s in states.values())
    write_json(root / "dispatch_status.json", {"status": "complete" if passed else "failed",
        "pbsdsh_returncode": result.returncode, "smoke_only": smoke_only, "workers": states})
    # Some PBS versions report remote failures in stderr but return zero.
    if not passed:
        raise RuntimeError("DRSP workers failed or are missing; see dispatch_status.json and logs/")
    if smoke_only:
        write_json(root / "smoke_summary.json", {"status": "passed", **allocation,
            "workers": states, "formal_experiment": False})
    else:
        summarize(root)


def gpu_smoke(output, rank):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from ouro_pipeline.drsp import DRSPCollector, DRSPBasisStore
    from ouro_pipeline.modeling import final_exit_loss, set_loop_count
    from ouro_pipeline.projected_full import HistoricalFullWeightProjection
    from scripts.parameters import REPO_ROOT
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(REPO_ROOT / "models/Ouro-1.4B", trust_remote_code=True)
    config.hidden_size, config.intermediate_size = 64, 96
    config.num_hidden_layers, config.layer_types = 1, ["full_attention"]
    config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 16
    config.vocab_size, config.max_position_embeddings = 128, 64
    config._attn_implementation, config.use_cache = "sdpa", False
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).float().cuda().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    set_loop_count(model, 4)
    batch = {"input_ids": torch.tensor([[1, 3, 4, 5, 6, 2]], device="cuda"),
             "attention_mask": torch.ones(1, 6, dtype=torch.long, device="cuda"),
             "labels": torch.tensor([[-100, -100, -100, 5, 6, 2]], device="cuda")}
    observer = DRSPCollector(model, 4, samples=1, width=8, strength=.5)
    observer.configure(1)
    try:
        observer.begin(0, batch, 16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = final_exit_loss(model, batch, 4)
        observer.before_backward()
        (loss / 16).backward()
        observer.finish()
        tensors = observer.tensors()
        # Actual requested rank and native-size down_proj input dimension exercise
        # the randomized spectral compression and eigenvalue-weighted merge.
        store = DRSPBasisStore(rank)
        native = torch.randn(rank * 2, 5632, generator=torch.Generator().manual_seed(43))
        store.update({"native_shape": native}, device="cuda")
        store.update({"native_shape": native * .5}, device="cuda")
        assert store.bases["native_shape"].shape == (5632, rank)
        torch.testing.assert_close(store.bases["native_shape"].T @ store.bases["native_shape"], torch.eye(rank), atol=1e-4, rtol=1e-4)
        tiny = DRSPBasisStore(min(rank, 8))
        tiny.update(tensors, device="cuda")
        assert tiny.bases["early_exit_gate"].shape[1] == 0
        guard = HistoricalFullWeightProjection(model, {}, joint_bases=tiny.bases)
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            second, _ = final_exit_loss(model, batch, 4)
        second.backward()
        optimizer.step()
        guard.after_step()
        diagnostics = guard.diagnostics()
        assert diagnostics["protected_linear_weights"] == 8
        assert diagnostics["max_WU_history_scaled_error"] < 1e-5
        write_json(output, {"status": "passed", "rank": rank, "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "checkpointed_bf16_forward_backward": True, "native_dimension_compression": [5632, rank],
            "projection": diagnostics, "statistics": observer.diagnostics()})
    finally:
        observer.close()


def worker(root, smoke_only=False):
    from scripts.parameters import REPO_ROOT
    from scripts.pbs_launch import launch
    from scripts.runtime import container_command
    root = output_root(root)
    allocation = json.loads((root / "allocation.json").read_text())
    if allocation["smoke_only"] != smoke_only:
        raise ValueError("Worker mode differs from frozen allocation")
    nodes = allocation["nodes"]
    host = socket.gethostname().split(".")[0]
    index = [node.split(".")[0] for node in nodes].index(host)
    rank = (64, 128, 512)[index]
    log = root / "logs" / f"rank{rank}.log"
    with log.open("x", buffering=1) as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        payload = {"rank": rank, "host": host, "job_id": allocation["job_id"], "smoke_only": smoke_only,
                   "started_at": datetime.now(timezone.utc).isoformat(), "status": "running", "log": str(log)}
        status = root / f"worker_rank{rank}.json"
        try:
            os.chdir(REPO_ROOT)
            # One physical GPU per node; keep nested vLLM's device selection local.
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["PBS_JOBID"] = allocation["job_id"]
            os.environ.setdefault("OMP_NUM_THREADS", "8")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
            payload["gpu_inventory"] = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"],
                check=True, capture_output=True, text=True).stdout.strip()
            write_json(status, payload)
            print(json.dumps(payload), flush=True)
            command, env = container_command("training", ["train", "--basis-statistic", "drsp"])
            entry = command.index(str(CODE_ROOT / "scripts/experiment.py"))
            command[entry:] = [str(Path(__file__).resolve()), "--gpu-smoke", str(root / f"gpu_validation_rank{rank}.json"), "--rank", str(rank)]
            subprocess.run(command, env=env, check=True)
            # All ranks exercise their real native-dimension spectral merge.
            # Only rank64 loads pretrained weights for the two-task pipeline.
            if not smoke_only or rank == 64:
                settings = json.loads((root / "configs" / f"rank{rank}.json").read_text())["settings"]
                if smoke_only:
                    if (settings["tasks"] != ["FOMC", "ScienceQA"] or settings["train_limit"] != 2
                            or settings["epochs"] != 1 or settings["max_length"] != 512
                            or settings["stat_samples"] != 2 or settings["early_stopping"]
                            or not settings["train_only"]):
                        raise ValueError("Unbounded smoke training configuration")
                launch("train", settings)
                if smoke_only:
                    summary = json.loads((Path(settings["output_dir"]) / "summary.json").read_text())
                    if summary["training_status"] != "complete" or len(summary["stages"]) != 2:
                        raise RuntimeError("Smoke pipeline did not finish both tasks")
                    diagnostics = summary["training"][1]["projection_diagnostics"]
                    if diagnostics["protected_linear_weights"] == 0 or diagnostics["max_WU_history_scaled_error"] >= 1e-5:
                        raise RuntimeError("Smoke history projection failed")
                    payload["pipeline_summary"] = str(Path(settings["output_dir"]) / "summary.json")
            payload["status"] = "complete"
        except BaseException as error:
            payload.update(status="failed", error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
            raise
        finally:
            payload["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(status, payload)


def summarize(root):
    import csv
    from scripts.coupled_statistics import read_complete, signature, summarize as scores
    manifest = json.loads((root / "submission.json").read_text())
    rows, sources, common = [], [], None
    for entry in manifest["experiments"]:
        path = Path(entry["run_dir"]) / "summary.json"
        summary, complete, digest = read_complete(path)
        if not complete:
            raise RuntimeError(f"Incomplete full-test run: {path}")
        config = summary["config"]
        if config != json.loads((root / "configs" / f"rank{entry['rank']}.json").read_text())["resolved"]:
            raise ValueError(f"Configuration drift: {path}")
        current = signature(summary)
        if common is not None and current != common:
            raise ValueError("Different training/evaluation protocols or data hashes")
        common = current
        counts = {d["task"]: d["samples"] for d in manifest["data_manifest"] if d["split"] == "test"}
        for evaluations in summary["evaluations"].values():
            if any(result["samples"] != counts[task] for task, result in evaluations.items()):
                raise ValueError("Partial test evaluation")
        if any(r["new_rank"] > entry["rank"] for s in summary["stages"] for r in s["basis_updates"]):
            raise ValueError("DRSP exceeded requested rank")
        rows.append({"rank": entry["rank"], "seed": 42, "lambda": .5, **scores(summary)})
        sources.append({"summary": str(path), "sha256": digest, "rank": entry["rank"], "status": "complete"})
    write_json(root / "comparison.json", {"status": "complete", "score_scale": "0-100; accuracy/ROUGE x100, Py150 similarity/SARI unchanged",
        "seed": 42, "n_per_rank": 1, "sample_sd_ddof1": None, "scores": rows, "sources": sources})
    with (root / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare-smoke", "dispatch", "worker", "check-allocation", "gpu-smoke", "summarize"):
        group.add_argument("--" + name, type=Path)
    parser.add_argument("--rank", type=int, choices=(64, 128, 512))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_only and not (args.dispatch or args.worker):
        parser.error("--smoke-only requires --dispatch or --worker")
    if args.prepare_smoke:
        prepare_smoke(args.prepare_smoke)
    elif args.dispatch:
        dispatch(args.dispatch.resolve(), args.smoke_only)
    elif args.check_allocation:
        print(json.dumps({"nodes": allocation_nodes(), "ranks": [64, 128, 512]}), flush=True)
    elif args.worker:
        worker(args.worker.resolve(), args.smoke_only)
    elif args.gpu_smoke:
        gpu_smoke(args.gpu_smoke, args.rank)
    else:
        summarize(args.summarize.resolve())
