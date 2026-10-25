"""Prepare, run and summarize no-replay baselines under the fixed TRACE protocol."""
from datetime import datetime
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

from scripts.parameters import REPO_ROOT, training_config
from scripts.pbs_launch import arguments
from scripts.coupled_statistics import read_complete, summarize
from scripts.study_statistics import PROTOCOL_KEYS

CODE_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("osft", "olora", "ewc", "lora")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def verify(root):
    hashes = json.loads((root / "source_manifest.json").read_text())
    for base, key in [(root / "code_snapshot", "snapshot_sha256"), (root, "study_sha256")]:
        for name, digest in hashes[key].items():
            if hashlib.sha256((base / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Frozen input changed: {base / name}")
    for source in json.loads((root / "data_manifest.json").read_text()):
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Dataset changed: {source['path']}")


def prepare(root, method):
    root = root.resolve()
    if method not in METHODS or not root.is_relative_to((REPO_ROOT / "runs").resolve()):
        raise ValueError("Expected a supported method and a new runs/ study")
    root.mkdir(exist_ok=False)
    for name in ("logs", "settings", "configs", "reference_configs", "code_snapshot"):
        (root / name).mkdir()
    snapshot = root / "code_snapshot"
    for name in ("scripts", "ouro_pipeline", "tests", "pbs"):
        shutil.copytree(CODE_ROOT / name, snapshot / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    for name in ("AGENTS.md", "README.md"):
        shutil.copy2(REPO_ROOT / name, snapshot / name)
    parameter_path = snapshot / "scripts/parameters.py"
    parameter_path.write_text(parameter_path.read_text().replace(
        "REPO_ROOT = Path(__file__).resolve().parents[1]", f"REPO_ROOT = Path({str(REPO_ROOT)!r})", 1))
    origin = REPO_ROOT / "runs/20261024_joint_local128_5seeds"
    data = json.loads((origin / "data_manifest.json").read_text())
    save(root / "data_manifest.json", data)
    base = json.loads((origin / "settings/seed42.json").read_text())
    jobs = []
    for seed in range(42, 47):
        output = root / method / f"seed{seed}"
        settings = dict(base, method=method, seed=seed, output_dir=str(output), basis_strategy="append",
            checkpoint_scope="all", lora_rank=8, lora_alpha=32., lora_dropout=.1,
            olora_rank=8, olora_alpha=32., olora_dropout=.1, olora_orthogonality_weight=.5,
            olora_norm_weight=0., ewc_lambda=400., ewc_fisher_samples=64, osft_tolerance=1e-4)
        result = subprocess.run([sys.executable, str(snapshot / "scripts/experiment.py"),
                                 *arguments("train", settings), "--dry-run"],
                                capture_output=True, text=True, check=True)
        from argparse import Namespace
        config = training_config(Namespace(**json.loads(result.stdout)))
        save(root / "settings" / f"seed{seed}.json", settings)
        save(root / "configs" / f"seed{seed}.json", config)
        (root / "logs" / f"seed{seed}.dryrun.json").write_text(result.stdout)
        jobs.append(dict(name=f"{method}_seed{seed}", seed=seed, method=method, run_dir=str(output),
            settings=str(root / "settings" / f"seed{seed}.json"),
            resolved_config=str(root / "configs" / f"seed{seed}.json"), status="prepared"))
    joint = json.loads((REPO_ROOT / "runs/20261019_joint128_5seeds/statistics.json").read_text())
    full = REPO_ROOT / "runs/20261017_fixed_epochs_mb8_depth_study/full/seed42/summary.json"
    references = [("full", 42, full)] + [("joint128", entry["seed"], Path(entry["summary"])) for entry in joint["sources"]]
    baselines = []
    for name, seed, path in references:
        summary, complete, digest = read_complete(path)
        if not complete:
            raise ValueError(f"Incomplete reference: {path}")
        config_path = root / "reference_configs" / f"{name}_seed{seed}.json"
        save(config_path, summary["config"])
        baselines.append(dict(method=name, seed=seed, run_dir=str(path.parent), summary_sha256=digest,
                              resolved_config=str(config_path)))
    full_config = json.loads((root / "reference_configs/full_seed42.json").read_text())
    for entry in jobs:
        config = json.loads(Path(entry["resolved_config"]).read_text())
        if any(config.get(key) != full_config.get(key) for key in PROTOCOL_KEYS):
            raise ValueError("Public training/evaluation protocol differs from completed Full")
    save(root / "submission.json", dict(status="prepared", method=method, jobs=jobs, baselines=baselines,
        seeds=list(range(42, 47)), created_at_jst=now(), replay=False,
        nodes_per_job=5, gpus_per_job=5, walltime_per_job="48:00:00",
        graceful_pause_after_seconds=170100, segments=2 if method == "osft" else 1,
        evaluation="all eight checkpoints, full test of seen tasks, 36 cells per seed",
        full_reused=True, score_scale="0-100; accuracy/ROUGE x100; similarity/SARI unchanged", sd_ddof=1))
    for name in ("worker", "compare", "submit"):
        call = {"worker": "worker(root, int(sys.argv[1]))", "compare": "compare(root)",
                "submit": "submit(root, verify_only=sys.argv[1:] == ['--verify'])"}[name]
        (root / f"{name}.py").write_text(
            "from pathlib import Path\nimport sys\nroot=Path(__file__).resolve().parent\n"
            "sys.path.insert(0,str(root/'code_snapshot'))\n"
            f"from scripts.matched_baselines import {name}\n{call}\n")
    shutil.copy2(CODE_ROOT / "scripts/parallel_seeds.py", root / "parallel_seeds.py")
    pbs = (CODE_ROOT / "pbs/train_seeds_parallel.pbs").read_text().replace("walltime=24:00:00", "walltime=48:00:00")
    (root / "train_parallel.pbs").write_text(pbs)
    subprocess.run(["bash", "-n", str(root / "train_parallel.pbs")], check=True)
    validation = REPO_ROOT / "reports/20261024_baseline_validation.json"
    evidence = json.loads(validation.read_text())
    if evidence["status"] != "passed":
        raise ValueError("Code validation has not passed")
    save(root / "pre_submission_validation.json", dict(status="passed", validated_at_jst=now(),
        source_validation=str(validation), method=method, seeds=list(range(42, 47)),
        shared_protocol_matches_full=True, source_checks=evidence, data_files_verified=len(data)))
    immutable = [p for p in root.rglob("*") if p.is_file() and "code_snapshot" not in p.parts
                 and p.name != "submission.json" and p.suffix != ".log"]
    save(root / "source_manifest.json", dict(
        snapshot_sha256={str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in snapshot.rglob("*") if p.is_file() and "__pycache__" not in p.parts},
        study_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in immutable}))
    verify(root)
    print(f"Prepared {method}: five independent seeds, one five-GPU allocation; Full seed42 reused", flush=True)


def validate_summary(summary, config, data):
    if summary["config"] != config or config["checkpoint_scope"] != "all" or config["eval_split"] != "test" or config["eval_limit"] is not None:
        raise ValueError("Run differs from its frozen full-test configuration")
    hashes = {(row["task"], row["split"]): row["sha256"] for row in data}
    if {(row["task"], row["split"]): row["sha256"] for row in summary["data_sources"]} != hashes:
        raise ValueError("Dataset provenance differs")
    counts = {row["task"]: row["samples"] for row in data if row["split"] == "test"}
    if len(summary["training"]) != 8 or len(summary["stages"]) != 8:
        raise ValueError("Incomplete training stages")
    for task, report in zip(config["tasks"], summary["training"]):
        epochs = config["epochs_by_task"][task]
        micro = math.ceil(5000 / config["micro_batch_size"])
        if (report["task"] != task or report["samples"] != 5000 or report["epochs_requested"] != epochs
                or report["micro_steps"] != micro * epochs
                or report["optimizer_steps"] != math.ceil(micro / config["gradient_accumulation"]) * epochs):
            raise ValueError("Fixed-epoch training budget differs")
    if any(result["samples"] != counts[task] for row in summary["evaluations"].values() for task, result in row.items()):
        raise ValueError("Evaluation is not full test")
    if summary.get("tf32_allowed") is not False:
        raise ValueError("Projection/optimizer precision differs")


def compare(root):
    if root.resolve().is_relative_to((REPO_ROOT / "result").resolve()):
        raise ValueError("New statistics must write outside result/")
    manifest = json.loads((root / "submission.json").read_text())
    data = json.loads((root / "data_manifest.json").read_text())
    with (root / ".comparison.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, refs, sources, statuses = [], {}, [], []
        common = None
        for entry in [*manifest["baselines"], *manifest["jobs"]]:
            path = Path(entry["run_dir"]) / "summary.json"
            summary, complete, digest = read_complete(path)
            reference = "summary_sha256" in entry
            if reference and (not complete or digest != entry["summary_sha256"]):
                raise ValueError(f"Reference changed: {path}")
            if not reference:
                statuses.append(dict(seed=entry["seed"], job_id=entry.get("job_id"),
                    training_status=summary.get("training_status", "not_started"), full_test_complete=complete, summary=str(path)))
            if not complete:
                continue
            config = json.loads(Path(entry["resolved_config"]).read_text())
            validate_summary(summary, config, data)
            protocol = {key: config.get(key) for key in PROTOCOL_KEYS}
            if common is not None and common != protocol:
                raise ValueError("Shared method protocol differs")
            common = protocol
            row = dict(seed=entry["seed"], **summarize(summary))
            if reference:
                refs.setdefault(entry["method"], []).append(row)
            else:
                rows.append(row)
            sources.append(dict(method=entry["method"], seed=entry["seed"], summary=str(path), sha256=digest,
                config=config, data_sources=summary["data_sources"], status="complete",
                test_samples={r["task"]: r["samples"] for r in data if r["split"] == "test"},
                recovery_history=summary.get("recovery_history", []),
                method_protocol=summary.get(f"{entry['method']}_protocol")))
        metrics = [*common["tasks"], "final_macro_100", "immediate_macro_100", "bwt_pp", "forgetting_from_best_pp"]
        def stats(records):
            return {key: dict(mean=statistics.mean(r[key] for r in records),
                sample_sd=statistics.stdev(r[key] for r in records) if len(records) > 1 else None)
                for key in metrics} if records else {}
        paired = {}
        for name, records in refs.items():
            lookup = {row["seed"]: row for row in records}
            paired[name] = [dict(seed=row["seed"], **{key: row[key] - lookup[row["seed"]][key] for key in metrics})
                            for row in rows if row["seed"] in lookup]
        payload = dict(status="complete" if len(rows) == 5 else "pending", method=manifest["method"],
            updated_at_jst=now(), n=len(rows), completed_seeds=[r["seed"] for r in rows], sd_ddof=1,
            score_scale=manifest["score_scale"], split="test", statuses=statuses, per_seed=rows, statistics=stats(rows),
            references={name: dict(n=len(records), per_seed=records, statistics=stats(records)) for name, records in refs.items()},
            paired_deltas=paired, paired_delta_statistics={name: dict(n=len(records), statistics=stats(records)) for name, records in paired.items()},
            paired_delta_direction="candidate minus reference for matching seeds only", sources=sources)
        save(root / "statistics.json", payload)
        if rows:
            with (root / "scores.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print(f"{manifest['method']}: {len(rows)}/5 full-test seeds complete", flush=True)
        return payload


def worker(root, index):
    from scripts.parameters import parse_args
    manifest = json.loads((root / "submission.json").read_text())
    entry = manifest["jobs"][index]
    status_path = root / "logs" / f"{entry['name']}.status.json"
    state = dict(seed=entry["seed"], name=entry["name"], job_id=os.environ.get("PBS_JOBID"),
                 host=socket.gethostname(), status="running", started_at_jst=now())
    save(status_path, state)
    started = time.time()
    try:
        verify(root)
        settings = json.loads(Path(entry["settings"]).read_text())
        path = Path(entry["run_dir"]) / "summary.json"
        prior, complete, _ = read_complete(path)
        if complete:
            state["status"] = "already_complete"
            return
        if prior.get("training_status") == "failed":
            raise RuntimeError("Existing training failed; do not conceal it by retrying")
        if path.exists():
            settings["resume"] = True
        if training_config(parse_args(arguments("train", settings))) != json.loads(Path(entry["resolved_config"]).read_text()):
            raise ValueError("Frozen method configuration changed")
        env = dict(os.environ)
        for key, value in {"OURO_RECOVERY_ENABLED": "1", "OURO_RECOVERY_DEADLINE": str(started + manifest["graceful_pause_after_seconds"])}.items():
            env[key] = value; env["APPTAINERENV_" + key] = value
        result = subprocess.run([sys.executable, str(root / "code_snapshot/scripts/experiment.py"),
                                 *arguments("train", settings)], env=env, cwd=REPO_ROOT)
        summary, complete, _ = read_complete(path)
        if complete and result.returncode == 0:
            state["status"] = "complete"
        elif summary.get("training_status") == "paused":
            checkpoint = Path(entry["run_dir"]) / "recovery" / f"{summary['current_training_stage']['task']}.pt"
            if not checkpoint.is_file() or not checkpoint.stat().st_size:
                raise RuntimeError("Paused without a recovery checkpoint")
            state["status"] = "paused"
        else:
            raise RuntimeError(f"Training/evaluation exited {result.returncode} without completion or safe pause")
    except BaseException as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        state["finished_at_jst"] = now()
        save(status_path, state)
        compare(root)


def submit(root, verify_only=False):
    verify(root)
    if json.loads((root / "pre_submission_validation.json").read_text())["status"] != "passed":
        raise ValueError("Submission validation has not passed")
    manifest = json.loads((root / "submission.json").read_text())
    compare(root)
    if verify_only:
        print(f"Verified {manifest['method']}: five GPUs, seeds42-46, Full reused", flush=True)
        return
    allocations = manifest.setdefault("allocations", [])
    for segment in range(manifest["segments"]):
        if len(allocations) > segment:
            continue
        if manifest.get("submission_uncertain"):
            raise RuntimeError("Inspect scheduler before retrying an uncertain submission")
        command = ["qsub", "-N", f"cmp_{manifest['method']}{segment}", "-o", str(root / "logs" / f"group{segment}.pbs.log"),
                   "-v", f"SEED_STUDY_ROOT={root}"]
        if segment:
            command += ["-W", f"depend=afterany:{allocations[segment - 1]['job_id']}"]
        command += [str(root / "train_parallel.pbs")]
        manifest["submission_uncertain"] = True
        save(root / "submission.json", manifest)
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
        job_id = result.stdout.strip()
        if result.returncode or not job_id or not job_id.split(".")[0].isdigit():
            raise RuntimeError(f"qsub response uncertain: {result.stdout!r} {result.stderr!r}")
        allocations.append(dict(segment=segment, job_id=job_id, command=command, submitted_at_jst=now()))
        manifest.update(status="submitted", submission_uncertain=False)
        if segment == 0:
            for entry in manifest["jobs"]:
                entry.update(job_id=job_id, status="submitted")
        save(root / "submission.json", manifest)
        print(f"{manifest['method']} segment{segment}: {job_id}; five GPUs, seeds42-46", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    args = parser.parse_args()
    prepare(args.prepare, args.method)
