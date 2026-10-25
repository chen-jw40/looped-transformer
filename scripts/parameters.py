"""Single parameter registry for every training and evaluation entry point."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ["FOMC", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "C-STANCE", "MeetingBank", "Py150", "20Minuten"]
# TRACE released train_seq_naive.sh / train_seq_cl.sh (commit 462e39f).
# Map by name: the upstream scripts use a different task order.
TRACE_EPOCHS = {"FOMC": 3, "ScienceQA": 3, "NumGLUE-cm": 5, "NumGLUE-ds": 5,
                "C-STANCE": 5, "MeetingBank": 7, "Py150": 5, "20Minuten": 7}
ROUTE_DEPTHS = [0, 0, 1, 1, 2, 2, 3, 3]
MAX_NEW_TOKENS = {"FOMC": 8, "ScienceQA": 1024, "NumGLUE-cm": 32, "NumGLUE-ds": 32,
                  "C-STANCE": 8, "MeetingBank": 1024, "Py150": 256, "20Minuten": 512}
PRIMARY_METRICS = {task: "similarity" if task == "Py150" else "sari" if task == "20Minuten"
                   else "rouge-L" if task == "MeetingBank" else "accuracy" for task in TASKS}
MODEL = "models/Ouro-1.4B"
DATA_ROOT = "data/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_5000"
TEST_LOOPS = 4
TEST_MAX_LENGTH = 4096
TEST_TEMPERATURE = 0.0
FINAL_CHECKPOINT = f"{len(TASKS):02d}_after_{TASKS[-1]}"
TEST_OUTPUT = Path("runs/20261005_direct_finetune_k4_5seeds_test")
TEST_SOURCES = {42: "runs/20261002_full_unconstrained_k4_vllm",
                **{seed: f"runs/20261004_full_unconstrained_k4_seeds/seed{seed}" for seed in range(43, 47)}}


def inference_options(parser):
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)


def generation_options(parser, *, inherit=False):
    parser.add_argument("--temperature", type=float, default=None if inherit else 0.0)
    parser.add_argument("--max-new-tokens-by-task", nargs="+", metavar="TASK=TOKENS",
                        help="Per-task generation caps; --max-new-tokens overrides these")


def generation_limits(config):
    return {**MAX_NEW_TOKENS, **(config.get("max_new_tokens_by_task") or {})}


def task_epochs(args, task):
    overrides = getattr(args, "epochs_by_task", None) or {}
    return overrides.get(task, args.epochs if args.epochs is not None else TRACE_EPOCHS[task])


def training_config(args):
    config = {key: value for key, value in vars(args).items()
              if key not in {"command", "dry_run", "train_only", "resume"}}
    config["output_dir"] = str(args.output_dir)
    config["primary_metrics"] = PRIMARY_METRICS
    if config.get("epochs_by_task") is None:
        config.pop("epochs_by_task", None)
    if getattr(args, "basis_statistic", "activation") == "activation":
        config.pop("basis_statistic", None)
    if getattr(args, "basis_statistic", "activation") != "drsp":
        config.pop("drsp_lambda", None)
        config.pop("drsp_sketch_width", None)
    if args.method != "ewc":
        config.pop("ewc_lambda", None)
        config.pop("ewc_fisher_samples", None)
    if getattr(args, "basis_strategy", "append") != "joint_balanced":
        config.pop("basis_balance_epsilon", None)
    if getattr(args, "basis_strategy", "append") != "last_trajectory":
        config.pop("trajectory_floor", None)
    if getattr(args, "basis_strategy", "append") == "append" and getattr(args, "basis_statistic", "activation") != "drsp":
        # Preserve legacy protocol dictionaries, including full-run resume.
        for key in ("basis_strategy", "history_rank", "basis_svd_method"):
            config.pop(key, None)
    if getattr(args, "basis_statistic", "activation") == "drsp":
        # These activation-bank knobs are unused by the DRSP collector/store.
        for key in ("history_rank", "stat_vectors_per_call", "stat_vectors_per_layer", "basis_residual_tolerance"):
            config.pop(key, None)
    if args.method == "full":
        # Do not change the saved protocol for task-boundary recovery of full runs.
        config.pop("basis_depth_mode", None)
    if args.method != "lora":
        for key in ("lora_rank", "lora_alpha", "lora_dropout"):
            config.pop(key, None)
    if args.method in {"osft", "olora", "ewc", "lora"}:
        for key in ("basis_depth_mode", "route_depths", "projection", "projection_scope", "stat_samples",
                    "stat_vectors_per_call", "stat_vectors_per_layer", "basis_max_rank",
                    "basis_residual_tolerance"):
            config.pop(key, None)
        if args.method == "osft":
            config["osft_retention_ratios"] = [i / len(args.tasks) for i in range(len(args.tasks))]
        elif args.method == "olora":
            config["olora_target_modules"] = ["q_proj", "v_proj"]
        elif args.method == "lora":
            config["lora_target_modules"] = ["q_proj", "v_proj"]
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unified eight-task Ouro experiments (PBS + Python)")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train for fixed per-task epochs, then score on test")
    train.add_argument("--method", choices=("full", "projected", "osft", "olora", "ewc", "lora"), default="full")
    train.add_argument("--lora-rank", type=int, default=8)
    train.add_argument("--lora-alpha", type=float, default=32.)
    train.add_argument("--lora-dropout", type=float, default=.1)
    train.add_argument("--ewc-lambda", type=float, default=400.0)
    train.add_argument("--ewc-fisher-samples", type=int, default=64)
    train.add_argument("--olora-rank", type=int, default=8)
    train.add_argument("--olora-alpha", type=float, default=32.0)
    train.add_argument("--olora-dropout", type=float, default=0.1)
    train.add_argument("--olora-orthogonality-weight", type=float, default=0.5)
    train.add_argument("--olora-norm-weight", type=float, default=0.0)
    train.add_argument("--osft-tolerance", type=float, default=1e-4,
                       help="Maximum final relative SVD-factor orthogonality residual")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--resume", action="store_true",
                       help="Resume full/activation/DRSP projection; restore an optimizer checkpoint when recovery is enabled")
    train.add_argument("--model", default=MODEL)
    train.add_argument("--data-root", default=DATA_ROOT)
    train.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    train.add_argument("--eval-tasks", nargs="+", choices=TASKS)
    train.add_argument("--loops", type=int, default=4)
    train.add_argument("--eval-loops", type=int)
    train.add_argument("--train-limit", type=int, default=5000)
    train.add_argument("--eval-limit", type=int)
    train.add_argument("--eval-split", choices=("eval", "test"), default="test")
    train.add_argument("--checkpoint-scope", choices=("final", "all"),
                       help="Defaults to all for projected/OSFT/O-LoRA/EWC, final for full")
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--micro-batch-size", type=int, default=8)
    train.add_argument("--gradient-accumulation", type=int, default=16)
    train.add_argument("--epochs", type=int,
                       help="Uniform epoch override; default uses TRACE's per-task 3/5/7 epochs")
    train.add_argument("--epochs-by-task", nargs="+", metavar="TASK=EPOCHS",
                       help="Per-task positive integer overrides, taking precedence over --epochs")
    train.add_argument("--max-length", type=int, default=4096)
    train.add_argument("--max-new-tokens", type=int)
    train.add_argument("--truncate-long-prompts", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument("--early-stopping-steps", type=int, default=150)
    train.add_argument("--early-stopping-max-evals", type=int, default=6)
    train.add_argument("--early-stopping-samples", type=int, default=128)
    train.add_argument("--early-stopping-seed", type=int, default=42)
    train.add_argument("--early-stopping-patience", type=int, default=2)
    train.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    train.add_argument("--projection-scope", choices=("all-linear", "o-proj"), default="all-linear")
    train.add_argument("--basis-depth-mode", choices=("all", "task", "last"),
                       help="all: separate depth banks (default); last: final loop only; "
                            "task: legacy routed history. "
                            "Explicit --route-depths without a mode selects task for compatibility")
    train.add_argument("--route-depths", type=int, nargs="+",
                       help="One history sampling depth per task, only for basis-depth-mode=task")
    train.add_argument("--stat-samples", type=int, default=64)
    train.add_argument("--stat-vectors-per-call", type=int, default=4)
    train.add_argument("--stat-vectors-per-layer", type=int, default=256,
                       help="Maximum activation vectors per physical linear module per depth")
    train.add_argument("--basis-max-rank", type=int, default=128,
                       help="append: cap per depth bank; joint strategies: cap per module's final joint basis")
    train.add_argument("--basis-residual-tolerance", type=float, default=0.05)
    train.add_argument("--drsp-lambda", type=float, default=.5)
    train.add_argument("--drsp-sketch-width", type=int, default=8)
    train.add_argument("--basis-statistic", choices=("activation", "depth_coupled", "depth_diagonal", "drsp"),
                       default="activation", help="activation/depth_*: post-task history; drsp: sampled normal-training backwards with fixed-rank spectral merge")
    train.add_argument("--basis-strategy", choices=("append", "joint_greedy", "joint_local", "joint_pca", "joint_balanced", "last_trajectory"), default="append",
                       help="append: permanent depth banks; joint_*: reselect historical spectra; "
                            "joint_local: source-depth gains with global candidate selection; "
                            "last_trajectory: preserve old basis and last coverage, select paired depth differences")
    train.add_argument("--trajectory-floor", choices=("tolerance", "reference"), default="tolerance",
                       help="last_trajectory: retain last's residual tolerance or its measured reference coverage")
    train.add_argument("--history-rank", type=int, default=512,
                       help="Per-depth historical spectrum rank for joint strategies")
    train.add_argument("--basis-balance-epsilon", type=float, default=.01,
                       help="Coverage offset in the balanced log-coverage objective, in (0,1]")
    train.add_argument("--basis-svd-method", choices=("exact", "randomized"), default="randomized")
    train.add_argument("--env-file", default=".env")
    train.add_argument("--wandb-project", default="ouro-continual-learning")
    train.add_argument("--wandb-entity")
    train.add_argument("--wandb-run-name")
    train.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    train.add_argument("--train-only", action="store_true")
    inference_options(train)
    generation_options(train)

    evaluate = sub.add_parser("evaluate", help="Evaluate saved checkpoints; new split uses a separate output")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument("--split", choices=("eval", "test"))
    evaluate.add_argument("--checkpoint-scope", choices=("final", "all"), default="final")
    evaluate.add_argument("--model", help="Tokenizer/base model override; full checkpoint weights stay unchanged")
    evaluate.add_argument("--data-root")
    evaluate.add_argument("--tasks", nargs="+", choices=TASKS)
    evaluate.add_argument("--eval-loops", type=int)
    evaluate.add_argument("--eval-limit", type=int)
    evaluate.add_argument("--max-length", type=int)
    evaluate.add_argument("--max-new-tokens", type=int)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--truncate-long-prompts", action=argparse.BooleanOptionalAction, default=None)
    inference_options(evaluate)
    generation_options(evaluate, inherit=True)

    base = sub.add_parser("base", help="Evaluate an untouched model on all eight tasks")
    base.add_argument("--model", default=MODEL)
    base.add_argument("--data-root", default=DATA_ROOT)
    base.add_argument("--output-dir", type=Path, required=True)
    base.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    base.add_argument("--split", choices=("eval", "test"), default="test")
    base.add_argument("--limit", type=int)
    base.add_argument("--loop-depth", type=int, default=4)
    base.add_argument("--max-length", type=int, default=4096)
    base.add_argument("--max-new-tokens", type=int)
    base.add_argument("--seed", type=int, default=42)
    base.add_argument("--truncate-long-prompts", action=argparse.BooleanOptionalAction, default=True)
    inference_options(base)
    generation_options(base)

    test = sub.add_parser("test-seeds", help="Test the five existing direct fine-tuning runs")
    test.add_argument("--seed", type=int, choices=range(42, 47), required=True)
    test.add_argument("--output-root", type=Path, default=TEST_OUTPUT)
    test.add_argument("--prepare-only", action="store_true")
    test.add_argument("--source-run", type=Path, help="Override the historical source directory for this seed")
    test.add_argument("--model", help="Tokenizer path; defaults to the source run")
    test.add_argument("--data-root", help="Dataset root; defaults to the source run")
    test.add_argument("--loop-depth", type=int, default=TEST_LOOPS)
    test.add_argument("--max-length", type=int, default=TEST_MAX_LENGTH)
    test.add_argument("--max-new-tokens", type=int)
    test.add_argument("--truncate-long-prompts", action=argparse.BooleanOptionalAction, default=True)
    inference_options(test)
    generation_options(test)
    summarize = sub.add_parser("summarize", help="Compute test means and sample standard deviations")
    summarize.add_argument("--output-root", type=Path, default=TEST_OUTPUT)
    summarize.add_argument("--if-ready", action="store_true")
    study = sub.add_parser("summarize-study", help="Compare matched full/OSFT runs on all-stage full test")
    study.add_argument("--output-root", type=Path, required=True)
    study.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    study.add_argument("--if-ready", action="store_true")
    study.add_argument("--methods", nargs="+", choices=("full", "projected", "osft", "olora", "ewc", "lora"),
                       default=["full", "osft"])
    depth_study = sub.add_parser("summarize-depth-controls", help="Report matched single-seed depth controls")
    depth_study.add_argument("--output-root", type=Path, required=True)
    recompute = sub.add_parser("recompute", help="Rescore saved predictions without inference")
    recompute.add_argument("run_dirs", nargs="+", type=Path)
    validate = sub.add_parser("validate", help="Validate eight-task data or GPU/model execution")
    validate.add_argument("--kind", choices=("data", "model", "gpu"), default="data")
    validate.add_argument("--data-root", default=DATA_ROOT)
    validate.add_argument("--model", default=MODEL)
    validate.add_argument("--loops", type=int, nargs="+", default=[1, 2, 4])
    validate.add_argument("--prompt", default="The future of AI is")
    validate.add_argument("--atol", type=float, default=0.05)
    validate.add_argument("--rtol", type=float, default=0.05)
    setup = sub.add_parser("setup", help="Provision environments or fetch pinned inputs")
    setup.add_argument("action", choices=("cpu", "gpu", "assets", "data"))
    worker = sub.add_parser("worker", help="Internal worker invoked by the unified entry point")
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--checkpoint", required=True)
    worker.add_argument("--loop-depth", type=int, required=True)
    worker.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    inference_options(worker)
    report = sub.add_parser("report", help="Internal worker invoked by the unified entry point")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--checkpoint", required=True)
    for command in sub.choices.values():
        command.add_argument("--dry-run", action="store_true", help="Print resolved settings; no execution or writes")
    args = parser.parse_args(argv)
    if args.command == "summarize-study" and (len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Study statistics require at least two distinct seeds")
    if args.command == "summarize-study" and len(set(args.methods)) != len(args.methods):
        parser.error("Study methods must be distinct")
    positive = ["loops", "eval_loops", "loop_depth", "train_limit", "eval_limit", "limit", "micro_batch_size",
                "gradient_accumulation", "epochs", "max_length", "max_new_tokens", "batch_size",
                "stat_samples", "stat_vectors_per_call", "stat_vectors_per_layer", "basis_max_rank",
                "early_stopping_steps", "early_stopping_max_evals", "early_stopping_samples", "early_stopping_patience",
                "olora_rank", "history_rank", "ewc_fisher_samples", "drsp_sketch_width"]
    for name in positive:
        value = getattr(args, name, None)
        values = value if isinstance(value, list) else [value]
        if any(v is not None and v <= 0 for v in values):
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "gpu_memory_utilization") and not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0,1]")
    if getattr(args, "max_length", None) is not None and args.max_length < 2:
        parser.error("--max-length must be at least 2")
    if getattr(args, "temperature", None) is not None and (not math.isfinite(args.temperature) or args.temperature < 0):
        parser.error("--temperature must be finite and nonnegative")
    if getattr(args, "max_new_tokens_by_task", None) is not None:
        caps = {}
        for item in args.max_new_tokens_by_task:
            try:
                task, value = item.split("=", 1)
                tokens = int(value)
            except ValueError:
                parser.error("--max-new-tokens-by-task requires TASK=positive_integer")
            if task not in TASKS or task in caps or tokens <= 0:
                parser.error(f"Invalid or duplicate generation cap: {item}")
            caps[task] = tokens
        args.max_new_tokens_by_task = {**MAX_NEW_TOKENS, **caps}
    if hasattr(args, "tasks") and args.tasks is not None and len(args.tasks) != len(set(args.tasks)):
        parser.error("Tasks must be distinct")
    if args.command == "train":
        epoch_overrides = {}
        for item in args.epochs_by_task or []:
            try:
                task, value = item.split("=", 1)
                epochs = int(value)
            except ValueError:
                parser.error("--epochs-by-task requires TASK=positive_integer")
            if task not in args.tasks or task in epoch_overrides or epochs <= 0:
                parser.error(f"Invalid or duplicate task epoch override: {item}")
            epoch_overrides[task] = epochs
        args.epochs_by_task = ({task: epoch_overrides.get(task, args.epochs if args.epochs is not None
                                                       else TRACE_EPOCHS[task]) for task in args.tasks}
                               if args.epochs is None or epoch_overrides else None)
        if not math.isfinite(args.drsp_lambda) or args.drsp_lambda < 0:
            parser.error("--drsp-lambda must be finite and nonnegative")
        if not math.isfinite(args.basis_balance_epsilon) or not 0 < args.basis_balance_epsilon <= 1:
            parser.error("--basis-balance-epsilon must be in (0,1]")
        if args.resume and not (args.method in {"full", "osft", "olora", "ewc", "lora"} or (args.method == "projected"
                and ((args.basis_statistic == "activation" and args.basis_strategy in {"append", "joint_greedy", "joint_local", "joint_pca", "joint_balanced"})
                     or (args.basis_statistic == "drsp" and args.basis_strategy == "append")))):
            parser.error("--resume supports full, activation-based append/joint projection, and DRSP")
        args.checkpoint_scope = args.checkpoint_scope or ("all" if args.method in {"projected", "osft", "olora", "ewc", "lora"} else "final")
        if args.lora_rank <= 0 or not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0 or not 0 <= args.lora_dropout < 1:
            parser.error("Invalid sequential LoRA rank, alpha or dropout")
        if not math.isfinite(args.ewc_lambda) or args.ewc_lambda < 0:
            parser.error("EWC lambda must be finite and nonnegative")
        args.eval_tasks = args.eval_tasks or list(args.tasks)
        args.eval_loops = args.eval_loops or args.loops
        args.learning_rate = args.learning_rate if args.learning_rate is not None else 1e-5
        args.basis_depth_mode = args.basis_depth_mode or ("task" if args.route_depths is not None else "all")
        if args.basis_statistic != "activation" and (args.method != "projected"
                or args.basis_depth_mode != "all" or args.basis_strategy != "append"
                or args.route_depths is not None):
            parser.error("Depth gradient statistics require projected, all depths, append, and no task routes")
        if args.basis_strategy == "last_trajectory":
            if (args.method != "projected" or args.basis_depth_mode != "all" or args.loops < 2
                    or args.route_depths is not None or args.basis_statistic != "activation"):
                parser.error("Last-anchored trajectories require projected, all depths, K>=2 and activation statistics")
        elif args.basis_strategy != "append":
            if args.method != "projected" or args.basis_depth_mode == "task" or args.route_depths is not None:
                parser.error("Joint basis strategies require projected with all or last depths and no task routes")
        if (args.basis_depth_mode == "task" or args.method == "full") and args.route_depths is None:
            args.route_depths = [ROUTE_DEPTHS[TASKS.index(t)] for t in args.tasks]
        args.eval_seen_only = True
        if len(set(args.tasks)) != len(args.tasks) or not set(args.eval_tasks) <= set(args.tasks):
            parser.error("Tasks must be distinct; eval tasks must be a subset of training tasks")
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
            parser.error("Learning rate must be finite and positive; weight decay finite and nonnegative")
        if not math.isfinite(args.early_stopping_min_delta) or args.early_stopping_min_delta < 0:
            parser.error("Early stopping min delta must be finite and nonnegative")
        if not math.isfinite(args.osft_tolerance) or not 0 < args.osft_tolerance < 1:
            parser.error("OSFT tolerance must be finite and in (0,1)")
        if not math.isfinite(args.olora_alpha) or args.olora_alpha <= 0:
            parser.error("O-LoRA alpha must be finite and positive")
        if not math.isfinite(args.olora_dropout) or not 0 <= args.olora_dropout < 1:
            parser.error("O-LoRA dropout must be in [0,1)")
        if any(not math.isfinite(v) or v < 0 for v in (args.olora_orthogonality_weight, args.olora_norm_weight)):
            parser.error("O-LoRA regularization weights must be finite and nonnegative")
        if not 0 <= args.basis_residual_tolerance <= 1:
            parser.error("Invalid basis residual tolerance")
        if args.route_depths is not None and len(args.route_depths) != len(args.tasks):
            parser.error("Provide one route depth per task")
        if args.method == "projected":
            if args.basis_depth_mode != "task" and args.route_depths is not None:
                parser.error("--route-depths requires --basis-depth-mode task")
            if args.route_depths is not None and any(d < 0 or d >= args.loops for d in args.route_depths):
                parser.error("Every route depth must satisfy 0 <= depth < K")
    return args
