"""Shared protocol for unconstrained and depth-aware projected full fine-tuning."""
from __future__ import annotations

import json
import hashlib
import random
import shutil
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ouro_pipeline.data import AnswerOnlyCollator, PromptAnswerDataset, fixed_subset, read_split
from ouro_pipeline.modeling import load_ouro, load_tokenizer, set_loop_count
from ouro_pipeline.projected_full import HistoricalFullWeightProjection, iter_full_linears
from ouro_pipeline.subspace import ACTIVATION_SAMPLING_PROTOCOL, BasisStore, FullDepthActivationCollector
from ouro_pipeline.adaptive_subspace import AdaptiveBasisStore, BALANCED_SELECTORS, SELECTORS
from ouro_pipeline.trajectory_subspace import TrajectoryBasisStore
from ouro_pipeline.drsp import DRSPBasisStore, DRSPCollector


from scripts.parameters import training_config

@torch.inference_mode()
def collect_history(model, tokenizer, records, args, task, depth):
    from ouro_pipeline.training import move, prompt_truncation

    selected = fixed_subset(records, args.stat_samples, args.seed)
    loader = DataLoader(
        PromptAnswerDataset(selected), batch_size=1, shuffle=False,
        collate_fn=AnswerOnlyCollator(tokenizer, args.max_length,
                                     prompt_truncation=prompt_truncation(args, task)),
    )
    set_loop_count(model, args.loops)
    model.eval()
    special_token_ids = set(getattr(tokenizer, "all_special_ids", []))
    special_token_ids.update(getattr(tokenizer, name, None)
                             for name in ("bos_token_id", "eos_token_id", "pad_token_id"))
    with FullDepthActivationCollector(model, args.stat_vectors_per_layer,
                                      args.stat_vectors_per_call, args.projection_scope,
                                      target_depth=depth) as collector:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for batch in loader:
                batch = move(batch, "cuda")
                collector.set_batch(batch["input_ids"], batch["attention_mask"], special_token_ids)
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                      labels=None, use_cache=False, exit_at_step=args.loops - 1, logits_to_keep=1)
        activations = collector.tensors()
    depths = range(args.loops) if depth is None else [depth]
    expected = {f"{key}@ut{d}" for key, _ in iter_full_linears(model, scope=args.projection_scope)
                for d in depths}
    missing = expected - activations.keys()
    if not expected or missing:
        raise RuntimeError(f"{task}: missing history activations: {sorted(missing)}")
    return activations, len(selected)


def run(args):
    projected = args.method == "projected"
    adaptive = projected and args.basis_strategy in SELECTORS
    trajectory = projected and args.basis_strategy == "last_trajectory"
    drsp = projected and getattr(args, "basis_statistic", "activation") == "drsp"
    coupled = projected and getattr(args, "basis_statistic", "activation") in {"depth_coupled", "depth_diagonal"}
    # Match the verified microbatch smoke: strict FP32 optimizer/projection,
    # with BF16 forward/backward autocast. Use the same setting for full.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    osft = args.method == "osft"
    olora = args.method == "olora"
    lora = args.method == "lora"
    adapters = olora or lora
    ewc = args.method == "ewc"
    if args.output_dir.resolve().is_relative_to(Path("result").resolve()):
        raise ValueError("New training experiments must write to runs/, not result/")
    resuming = getattr(args, "resume", False)
    restored_summary, completed, model_path = None, 0, args.model
    if resuming:
        from scripts.training_resume import resume_plan
        restored_summary, completed, model_path = resume_plan(args)
        if completed == len(args.tasks):
            print("All training tasks are complete; continuing to evaluation", flush=True)
            return
    elif args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Full fine-tuning requires a CUDA GPU with BF16 autocast support")
    from ouro_pipeline.training import init_wandb, train_task, write_summary

    args.output_dir.mkdir(parents=True, exist_ok=resuming)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    run = init_wandb(args, args.output_dir)
    tokenizer = load_tokenizer(args.model)
    # FP32 master parameters keep the task-entry WU invariant numerically stable;
    # BF16 autocast handles the expensive forward and backward matrix products.
    model = load_ouro(args.model if adapters else model_path, args.loops, "cuda", dtype=torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Full fine-tuning requires every model parameter to be trainable")
    olora_controller = None
    ewc_controller = None
    if ewc:
        from ouro_pipeline.ewc import ContinualEWC
        ewc_controller = ContinualEWC(model, args.ewc_lambda)
        if completed:
            ewc_controller.load(args.output_dir / "ewc" / f"{completed:02d}_after_{args.tasks[completed - 1]}.pt")
    if adapters:
        from ouro_pipeline.olora import ContinualOLoRA
        if lora:
            from ouro_pipeline.lora import ContinualLoRA
            olora_controller = ContinualLoRA(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
        else:
            olora_controller = ContinualOLoRA(model, args.olora_rank, args.olora_alpha,
                                             args.olora_dropout, args.olora_orthogonality_weight,
                                             args.olora_norm_weight)
        if completed:
            state = torch.load(args.output_dir / "adapters" / f"{completed:02d}_after_{args.tasks[completed - 1]}.pt",
                               map_location="cpu", weights_only=True)
            olora_controller.load_adapter_state(state)
    bases = (AdaptiveBasisStore(args.history_rank, args.basis_strategy, args.basis_svd_method,
                                args.basis_balance_epsilon)
             if adaptive else TrajectoryBasisStore(args.basis_svd_method, args.trajectory_floor)
             if trajectory else DRSPBasisStore(args.basis_max_rank, args.basis_svd_method)
             if drsp else BasisStore() if projected else None)
    summary = {
        "config": training_config(args),
        "method": ("multi-task diagonal empirical EWC" if ewc else
                   "Sequential LoRA with one fixed-rank q/v adapter" if lora else
                   "O-LoRA with incremental frozen adapters and input-A overlap regularization" if olora else
                   "OSFT with task-stage weight-SVD budgets" if osft else
                   "DRSP: trajectory call-gradient mean and variation, fixed-rank spectral protection" if drsp else
                   f"full fine-tuning with {args.basis_statistic} gradient-sketch protection" if coupled else
                   "full fine-tuning with last-anchored depth trajectory protection" if trajectory else
                   f"full fine-tuning with {args.basis_strategy} historical protection" if adaptive else
                   "full fine-tuning with final-depth activation projection" if projected and args.basis_depth_mode == "last" else
                   "full fine-tuning with all-depth activation projection" if projected and args.basis_depth_mode == "all" else
                   "full fine-tuning with task-depth activation projection" if projected else "unconstrained full fine-tuning"),
        "projection_scope": args.projection_scope if projected else None,
        "precision": "FP32 parameters and AdamW states; BF16 autocast; gradient checkpointing",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_allowed": torch.backends.cuda.matmul.allow_tf32,
        "training": [], "stages": [], "evaluations": {},
        "evaluation_engine": "vLLM (Python experiment evaluate entry)",
        "wandb_run_id": run.id if run is not None else None,
        "training_status": "running",
    }
    if lora:
        from ouro_pipeline.lora import PAPER_URL, REFERENCE_URL
        summary["lora_protocol"] = dict(paper=PAPER_URL, reference_code=REFERENCE_URL,
            variant="one fixed-rank q/v adapter shared across tasks and loops",
            rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"], reinitialize_between_tasks=False,
            historical_adapters=False, regularization=False, replay=False,
            optimizer_reset_per_task=True, checkpoint_format="dense export plus unmerged adapter state")
    if ewc:
        from ouro_pipeline.ewc import PAPER_URL, REFERENCE_URL, REFERENCE_COMMIT
        summary["ewc_protocol"] = {
            "paper": PAPER_URL, "reference_code": REFERENCE_URL, "reference_commit": REFERENCE_COMMIT,
            "variant": "multi-task diagonal empirical Fisher; all historical task anchors",
            "objective": "CE + lambda/2 * sum_k sum_i F_ki * (theta_i - theta_ki)^2",
            "lambda": args.ewc_lambda, "fisher_samples_per_task": args.ewc_fisher_samples,
            "fisher": "mean of per-example squared gradients of mean answer-token CE",
            "fisher_split": "train", "fisher_timing": ("after best eval-CE weights restored" if args.early_stopping else "after final fixed-epoch weights"),
            "fisher_dropout": "disabled; activation checkpointing enabled",
            "regularization_storage": "sum Fisher and Fisher-weighted historical center; parameter-independent constant omitted",
            "reference_difference": "TRACE updates Fisher during training and uses latest anchor; this uses post-task Fisher and all task anchors",
            "optimizer_reset_per_task": True, "replay": False, "shared_across_loops": True,
            "validation_metric": "answer-token CE only, without regularization",
            "adaptation": "Ouro K=4/TRACE and existing project training protocol; not the original paper benchmark",
            "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name()},
        }
    if olora:
        from ouro_pipeline.olora import PAPER_URL, REFERENCE_URL, REFERENCE_COMMIT
        summary["olora_protocol"] = {
            "paper": PAPER_URL, "reference_code": REFERENCE_URL, "reference_commit": REFERENCE_COMMIT,
            "variant": "released code: input-side A, sum(abs(A_old @ A_new.T))",
            "norm_penalty": "sum of unsquared Frobenius norms of current A/B",
            "rank_per_task": args.olora_rank, "alpha": args.olora_alpha,
            "dropout": args.olora_dropout, "orthogonality_weight": args.olora_orthogonality_weight,
            "norm_weight": args.olora_norm_weight, "target_modules": ["q_proj", "v_proj"],
            "shared_across_loops": True, "base_and_historical_factors_frozen": True,
            "optimizer_reset_per_task": True, "replay": False,
            "validation_metric": "answer-token CE only, without regularization",
            "checkpoint_format": "dense cumulative weights plus separate historical adapter factors",
            "adaptation": "Ouro K=4/TRACE; training hyperparameters recorded in config; differs from original training setup",
            "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name()},
        }
    if osft:
        from ouro_pipeline.osft import StageOSFT, PAPER_URL, REFERENCE_URL, REFERENCE_COMMIT
        summary["osft_protocol"] = {
            "paper": PAPER_URL, "reference_code": REFERENCE_URL, "reference_commit": REFERENCE_COMMIT,
            "variant": "Appendix A.10 task-stage budget; local Ouro adaptation",
            "rank_schedule": "floor((stage - 1) / total_tasks * min(weight.shape))",
            "forward": "FP32 dense reconstruction; BF16 linear autocast",
            "svd_driver": "gesvd", "float32_matmul_precision": "highest; TF32 disabled for OSFT",
            "gradient_projection": True, "post_adamw_parameter_projection": True,
            "loop_depth_specific": False, "optimizer_reset_per_task": True,
            "checkpoint_format": "dense Ouro weights after restoring best eval-loss factors",
            "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name()},
        }
    summary["data_sources"] = [{"task": task, "split": split,
                                "path": str((Path(args.data_root) / task / f"{split}.json").resolve()),
                                "sha256": hashlib.sha256((Path(args.data_root) / task / f"{split}.json").read_bytes()).hexdigest()}
                               for task in args.tasks for split in ("train", "eval", "test")]
    if resuming:
        if summary["data_sources"] != restored_summary.get("data_sources"):
            raise ValueError("Recovery data sources differ from the original experiment")
        backup = args.output_dir / f"summary.before_resume.{time.time_ns()}.json"
        shutil.copy2(args.output_dir / "summary.json", backup)
        summary = restored_summary
        summary["training_status"] = "running"
        summary.pop("training_error", None)
        summary.setdefault("recovery_history", []).append({
            "resumed_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": model_path, "completed_tasks": completed,
            "restart_task": args.tasks[completed], "summary_backup": str(backup),
            "mode": ("optimizer boundary; restore model/AdamW/RNG/sampler/projection" if
                     (args.output_dir / "recovery" / f"{args.tasks[completed]}.pt").exists()
                     else "task boundary; interrupted task restarted with a fresh AdamW"),
            "rng": "saved RNG at optimizer boundary when available; otherwise original seed and per-task DataLoader generator",
        })
        if projected and completed:
            basis_path = args.output_dir / "bases" / f"{completed:02d}_after_{args.tasks[completed-1]}.pt"
            bases = (DRSPBasisStore.load(basis_path) if drsp else
                     AdaptiveBasisStore.load(basis_path) if adaptive else BasisStore.load(basis_path))
        print(f"Resuming full training after {completed} completed tasks from {model_path}; "
              f"restarting {args.tasks[completed]}", flush=True)
    if projected:
        summary["projection_protocol"] = {
            "basis_depth_mode": args.basis_depth_mode,
            "basis_axes": ["physical_linear_module", "loop_depth"],
            "history_update": "append residual directions; preserve existing basis",
            "basis_max_rank_per_bank": args.basis_max_rank,
            "stat_vectors_per_module_per_depth": args.stat_vectors_per_layer,
            "constraint": "W U = W_task_entry U; U is the orthogonal union of all stored depths",
            "basis_max_rank_per_module_per_depth": args.basis_max_rank,
            "scope": args.projection_scope,
        }
        if adaptive:
            summary["projection_protocol"] = {
                "basis_depth_mode": args.basis_depth_mode, "basis_strategy": args.basis_strategy,
                "basis_axes": ["physical_linear_module"],
                "history_axes": ["physical_linear_module", "loop_depth"],
                "history_update": "task-equal trace-normalized second moments; truncated spectral merge; reselect directions",
                "history_rank_per_depth": args.history_rank,
                "basis_max_rank_per_module": args.basis_max_rank,
                "svd_method": args.basis_svd_method, "depth_weights": "uniform over collected depths",
                "stat_vectors_per_module_per_depth": args.stat_vectors_per_layer,
                "constraint": "W Q = W_task_entry Q; Q fixed during each task",
                "scope": args.projection_scope,
            }
            if args.basis_strategy in {"joint_greedy", "joint_local"}:
                summary["projection_protocol"].update(
                    selection_gain_scope=("source_depth_only" if args.basis_strategy == "joint_local" else "all_depths"),
                    candidate_orthogonalization="global across all depths",
                )
            if args.basis_strategy in BALANCED_SELECTORS:
                summary["projection_protocol"].update(
                    selection_objective="sum_d alpha_d * log(epsilon + depth_coverage_d)",
                    balance_epsilon=args.basis_balance_epsilon,
                    depth_weights="uniform",
                    history_update="task-equal per-depth trace-normalized second moments",
                )
        if coupled:
            summary["projection_protocol"] = {
                "basis_statistic": args.basis_statistic, "basis_depth_mode": "all",
                "basis_axes": ["physical_linear_module"],
                "basis_max_rank_per_module": args.basis_max_rank,
                "stat_vectors_per_module": args.stat_vectors_per_layer,
                "history_update": "append residual sketch directions; preserve existing basis; no replacement at capacity",
                "statistic": "z_i = sum_k (r_i^T b_ik) x_ik; uncentered second moment",
                "probe_sharing": "across loops" if args.basis_statistic == "depth_coupled" else "independent across loops",
                "calibration": "train samples, best eval-CE weights, dropout disabled, non-reentrant checkpointing",
                "loss": "mean answer-token CE per sequence; no accumulation scaling",
                "heads": "lm_head and early_exit_gate retain final-depth activation protection",
                "extra_backward_passes_per_task": args.stat_samples,
                "constraint": "W Q = W_task_entry Q; Q fixed during each task",
                "scope": args.projection_scope,
                "projection_arithmetic": "FP32 matmul; TF32 disabled for gradient-sketch variants",
                "projection_tf32_allowed": torch.backends.cuda.matmul.allow_tf32,
            }
        if trajectory:
            summary["projection_protocol"] = {
                "basis_depth_mode": "all", "basis_strategy": "last_trajectory",
                "basis_axes": ["physical_linear_module"],
                "basis_max_rank_per_module": args.basis_max_rank,
                "stat_vectors_per_module_per_depth": args.stat_vectors_per_layer,
                "statistic": "position-paired adjacent-loop activation differences; no backward",
                "selection_objective": "maximize mean normalized adjacent-difference coverage subject to last-depth floor",
                "last_floor_policy": args.trajectory_floor,
                "last_floor": "max(tolerance * last energy, same-weight last reference residual)"
                              if args.trajectory_floor == "tolerance" else "same-weight last reference residual",
                "rank": "same new rank as last/append at current weights with current historical basis",
                "solver": "candidate-subspace Lagrange eigenspace search; feasible last fallback; approximate optimum",
                "candidate_budget": "last reference additions plus at most the same number of residual difference directions",
                "history_update": "preserve all old directions; append only; no replacement at capacity",
                "heads": "lm_head and early_exit_gate retain final-depth activation protection",
                "extra_calibration_backward_passes": 0,
                "constraint": "W Q = W_task_entry Q; Q fixed during each task",
                "scope": args.projection_scope,
            }
        if drsp:
            summary["projection_protocol"] = {
                "basis_statistic": "drsp", "basis_axes": ["physical_linear_module"],
                "basis_max_rank_per_module": args.basis_max_rank,
                "lambda": args.drsp_lambda, "sketch_width": args.drsp_sketch_width,
                "statistic": "E[barG.T barG + lambda/D sum_d (G_d-barG).T (G_d-barG)]",
                "gradient": "per-example final answer-token mean CE; raw pre-projection adjoints; undo accumulation and batch token-mean scales",
                "probe": "Rademacher/sqrt(s), same output-space R for all calls per module per sampled microbatch",
                "sampling": "one example (rotating batch slot) from each of up to stat_samples evenly spaced normal SFT microbatches; early stop may reduce count",
                "max_samples_per_task": args.stat_samples,
                "history_update": "sum of task-mean second moments; merge U*sqrt(eigenvalues), truncate to k after every task",
                "trace_normalization": False, "svd_method": args.basis_svd_method,
                "training_weights": ("restore best eval-CE weights; statistics retain sampled trajectory (including later discarded steps)"
                                     if args.early_stopping else "final fixed-epoch weights; statistics retain sampled training trajectory"),
                "heads": "actual CE call gradients only; lm_head D=1; disconnected early_exit_gate has zero basis",
                "extra_backward_passes": 0,
                "constraint": "W Q = W_task_entry Q after AdamW; fixed Q within task; spectral reselection between tasks",
                "scope": args.projection_scope, "projection_tf32_allowed": torch.backends.cuda.matmul.allow_tf32,
            }
        if not coupled and not drsp:
            summary["projection_protocol"]["activation_sampling"] = dict(ACTIVATION_SAMPLING_PROTOCOL)
        projectable = {id(parameter)
                       for _, module in iter_full_linears(model, scope=args.projection_scope)
                       for parameter in (module.weight, module.bias) if parameter is not None}
        summary["projection_coverage"] = {
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "projectable_parameters": sum(parameter.numel() for parameter in model.parameters() if id(parameter) in projectable),
            "unprojected_parameter_names": [name for name, parameter in model.named_parameters()
                                            if parameter.requires_grad and id(parameter) not in projectable],
        }
    curves = {}
    write_summary(summary, args.output_dir)
    if resuming and completed and args.method in {"osft", "olora", "ewc", "lora"}:
        from ouro_pipeline.recovery import restore_task_boundary
        restore_task_boundary(args, f"{completed:02d}_after_{args.tasks[completed - 1]}")
    try:
        for stage, task in enumerate(args.tasks, 1):
            if stage <= completed:
                continue
            started = time.perf_counter()
            if osft or adapters or ewc or drsp:
                torch.cuda.reset_peak_memory_stats()
            summary["current_training_stage"] = {"stage": stage, "task": task, "status": "running"}
            write_summary(summary, args.output_dir)
            route_depth = args.route_depths[stage - 1] if projected and args.basis_depth_mode == "task" else None
            if projected and args.basis_depth_mode == "last":
                route_depth = args.loops - 1
            source_depths = (list(range(args.loops)) if route_depth is None else [route_depth]) if projected else []
            set_loop_count(model, args.loops)
            if adapters:
                olora_controller.begin_task(stage)
            controller = (HistoricalFullWeightProjection(
                model, {} if adaptive or coupled or trajectory or drsp else bases.bases, scope=args.projection_scope,
                joint_bases=bases.bases if adaptive or coupled or trajectory or drsp else None,
            ) if projected else None)
            if osft:
                controller = StageOSFT(model, stage, len(args.tasks), args.osft_tolerance)
            if projected and stage > 1 and not controller.constraints:
                raise RuntimeError(f"{task}: no historical full-weight constraints were built")
            records = read_split(args.data_root, task, "train")
            observer = (DRSPCollector(model, args.loops, args.stat_samples, args.drsp_sketch_width,
                        args.drsp_lambda, args.seed + stage, args.projection_scope) if drsp else None)
            try:
                report = train_task(model, tokenizer, records, args, task, run, curves,
                                    update_projection=controller, autocast_dtype=torch.bfloat16,
                                    optimizer_step_offset=sum(item["optimizer_steps"] for item in summary["training"]),
                                    regularizer=ewc_controller if ewc else olora_controller if olora else None,
                                    gradient_observer=observer)
                if drsp:
                    report["drsp_statistics"] = observer.diagnostics()
            finally:
                if observer is not None:
                    observer.close()
            report.update({"stage": stage, "loop_depth": args.loops, "basis_source_depth": route_depth})
            if projected:
                report["basis_source_depths"] = source_depths
            if controller is not None:
                report["osft_diagnostics" if osft else "projection_diagnostics"] = controller.diagnostics()
                if osft:
                    controller.materialize()
                del controller
            if adapters:
                report["lora_diagnostics" if lora else "olora_diagnostics"] = olora_controller.diagnostics()
                olora_controller.consolidate()
            if ewc:
                report["ewc_fisher"] = ewc_controller.collect(
                    tokenizer, records, args, task, autocast_dtype=torch.bfloat16)
            if osft or adapters or ewc or drsp:
                torch.cuda.synchronize()
                report.update(status="complete", stage_seconds=time.perf_counter() - started,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
            summary["training"].append(report)
            checkpoint = f"{stage:02d}_after_{task}"
            checkpoint_path = args.output_dir / "checkpoints" / checkpoint
            checkpoint_path.mkdir(parents=True)
            with olora_controller.dense_export() if adapters else nullcontext():
                model.save_pretrained(checkpoint_path, safe_serialization=True, max_shard_size="2GB")
            tokenizer.save_pretrained(checkpoint_path)
            # vLLM loads each full checkpoint as a standalone trust_remote_code model.
            model_source = Path(args.model)
            if model_source.is_dir():
                for source_name in ("configuration_ouro.py", "modeling_ouro.py"):
                    source = model_source / source_name
                    if source.is_file() and not (checkpoint_path / source_name).is_file():
                        shutil.copy2(source, checkpoint_path / source_name)
            stage_report = {"stage": stage, "task": task, "loop_depth": args.loops,
                            "checkpoint": str(checkpoint_path)}
            if ewc:
                state_path = args.output_dir / "ewc" / f"{checkpoint}.pt"
                ewc_controller.save(state_path)
                stage_report["ewc_state"] = str(state_path)
            if adapters:
                adapter_path = args.output_dir / "adapters" / f"{checkpoint}.pt"
                adapter_path.parent.mkdir(exist_ok=True)
                torch.save(olora_controller.adapter_state(), adapter_path)
                stage_report["adapter_factors"] = str(adapter_path)
            if projected:
                # Randomized PCA must not perturb the next task's training RNG.
                basis_started = time.perf_counter()
                basis_device = next(model.parameters()).device
                track_basis_cost = coupled or trajectory or drsp or args.basis_strategy in BALANCED_SELECTORS
                if track_basis_cost and basis_device.type == "cuda":
                    torch.cuda.synchronize(basis_device)
                    torch.cuda.reset_peak_memory_stats(basis_device)
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(args.seed + stage)
                    collection_started = time.perf_counter()
                    if drsp:
                        activations, sample_count = observer.tensors(), observer.count
                        stage_report["drsp_statistics"] = report["drsp_statistics"]
                        observer.chunks.clear()
                        del observer
                    elif coupled:
                        from ouro_pipeline.depth_coupled import collect_coupled_history
                        activations, sample_count, coupling_report = collect_coupled_history(
                            model, tokenizer, records, args, task, stage)
                        stage_report["coupling_statistics"] = coupling_report
                    else:
                        activations, sample_count = collect_history(model, tokenizer, records, args, task, route_depth)
                    if track_basis_cost and basis_device.type == "cuda":
                        torch.cuda.synchronize(basis_device)
                    collection_seconds = time.perf_counter() - collection_started
                    sampled_vectors_by_depth = {key: value.shape[0] for key, value in activations.items()}
                    if drsp:
                        updates = bases.update(activations, device=basis_device)
                    elif adaptive:
                        updates = bases.update(activations, args.basis_max_rank,
                                               device=basis_device)
                    elif trajectory:
                        updates = bases.update(activations, args.basis_max_rank, args.basis_residual_tolerance,
                                               device=basis_device)
                    else:
                        updates = bases.update(activations, args.basis_max_rank, args.basis_residual_tolerance)
                basis_path = args.output_dir / "bases" / f"{checkpoint}.pt"
                bases.save(basis_path)
                stage_report.update({"basis_source_depth": route_depth, "basis_source_depths": source_depths,
                                     "basis": str(basis_path),
                                     "statistics_samples": sample_count, "basis_updates": updates,
                                     "sampled_vectors_by_depth": sampled_vectors_by_depth,
                                     "basis_storage_bytes": bases.storage_bytes()})
                if coupled or drsp:
                    stage_report["sampled_vectors_per_module"] = stage_report.pop("sampled_vectors_by_depth")
                if adaptive:
                    stage_report["basis_storage_breakdown"] = bases.storage_breakdown()
                if track_basis_cost:
                    if basis_device.type == "cuda":
                        torch.cuda.synchronize(basis_device)
                    stage_report["basis_cost"] = {
                        "collection_seconds": collection_seconds,
                        "total_seconds": time.perf_counter() - basis_started,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(basis_device) if basis_device.type == "cuda" else None,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(basis_device) if basis_device.type == "cuda" else None,
                    }
                del activations
            stage_report["evaluation"] = "pending"
            summary["stages"].append(stage_report)
            summary["current_training_stage"]["status"] = "complete"
            if args.method in {"osft", "olora", "ewc", "lora"}:
                from ouro_pipeline.recovery import save_task_boundary
                save_task_boundary(args, checkpoint)
            write_summary(summary, args.output_dir)
            from ouro_pipeline.recovery import checkpoint_path, enabled
            if enabled():
                checkpoint_path(args, task).unlink(missing_ok=True)
        summary["training_status"] = "complete"
        write_summary(summary, args.output_dir)
    except Exception as error:
        from ouro_pipeline.recovery import TrainingPaused
        summary.update(training_status="paused" if isinstance(error, TrainingPaused) else "failed",
                       training_error=f"{type(error).__name__}: {error}")
        write_summary(summary, args.output_dir)
        raise
    finally:
        if run is not None:
            run.finish()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
