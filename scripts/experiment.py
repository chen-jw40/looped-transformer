#!/usr/bin/env python3
"""The only executable Python entry point for the Ouro experiment pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.parameters import REPO_ROOT, parse_args
from scripts.runtime import run_in_container


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(arguments)
    if args.dry_run:
        print(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2))
        return
    os.chdir(REPO_ROOT)
    if args.command == "train":
        if os.environ.get("OURO_RUNTIME") != "training":
            run_in_container("training", [*arguments, "--train-only"])
            if not args.train_only:
                from ouro_pipeline.evaluation import run
                run(parse_args(["evaluate", "--run-dir", str(args.output_dir),
                                "--checkpoint-scope", args.checkpoint_scope,
                                "--batch-size", str(args.batch_size),
                                "--gpu-memory-utilization", str(args.gpu_memory_utilization)]))
            return
        from ouro_pipeline.full_training import run
        run(args)
    elif args.command in {"worker", "base", "report"}:
        runtime = "training" if args.command == "report" else "vllm"
        if os.environ.get("OURO_RUNTIME") != runtime:
            run_in_container(runtime, arguments)
            return
        if args.command == "worker":
            from ouro_pipeline.vllm_worker import run
        elif args.command == "base":
            from ouro_pipeline.base_evaluation import run
        else:
            from scripts.checkpoint_reporting import run
        run(args)
    elif args.command == "evaluate":
        from ouro_pipeline.evaluation import run
        run(args)
    elif args.command == "test-seeds":
        from scripts.test_runs import run
        run(args)
        if not args.prepare_only:
            from scripts.test_statistics import run as summarize
            summarize(parse_args(["summarize", "--output-root", str(args.output_root), "--if-ready"]))
    elif args.command == "summarize":
        from scripts.test_statistics import run
        run(args)
    elif args.command == "summarize-study":
        from scripts.study_statistics import run
        run(args)
    elif args.command == "summarize-depth-controls":
        from scripts.depth_controls_statistics import run
        run(args.output_root)
    elif args.command in {"recompute", "validate"}:
        needs_gpu = args.command == "validate" and args.kind in {"model", "gpu"}
        needs_training = args.command == "recompute" or needs_gpu
        if needs_training and os.environ.get("OURO_RUNTIME") != "training":
            run_in_container("training", arguments)
            return
        from scripts.maintenance import run
        run(args)
    elif args.command == "setup":
        from scripts.setup import run
        run(args)


if __name__ == "__main__":
    main()
