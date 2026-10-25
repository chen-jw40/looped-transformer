"""Data validation, model checks and rescoring behind the unified CLI."""
import json
from pathlib import Path
import subprocess

from scripts.parameters import TASKS


def run(args):
    if args.command == "recompute":
        from ouro_pipeline.metrics import evaluate_task
        from ouro_pipeline.evaluation_io import save_summary
        from ouro_pipeline.training import continual_metrics, continual_aggregates

        for root in args.run_dirs:
            summary = json.loads((root / "summary.json").read_text())
            for checkpoint, results in summary["evaluations"].items():
                for task, old in results.items():
                    rows = [json.loads(line) for line in (root / "predictions" / checkpoint / f"{task}.jsonl").open()]
                    metrics = evaluate_task(task, [r["prediction"] for r in rows],
                                            [r["reference"] for r in rows], sources=[r["prompt"] for r in rows])
                    for key in ("prompt_overflow", "truncated_prompts", "lengths", "prompt_truncation", "eval_loops"):
                        if key in old:
                            metrics[key] = old[key]
                    results[task] = metrics
            matrix, forgetting = continual_metrics(summary["evaluations"], summary["config"]["tasks"],
                                                   summary["config"].get("primary_metrics"))
            summary.update(performance_matrix=matrix, forgetting=forgetting,
                           continual_aggregates=continual_aggregates(matrix, forgetting, summary["config"]["tasks"]),
                           metrics_recomputed_from_predictions=True)
            save_summary(summary, root)
        return
    if args.kind == "data":
        report = {}
        for task in TASKS:
            report[task] = {}
            for split in ("train", "eval", "test"):
                path = Path(args.data_root) / task / f"{split}.json"
                rows = json.loads(path.read_text())
                if not isinstance(rows, list) or not rows:
                    raise ValueError(f"{path}: expected a nonempty list")
                for index, row in enumerate(rows):
                    if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in ("prompt", "answer")):
                        raise ValueError(f"{path}: invalid row {index}")
                report[task][split] = len(rows)
        print(json.dumps({"complete": True, "tasks": report}, indent=2))
    elif args.kind == "model":
        from scripts.model_validation import run as validate_model
        validate_model(args)
    else:
        import torch

        subprocess.run(["nvidia-smi"], check=True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        value = torch.tensor([1.0], device="cuda") * 2
        print(json.dumps({"cuda": torch.version.cuda, "gpu_count": torch.cuda.device_count(),
                          "device": torch.cuda.get_device_name(), "tensor_check": value.item()}))
