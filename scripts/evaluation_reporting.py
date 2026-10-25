"""Report continual-learning evaluations as separate W&B metric tables."""


def _stage(checkpoint):
    return int(checkpoint[:2])


def _score_100(result):
    if result is None or result["value"] is None:
        return None
    factor = 1.0 if result["metric"] in {"similarity", "sari"} else 100.0
    return result["value"] * factor


def log_evaluation_tables(run, summary):
    """Update one table per metric, without evaluation scalars or plots.

    Rows are checkpoints; columns are tasks sharing a primary metric. Scores
    use 0–100; forgetting and BWT use percentage points. Missing acquisition
    checkpoints stay null, including final-only evaluations.
    """
    if run is None:
        return
    import wandb

    config = summary["config"]
    tasks = config["tasks"]
    matrix = summary["performance_matrix"]
    aggregate = summary["continual_aggregates"]
    checkpoints = sorted(matrix, key=_stage)
    metric_by_task = {}
    for checkpoint in checkpoints:
        for task, result in matrix[checkpoint].items():
            metric = result["metric"]
            if task in metric_by_task and metric_by_task[task] != metric:
                raise ValueError(f"{task}: primary metric changed between checkpoints")
            metric_by_task[task] = metric

    metadata_columns = ["stage", "checkpoint", "split", "seed", "status", "unit"]

    def metadata(checkpoint, unit):
        status = summary.get("stage_status", {}).get(checkpoint, {}).get("evaluation", "unknown")
        return [_stage(checkpoint), checkpoint, config.get("eval_split"),
                config.get("seed"), status, unit]

    payload = {}
    for metric in dict.fromkeys(metric_by_task.values()):
        metric_tasks = [task for task in tasks if metric_by_task.get(task) == metric]
        columns = metadata_columns + metric_tasks + [f"{task}/samples" for task in metric_tasks]
        rows = {kind: [] for kind in ("scores", "forgetting_from_acquisition", "forgetting_from_best")}
        for index, checkpoint in enumerate(checkpoints):
            scores, acquisition_forgetting, best_forgetting, samples = [], [], [], []
            for task in metric_tasks:
                current = _score_100(matrix[checkpoint].get(task))
                scores.append(current)
                acquisition = f"{tasks.index(task) + 1:02d}_after_{task}"
                baseline = _score_100(matrix.get(acquisition, {}).get(task))
                # A lone final checkpoint cannot measure forgetting, even
                # for the final task whose acquisition checkpoint is present.
                has_history = (len(checkpoints) > 1 and current is not None
                               and baseline is not None
                               and acquisition in checkpoints[:index + 1])
                if has_history:
                    learned_index = checkpoints.index(acquisition)
                    history = [_score_100(matrix[name].get(task))
                               for name in checkpoints[learned_index:index + 1]]
                    acquisition_forgetting.append(baseline - current)
                    best_forgetting.append(max(value for value in history if value is not None) - current)
                else:
                    acquisition_forgetting.append(None)
                    best_forgetting.append(None)
                samples.append(summary.get("evaluations", {}).get(checkpoint, {}).get(task, {}).get("samples"))
            rows["scores"].append(metadata(checkpoint, "0-100") + scores + samples)
            rows["forgetting_from_acquisition"].append(
                metadata(checkpoint, "percentage points; acquisition - current") + acquisition_forgetting + samples)
            rows["forgetting_from_best"].append(
                metadata(checkpoint, "percentage points; best since acquisition - current") + best_forgetting + samples)
        for kind, data in rows.items():
            payload[f"eval/{kind}/{metric}"] = wandb.Table(columns=columns, data=data)

    def percentage_points(value):
        return None if value is None else value * 100.0

    payload["eval/aggregates/macro_average_available_tasks"] = wandb.Table(
        columns=metadata_columns + ["value", "task_count"],
        data=[metadata(checkpoint, "0-100; equal task weights")
              + [percentage_points(aggregate["checkpoint_macro_average_available_tasks"].get(checkpoint)),
                 sum(result["value"] is not None for result in matrix[checkpoint].values())]
              for checkpoint in checkpoints],
    )
    final = aggregate["final_checkpoint"]
    payload["eval/aggregates/backward_transfer"] = wandb.Table(
        columns=metadata_columns + ["value", "task_count"],
        data=[metadata(final, "percentage points; current - acquisition; excludes final training task")
              + [percentage_points(aggregate.get("backward_transfer")),
                 aggregate.get("backward_transfer_task_count", 0)]],
    )
    payload["eval/aggregates/mean_final_forgetting_from_best"] = wandb.Table(
        columns=metadata_columns + ["value", "task_count"],
        data=[metadata(final, "percentage points; equal task weights")
              + [percentage_points(aggregate.get("mean_final_forgetting_from_best")),
                 len(summary.get("forgetting", {}))]],
    )
    run.log(payload)
