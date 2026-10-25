#!/usr/bin/env python3
"""Evaluate one continual-learning checkpoint and loop depth with vLLM."""
from __future__ import annotations

import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, __version__ as vllm_version

from ouro_pipeline.data import fixed_subset, read_split, token_length_report
from ouro_pipeline.metrics import evaluate_task
from scripts.parameters import generation_limits
from ouro_pipeline.evaluation_io import encode_prompt, save_summary
from ouro_pipeline.vllm_budget import gpu_budget


def run(args):
    with (args.run_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    config = summary["config"]
    token_caps = generation_limits(config)
    if any(task not in config["tasks"] for task in args.tasks):
        raise ValueError("--tasks must be from the training task order")
    checkpoint_path = args.run_dir / "checkpoints" / args.checkpoint
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(checkpoint_path)
    if not (checkpoint_path / "config.json").is_file():
        raise FileNotFoundError(f"No full model exists at {checkpoint_path}")
    batch_size, utilization, free_gib, total_gib = gpu_budget(
        args.batch_size, args.gpu_memory_utilization)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_new_tokens = max(config.get("max_new_tokens") or token_caps[task] for task in args.tasks)
    llm_kwargs = dict(
        model=str(checkpoint_path),
        dtype="bfloat16",
        trust_remote_code=True,
        hf_overrides={"total_ut_steps": args.loop_depth, "use_cache": True,
                      "early_exit_threshold": 1.0},
        max_model_len=config["max_length"] + max_new_tokens,
        gpu_memory_utilization=utilization,
        max_num_seqs=batch_size,
        seed=config["seed"],
    )
    llm = LLM(**llm_kwargs)
    summary["evaluation_engine"] = f"vLLM {vllm_version}, native OuroForCausalLM"
    summary.setdefault("evaluation_runtime", {}).setdefault(args.checkpoint, {})[str(args.loop_depth)] = {
        "batch_size": batch_size,
        "gpu_free_gib_at_start": free_gib,
        "gpu_total_gib": total_gib,
        "gpu_memory_utilization": utilization,
    }
    evaluations = summary.setdefault("evaluations", {}).setdefault(args.checkpoint, {})
    for task in args.tasks:
        records = fixed_subset(read_split(config["data_root"], task, config["eval_split"]),
                               config.get("eval_limit"), config["seed"])
        print(f"Evaluating {args.checkpoint}/{task}: {len(records)} samples, K={args.loop_depth}", flush=True)
        predictions, references, sources = [], [], []
        truncated_count = 0
        path = args.run_dir / "predictions" / args.checkpoint / f"{task}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".jsonl.tmp")
        sampling = SamplingParams(temperature=config.get("temperature", 0.0),
                                  max_tokens=config.get("max_new_tokens") or token_caps[task],
                                  detokenize=False)
        with temporary.open("w", encoding="utf-8") as output:
            for offset in range(0, len(records), batch_size):
                chunk = records[offset:offset + batch_size]
                encoded = [encode_prompt(tokenizer, row["prompt"], config["max_length"], task)
                           for row in chunk]
                if not config.get("truncate_long_prompts", True) and any(flag for _, flag in encoded):
                    raise ValueError(f"{task}: prompt exceeds max_length and truncation is disabled")
                generated = llm.generate([{"prompt_token_ids": ids} for ids, _ in encoded], sampling,
                                         use_tqdm=False)
                if len(generated) != len(chunk):
                    raise RuntimeError(f"{task}: expected {len(chunk)} outputs, got {len(generated)}")
                for index, (record, result, (_, truncated)) in enumerate(zip(chunk, generated, encoded), offset):
                    prediction = tokenizer.decode(result.outputs[0].token_ids,
                                                  skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
                    predictions.append(prediction)
                    references.append(record["answer"])
                    sources.append(record["prompt"])
                    truncated_count += int(truncated)
                    output.write(json.dumps({
                        "index": index, "prompt": record["prompt"], "prediction": prediction,
                        "reference": record["answer"], "prompt_truncated": truncated,
                    }, ensure_ascii=False) + "\n")
                output.flush()
                print(f"{task}: {min(offset + len(chunk), len(records))}/{len(records)}", flush=True)
        os.replace(temporary, path)
        metrics = evaluate_task(task, predictions, references, sources=sources)
        metrics.update({
            "prompt_overflow": 0, "truncated_prompts": truncated_count,
            "lengths": token_length_report(records, tokenizer, config["max_length"]),
            "prompt_truncation": "left" if task == "Py150" else "right",
            "eval_loops": args.loop_depth,
        })
        evaluations[task] = metrics
        save_summary(summary, args.run_dir)
