#!/usr/bin/env python3
"""Evaluate an untouched Ouro checkpoint on TRACE with native vLLM inference."""
from __future__ import annotations

import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, __version__ as vllm_version

from ouro_pipeline.data import read_split, token_length_report
from ouro_pipeline.metrics import evaluate_task
from ouro_pipeline.vllm_budget import gpu_budget

from scripts.parameters import generation_limits
from ouro_pipeline.evaluation_io import encode_prompt, save_summary

def run(args):
    loops = args.loop_depth
    tasks = args.tasks
    token_caps = generation_limits(vars(args))
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    largest_generation = max(args.max_new_tokens or token_caps[task] for task in tasks)
    batch_size, utilization, free_gib, total_gib = gpu_budget(args.batch_size, args.gpu_memory_utilization)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        hf_overrides={"total_ut_steps": loops},
        max_model_len=args.max_length + largest_generation,
        gpu_memory_utilization=utilization,
        max_num_seqs=batch_size,
        seed=args.seed,
    )

    root = args.output_dir
    root.mkdir(parents=True)
    summary = {
        "config": {**vars(args), "output_dir": str(root), "tasks": tasks, "loop_depth": loops},
        "inference_engine": f"vLLM {vllm_version}, native OuroForCausalLM",
        "model_state": "untouched base model; no LoRA and no optimization",
        "evaluations": {"00_base_untrained": {}},
        "stage_status": {"00_base_untrained": {"evaluation": "running"}},
        "evaluation_runtime": {"batch_size": batch_size, "gpu_free_gib_at_start": free_gib,
                               "gpu_total_gib": total_gib, "gpu_memory_utilization": utilization},
    }
    save_summary(summary, root)

    for task in tasks:
        records = read_split(args.data_root, task, args.split)
        if args.limit is not None:
            records = records[:args.limit]
        report = token_length_report(records, tokenizer, args.max_length)
        predictions, references, sources = [], [], []
        truncated_count = 0
        path = root / "predictions" / "00_base_untrained" / f"{task}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens or token_caps[task], detokenize=False)
        print(f"Evaluating {task}: split={args.split}, samples={len(records)}, K={loops}", flush=True)
        with path.open("w", encoding="utf-8") as handle:
            for offset in range(0, len(records), batch_size):
                chunk = records[offset:offset + batch_size]
                encoded = [encode_prompt(tokenizer, row["prompt"], args.max_length, task) for row in chunk]
                if not args.truncate_long_prompts and any(truncated for _, truncated in encoded):
                    raise ValueError(f"{task}: prompt exceeds max_length and truncation is disabled")
                outputs = llm.generate([{"prompt_token_ids": ids} for ids, _ in encoded], sampling, use_tqdm=False)
                if len(outputs) != len(chunk):
                    raise RuntimeError(f"{task}: expected {len(chunk)} outputs, got {len(outputs)}")
                for index, (record, output, (_, truncated)) in enumerate(zip(chunk, outputs, encoded), start=offset):
                    prediction = tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    predictions.append(prediction)
                    references.append(record["answer"])
                    sources.append(record["prompt"])
                    truncated_count += int(truncated)
                    handle.write(json.dumps({"index": index, "prompt": record["prompt"], "prediction": prediction, "reference": record["answer"], "prompt_truncated": truncated}, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"{task}: {min(offset + len(chunk), len(records))}/{len(records)}", flush=True)

        metrics = evaluate_task(task, predictions, references, sources=sources)
        metrics.update({"prompt_overflow": 0, "truncated_prompts": truncated_count, "lengths": report, "prompt_truncation": "left" if task == "Py150" else "right", "eval_loops": loops})
        summary["evaluations"]["00_base_untrained"][task] = metrics
        save_summary(summary, root)

    summary["stage_status"]["00_base_untrained"]["evaluation"] = "complete"
    save_summary(summary, root)
