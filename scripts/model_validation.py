#!/usr/bin/env python3
import json

import torch

from ouro_pipeline.modeling import load_ouro, load_tokenizer


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This acceptance test requires a CUDA GPU")
    device = "cuda"
    tokenizer = load_tokenizer(args.model)
    batch = tokenizer(args.prompt, return_tensors="pt").to(device)
    model = load_ouro(args.model, max(args.loops), device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report = {"parameter_count": parameter_count, "depths": {}}
    trajectories = {}

    with torch.inference_mode():
        for loops in args.loops:
            model.config.total_ut_steps = loops
            model.model.total_ut_steps = loops
            calls = [0] * len(model.model.layers)
            hooks = []
            for index, layer in enumerate(model.model.layers):
                hooks.append(layer.register_forward_hook(lambda _m, _i, _o, i=index: calls.__setitem__(i, calls[i] + 1)))
            base_output, hidden_states, _ = model.model(**batch, use_cache=False)
            for hook in hooks:
                hook.remove()
            if calls != [loops] * len(calls):
                raise AssertionError(f"Depth {loops}: physical layer calls are {calls}")
            trajectories[loops] = [state.float().cpu() for state in hidden_states]
            no_cache = model(**batch, labels=None, exit_at_step=loops - 1, use_cache=False).logits
            with_cache = model(**batch, labels=None, exit_at_step=loops - 1, use_cache=True).logits
            cache_diff = (no_cache.float() - with_cache.float()).abs().max().item()
            if not torch.allclose(no_cache.float(), with_cache.float(), atol=args.atol, rtol=args.rtol):
                raise AssertionError(f"Depth {loops}: cache/no-cache logits differ by {cache_diff}")
            generated = model.generate(**batch, max_new_tokens=2, do_sample=False, exit_at_step=loops - 1)
            report["depths"][str(loops)] = {
                "layer_calls": sum(calls),
                "expected_layer_calls": loops * len(calls),
                "cache_max_abs_diff": cache_diff,
                "generated_tokens": generated.shape[1] - batch["input_ids"].shape[1],
            }

    largest = max(args.loops)
    for loops in args.loops:
        for index in range(min(loops, largest)):
            if not torch.allclose(trajectories[loops][index], trajectories[largest][index], atol=args.atol, rtol=args.rtol):
                raise AssertionError(f"Shared trajectory mismatch at depth={loops}, step={index}")
    report["shared_trajectory"] = True
    report["passed"] = True
    print(json.dumps(report, indent=2))

