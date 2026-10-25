"""Choose vLLM request concurrency from actual available GPU memory."""

GIB = 1024 ** 3


def choose_batch_size(free_gib):
    if free_gib >= 88:
        return 64
    if free_gib >= 64:
        return 48
    if free_gib >= 40:
        return 32
    if free_gib >= 24:
        return 16
    return 8


def gpu_budget(batch_override=None, utilization_override=None):
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gib, total_gib = free_bytes / GIB, total_bytes / GIB
    batch_size = batch_override or choose_batch_size(free_gib)
    # vLLM's utilization is a fraction of total memory. Reserve 4 GiB beyond
    # the memory currently used by any other process on the assigned GPU.
    safe_utilization = (free_bytes - 4 * GIB) / total_bytes
    if safe_utilization <= 0:
        raise RuntimeError(f"Only {free_gib:.1f} GiB is free on the GPU")
    utilization = min(utilization_override or 0.85, safe_utilization)
    print(f"vLLM GPU budget: free={free_gib:.1f}/{total_gib:.1f} GiB, "
          f"batch_size={batch_size}, gpu_memory_utilization={utilization:.3f}", flush=True)
    return batch_size, utilization, free_gib, total_gib
