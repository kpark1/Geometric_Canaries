"""Small CUDA-memory checkpoints for inference debugging."""

import torch

# Decode lengths are sampled logarithmically to avoid printing for every token.
DECODE_MEMORY_CHECKPOINTS = frozenset(
    {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}
)


def reset_cuda_peak_memory() -> None:
    """Reset peak counters independently for every visible CUDA device."""
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)


def log_cuda_memory(label: str) -> None:
    """Print current and peak allocator usage for every visible CUDA device."""
    if not torch.cuda.is_available():
        return

    readings = []
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        readings.append(
            f"cuda:{device_index} "
            f"allocated={torch.cuda.memory_allocated(device_index) / 2**30:.2f}GiB "
            f"reserved={torch.cuda.memory_reserved(device_index) / 2**30:.2f}GiB "
            f"peak={torch.cuda.max_memory_allocated(device_index) / 2**30:.2f}GiB "
            f"free={free_bytes / 2**30:.2f}/{total_bytes / 2**30:.2f}GiB"
        )
    print(f"[cuda-memory] {label}: " + "; ".join(readings), flush=True)
