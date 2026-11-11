import torch
from time import perf_counter
from statistics import median


def synchronize_device(timing_device):
    if timing_device.type == "cuda":
        torch.cuda.synchronize(timing_device)
    elif timing_device.type == "mps":
        torch.mps.synchronize()


def measure_inference_ms(timing_model, timing_images, timing_device, warmups=5, repeats=20):
    """Time completed forward passes on a batch already moved to the device."""
    timing_model.eval()
    with torch.inference_mode():
        for _ in range(warmups):
            timing_model(timing_images)
        # Finish warmup before timing; each repeat also finishes before the next.
        synchronize_device(timing_device)
        times_ms = []
        for _ in range(repeats):
            start = perf_counter()
            timing_model(timing_images)
            synchronize_device(timing_device)
            times_ms.append((perf_counter() - start) * 1000)
    return median(times_ms)

