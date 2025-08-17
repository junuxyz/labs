import torch
from add_triton import triton_add


def benchmark_pytorch(a, b):
    return a + b


def run_benchmark(fn, *args):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # Warm-up for GPU
    for _ in range(10):
        fn(*args)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        fn(*args)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms / 100


device = "cuda" if torch.cuda.is_available() else "cpu"
size = 1024 * 1024
a = torch.randn(size, device=device)
b = torch.randn(size, device=device)

pytorch_time = run_benchmark(benchmark_pytorch, a, b)
triton_time = run_benchmark(triton_add, a, b)

print(f"Vector size: {size}")
print(f"PyTorch average time: {pytorch_time:.6f} ms")
print(f"Triton average time: {triton_time:.6f} ms")


# GPU used: 3050ti laptop GPU
# ❯ python benchmark.py
# Vector size: 1048576
# PyTorch average time: 0.124672 ms
# Triton average time: 0.124037 ms
# There seems no room for optimizing vector addition. PyTorch is as good.
