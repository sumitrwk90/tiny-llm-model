
# -*- coding: utf-8 -*-
"""
Universal PyTorch Benchmark Function
-----------------------------------
Use this to benchmark any mathematical or activation function.
"""

import torch
import time

# Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
DTYPE = torch.float32

def benchmark_function(fn, input_shape, backward=False, num_iters=100, dtype=DTYPE):
    """
    Generic benchmarking utility for PyTorch functions.

    Args:
        fn: Callable, the function to benchmark (e.g., torch.relu, custom_fn)
        input_shape: tuple, e.g. (batch, seq_len, hidden)
        backward: bool, whether to test backward pass
        num_iters: int, number of iterations for timing
        dtype: torch.dtype

    Returns:
        avg_time_ms: float, average execution time in milliseconds
    """
    x = torch.randn(*input_shape, device=device, dtype=dtype, requires_grad=True)

    # Warmup (to stabilize GPU)
    for _ in range(10):
        y = fn(x)
        if backward:
            y.backward(torch.randn_like(y), retain_graph=True)
    # torch.cuda.synchronize()

    # Timing loop
    start = time.time()
    for _ in range(num_iters):
        y = fn(x)
        if backward:
            g = torch.randn_like(y)
            y.backward(g, retain_graph=True)
    # torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) * 1000 / num_iters
    return avg_ms


# ==========================================================
# Example Usage
# ==========================================================
if __name__ == "__main__":
    activations = {
        "ReLU": torch.relu,
        "Sigmoid": torch.sigmoid,
        "Tanh": torch.tanh,
        "Swish": lambda x: x * torch.sigmoid(x),
        "GELU": torch.nn.functional.gelu
    }

    shapes = [
        (4, 1024, 1024),
        (4, 4096, 1024),
        (4, 8192, 1024),
    ]

    for shape in shapes:
        print(f"\n=== Input Shape: {shape} ===")
        for name, fn in activations.items():
            fwd = benchmark_function(fn, shape, backward=False)
            fwdbwd = benchmark_function(fn, shape, backward=True)
            print(f"{name:10s} | fwd: {fwd:.3f} ms | fwdbwd: {fwdbwd:.3f} ms")


# benchmark_function(lambda x: torch.nn.functional.leaky_relu(x, 0.1), (8, 2048, 2048))