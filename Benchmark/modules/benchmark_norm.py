# ==============================================================
# ✅ Benchmark: LayerNorm vs RMSNorm (PyTorch, easy to understand)
# ==============================================================

import torch
import time
import matplotlib.pyplot as plt
import gc

torch.manual_seed(42)

device = torch.device("cpu")
# --------------------------------------------------------------
# 1️⃣ Define LayerNorm (PyTorch built-in)
# --------------------------------------------------------------
layernorm = torch.nn.LayerNorm


# --------------------------------------------------------------
# 2️⃣ Define RMSNorm (manual implementation)
# --------------------------------------------------------------
class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute RMS: sqrt(mean(x²))
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        normed = x / rms
        return normed * self.weight


# --------------------------------------------------------------
# 3️⃣ Benchmark Function (forward + backward + memory)
# --------------------------------------------------------------
def benchmark_norm(norm_layer, batch, seq, dim, device=device):
    # torch.cuda.empty_cache()
    gc.collect()

    x = torch.randn(batch, seq, dim, device=device, requires_grad=True)
    norm = norm_layer(dim).to(device)

    # Warmup
    for _ in range(3):
        y = norm(x)
        loss = y.sum()
        loss.backward()
        x.grad = None
    # torch.cuda.synchronize()

    # Measure runtime and memory
    # torch.cuda.reset_peak_memory_stats()
    start = time.time()

    y = norm(x)
    loss = y.sum()
    loss.backward()

    # torch.cuda.synchronize()
    end = time.time()

    # peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    runtime = (end - start) * 1000  # ms

    del x, y, loss, norm
    # torch.cuda.empty_cache()
    gc.collect()

    return runtime#, peak_mem


# --------------------------------------------------------------
# 4️⃣ Run Benchmark
# --------------------------------------------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Running benchmark on {device.upper()}...")

dims = [512, 1024, 2048, 4096, 8192]
batch, seq = 4, 512

layernorm_times, rmsnorm_times = [], []
layernorm_mem, rmsnorm_mem = [], []

for d in dims:
    t1 = benchmark_norm(layernorm, batch, seq, d, device)
    t2 = benchmark_norm(RMSNorm, batch, seq, d, device)
    layernorm_times.append(t1)
    rmsnorm_times.append(t2)
    # layernorm_mem.append(m1)
    # rmsnorm_mem.append(m2)

    # print(f"Dim={d:<6} | LayerNorm: {t1:6.2f} ms, {m1:6.1f} MB | RMSNorm: {t2:6.2f} ms, {m2:6.1f} MB | Speedup: {t1/t2:5.2f}×")
    print(f"Dim={d:<6} | LayerNorm: {t1:6.2f} ms, | RMSNorm: {t2:6.2f} ms, | Speedup: {t1/t2:5.2f}×")

# --------------------------------------------------------------
# 5️⃣ Plot Results
# --------------------------------------------------------------
# fig, ax = plt.subplots(1, 2, figsize=(12,5))

# # Runtime plot
# ax[0].plot(dims, layernorm_times, "o--", label="LayerNorm")
# ax[0].plot(dims, rmsnorm_times, "o-", label="RMSNorm")
# ax[0].set_xlabel("Hidden Dimension (D)")
# ax[0].set_ylabel("Runtime (ms, forward + backward)")
# ax[0].set_title("Runtime Comparison")
# ax[0].grid(True, ls=":")
# ax[0].legend()

# # Memory plot
# ax[1].plot(dims, layernorm_mem, "o--", label="LayerNorm")
# ax[1].plot(dims, rmsnorm_mem, "o-", label="RMSNorm")
# ax[1].set_xlabel("Hidden Dimension (D)")
# ax[1].set_ylabel("Peak Memory (MB)")
# ax[1].set_title("Memory Usage Comparison")
# ax[1].grid(True, ls=":")
# ax[1].legend()

# plt.suptitle("LayerNorm vs RMSNorm — Forward + Backward Benchmark", fontsize=13)
# plt.tight_layout()
# plt.show()
