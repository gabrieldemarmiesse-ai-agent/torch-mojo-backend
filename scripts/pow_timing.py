"""Streamed device-time proxy for fp32 pow on 16M elements: synchronize,
burst of N launches, synchronize, divide. Argument: a label for the tree."""

import sys
import time
from collections.abc import Callable

import torch
from torch_mojo_backend import register_mojo_devices
import torch_mojo_backend.native as n

register_mojo_devices()
label = sys.argv[1]
print(f"[{label}] KERNELS_DIR {n._KERNELS_DIR}", flush=True)
DEV = torch.device("mojo:0")
sync = torch.mojo.synchronize  # ty: ignore[unresolved-attribute]


def time_us(fn: Callable[[], object], iters: int = 20, reps: int = 7) -> float:
    for _ in range(3):
        fn()
    sync()
    best = float("inf")
    for _ in range(reps):
        sync()
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        best = min(best, (time.perf_counter() - t) / iters * 1e6)
    return best


torch.manual_seed(0)
M = 1 << 24
xt = (torch.rand(M) * 49.5 + 0.5).to(DEV)
yt = (torch.rand(M) * 6).to(DEV)
yi = torch.randint(0, 6, (M,)).float().to(DEV)
base = torch.tensor(2.5).to(DEV)
xs, ys = xt[: 1 << 20], yt[: 1 << 20]
for name, fn, nbytes in (
    ("16M pow(T,T) fractional y", lambda: torch.pow(xt, yt), 12 * M),
    ("16M pow(T,T) integral y", lambda: torch.pow(xt, yi), 12 * M),
    ("16M pow(T, 2.5)", lambda: torch.pow(xt, 2.5), 8 * M),
    ("16M pow(T, 2.0)", lambda: torch.pow(xt, 2.0), 8 * M),
    ("16M tensor(2.5) ** T", lambda: torch.pow(base, yt), 8 * M),
    ("1M pow(T,T) fractional y", lambda: torch.pow(xs, ys), 12 * (M >> 4)),
):
    us = time_us(fn)
    print(
        f"[{label}] {name:28s} {us:9.1f} us/iter  {nbytes / us / 1e3:7.0f} GB/s",
        flush=True,
    )
