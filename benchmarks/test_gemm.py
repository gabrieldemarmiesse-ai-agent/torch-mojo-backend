"""GEMM performance-regression benchmarks: mm / bmm / addmm / linear.

One pytest node per (shape, layout, dtype, op) case; the node id names the
kernel regime, so the list of failing tests is the list of regressed
regimes.  Select with ordinary pytest, e.g.:

    uv run pytest benchmarks/test_gemm.py -k "test_mm and S7 and TN and bf16"
    uv run pytest "benchmarks/test_gemm.py::test_mm[S1_4096x4096x4096-NN-bf16]"

Shapes come from the 2026-08 GEMM campaign.  S7 is the nanoGPT lm_head
weight gradient (GPT-2 124M, 48x1024 tokens, vocab padded to 50304):
tiny-M / huge-N / huge-K, the regime the campaign initially missed.

Layout letters follow BLAS: for A (M,K) x B (K,N), 'T' means the operand
is stored transposed and reached through a .t() view, exactly as
linear_backward produces it (NN = dgrad, NT = forward linear, TN = wgrad).

The "tf32" dtype id is float32 run at torch.set_float32_matmul_precision
("high") — the TF32 route — while "f32" is precision "highest".
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest
import torch
from bench_lib.check import Bench
from bench_lib.hw import Hardware

# M, N, K
SHAPES = {
    "S1_4096x4096x4096": (4096, 4096, 4096),
    "S2_8192x2048x2048": (8192, 2048, 2048),
    "S3_2048x8192x2048": (2048, 8192, 2048),
    "S4_1024x1024x8192": (1024, 1024, 8192),  # deep-K
    "S5_357x789x333": (357, 789, 333),  # awkward, exercises edge tiles
    "S6_32768x768x768": (32768, 768, 768),  # tall-skinny
    "S7_768x50304x49152": (768, 50304, 49152),  # lm_head wgrad: tiny-M/huge-N/huge-K
    # GPT-2 XL (1600 embedding, B=16 x T=1024 tokens) projection sites, as
    # aten::linear sees them: M x out_features x in_features.  1600, 4800 and
    # 6400 are all 64 modulo 128 -- a residue no shape above has in ANY
    # dimension, and the one the gemm16 candidates (rolling NN, the widened TN
    # selection, the fused-bias NT) are gated on.  test_linear measures the
    # fused forward; test_linear_backward's two legs are the dX (M x K x N) and
    # dW (N x K x M) products, which is where the NN and TN candidates run.
    "S8_16384x4800x1600": (16384, 4800, 1600),  # c_attn
    "S9_16384x1600x1600": (16384, 1600, 1600),  # attn.c_proj
    "S10_16384x6400x1600": (16384, 6400, 1600),  # mlp.c_fc
    "S11_16384x1600x6400": (16384, 1600, 6400),  # mlp.c_proj
    # lm_head, the fifth site of the same step and bias-free in the model:
    # measured at parity with cuda before the candidates went in, and outside
    # every one of their gates (the aspect limits reject a 50304-wide
    # projection). A regression guard for the routes it already had, not a
    # candidate target. test_linear passes a bias on every shape, so this node
    # measures the biased call rather than the model's bias-free one.
    "S12_16384x50304x1600": (16384, 50304, 1600),
}

# Batched, these four would be 2-3 TFLOP and several GB per leg for a regime
# the candidates never serve: the single-matrix GEMM entry is where they are
# installed, so every BMM item keeps its existing route by construction.
BMM_EXCLUDED = {
    "S7_768x50304x49152",  # ~40 GB per leg
    "S8_16384x4800x1600",
    "S9_16384x1600x1600",
    "S10_16384x6400x1600",
    "S11_16384x1600x6400",
    "S12_16384x50304x1600",  # tens of GB per leg batched
}

LAYOUTS = ("NN", "NT", "TN", "TT")

# id -> (torch dtype, float32 matmul precision during the case)
DTYPES = {
    "bf16": (torch.bfloat16, "highest"),
    "f16": (torch.float16, "highest"),
    "f32": (torch.float32, "highest"),
    "tf32": (torch.float32, "high"),
}

COVERS: dict[str, str] = {
    "aten::mm": "test_mm",
    "aten::bmm": "test_bmm",
    "aten::addmm": "test_addmm",
    "aten::linear": "test_linear",
    "aten::linear_backward": "test_linear_backward",
}

SKIPPED: dict[str, str] = {}

BMM_BATCH = 8
BMM_SHAPES = {
    f"{tag.split('_')[0]}_{BMM_BATCH}x{tag.split('_')[1]}": dims
    for tag, dims in SHAPES.items()
    if tag not in BMM_EXCLUDED
}


@contextlib.contextmanager
def matmul_precision(precision: str) -> Iterator[None]:
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _mat(
    rows: int,
    cols: int,
    transposed: bool,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    if transposed:
        return torch.randn(cols, rows, dtype=dtype, device=device).t()
    return torch.randn(rows, cols, dtype=dtype, device=device)


def _bmat(
    batch: int,
    rows: int,
    cols: int,
    transposed: bool,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    if transposed:
        return torch.randn(batch, cols, rows, dtype=dtype, device=device).transpose(
            1, 2
        )
    return torch.randn(batch, rows, cols, dtype=dtype, device=device)


def _operand_pair(
    layout: str, m: int, n: int, k: int, dtype: torch.dtype, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    a = _mat(m, k, layout[0] == "T", dtype, device)
    b = _mat(k, n, layout[1] == "T", dtype, device)
    return a, b


@pytest.mark.parametrize("dtype_id", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape_id", SHAPES)
def test_mm(
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    m, n, k = SHAPES[shape_id]
    dtype, precision = DTYPES[dtype_id]
    with matmul_precision(precision):
        a_ref, b_ref = _operand_pair(layout, m, n, k, dtype, hw.stock_device)
        a_our, b_our = _operand_pair(layout, m, n, k, dtype, mojo_device)
        bench.run(
            lambda: torch.mm(a_ref, b_ref),
            lambda: torch.mm(a_our, b_our),
            flops=2.0 * m * n * k,
        )


@pytest.mark.parametrize("dtype_id", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape_id", BMM_SHAPES)
def test_bmm(
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    m, n, k = BMM_SHAPES[shape_id]
    dtype, precision = DTYPES[dtype_id]
    with matmul_precision(precision):
        a_ref = _bmat(BMM_BATCH, m, k, layout[0] == "T", dtype, hw.stock_device)
        b_ref = _bmat(BMM_BATCH, k, n, layout[1] == "T", dtype, hw.stock_device)
        a_our = _bmat(BMM_BATCH, m, k, layout[0] == "T", dtype, mojo_device)
        b_our = _bmat(BMM_BATCH, k, n, layout[1] == "T", dtype, mojo_device)
        bench.run(
            lambda: torch.bmm(a_ref, b_ref),
            lambda: torch.bmm(a_our, b_our),
            flops=2.0 * BMM_BATCH * m * n * k,
        )


@pytest.mark.parametrize("dtype_id", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape_id", SHAPES)
def test_addmm(
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    m, n, k = SHAPES[shape_id]
    dtype, precision = DTYPES[dtype_id]
    with matmul_precision(precision):
        a_ref, b_ref = _operand_pair(layout, m, n, k, dtype, hw.stock_device)
        a_our, b_our = _operand_pair(layout, m, n, k, dtype, mojo_device)
        bias_ref = torch.randn(n, dtype=dtype, device=hw.stock_device)
        bias_our = torch.randn(n, dtype=dtype, device=mojo_device)
        bench.run(
            lambda: torch.addmm(bias_ref, a_ref, b_ref),
            lambda: torch.addmm(bias_our, a_our, b_our),
            flops=2.0 * m * n * k,
        )


# linear has no layout axis: x @ weight.t() + bias with weight stored
# (out_features, in_features) IS the NT regime by construction.
@pytest.mark.parametrize("dtype_id", DTYPES)
@pytest.mark.parametrize("shape_id", SHAPES)
def test_linear(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    m, n, k = SHAPES[shape_id]
    dtype, precision = DTYPES[dtype_id]
    with matmul_precision(precision):
        x_ref = torch.randn(m, k, dtype=dtype, device=hw.stock_device)
        w_ref = torch.randn(n, k, dtype=dtype, device=hw.stock_device)
        bias_ref = torch.randn(n, dtype=dtype, device=hw.stock_device)
        x_our = torch.randn(m, k, dtype=dtype, device=mojo_device)
        w_our = torch.randn(n, k, dtype=dtype, device=mojo_device)
        bias_our = torch.randn(n, dtype=dtype, device=mojo_device)
        bench.run(
            lambda: torch.nn.functional.linear(x_ref, w_ref, bias_ref),
            lambda: torch.nn.functional.linear(x_our, w_our, bias_our),
            flops=2.0 * m * n * k,
        )


# aten::linear_backward has no CUDA registration in stock PyTorch (CUDA
# decomposes linear to addmm, so its backward is matmul nodes); the stock
# reference leg therefore composes the exact equivalent three-op sequence:
# dgrad g @ w, wgrad g.t() @ x, bgrad g.sum(0).
@pytest.mark.parametrize("dtype_id", DTYPES)
@pytest.mark.parametrize("shape_id", SHAPES)
def test_linear_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    m, n, k = SHAPES[shape_id]
    dtype, precision = DTYPES[dtype_id]
    with matmul_precision(precision):
        x_ref = torch.randn(m, k, dtype=dtype, device=hw.stock_device)
        w_ref = torch.randn(n, k, dtype=dtype, device=hw.stock_device)
        g_ref = torch.randn(m, n, dtype=dtype, device=hw.stock_device)
        x_our = torch.randn(m, k, dtype=dtype, device=mojo_device)
        w_our = torch.randn(n, k, dtype=dtype, device=mojo_device)
        g_our = torch.randn(m, n, dtype=dtype, device=mojo_device)

        def ref_leg() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return g_ref @ w_ref, g_ref.t() @ x_ref, g_ref.sum(0)

        bench.run(
            ref_leg,
            lambda: torch.ops.aten.linear_backward(
                x_our, g_our, w_our, [True, True, True]
            ),
            flops=4.0 * m * n * k,
        )
