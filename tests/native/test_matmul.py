"""End-to-end checks of the matmul group on the native backend: mm, bmm,
addmm, linear, linear_backward, addr and the convolution forward.

Public torch API only — every assertion compares the mojo device against the
same computation on CPU, and `assert_ran` proves the native op is what
produced it (rather than a decomposition into something else).
"""

import contextlib
import os

import pytest
import torch

from torch_mojo_backend import aten_functions, get_accelerators, native
from torch_mojo_backend.testing import CallChecker


@contextlib.contextmanager
def assert_ran(*op_names: str):
    """Assert that each aten op ran as a native boxed kernel in the block."""
    native.op_counting(True)
    before = native.op_counts()
    yield
    after = native.op_counts()
    for name in op_names:
        assert after.get(name, 0) > before.get(name, 0), (
            f"{name} did not run natively (counted: {sorted(after)})"
        )


@pytest.fixture
def mojo_h100(mojo_gpu):
    """H100 mojo device: the gemm16 / tf32 tensor-core bridges are gated to
    compute capability 9.0 and decline everywhere else."""
    accelerator = list(get_accelerators())[0]
    if accelerator.api != "cuda" or accelerator.architecture_name != "sm_90a":
        pytest.skip("the pure-Mojo tensor-core fast paths require an H100")
    return mojo_gpu


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    """(atol, rtol): fp32 to fp32 accuracy, 16-bit to 16-bit rounding."""
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 5e-2, 5e-2


def _ref(*tensors: torch.Tensor) -> list[torch.Tensor]:
    """CPU float32 copies: the reference is always accumulated in fp32, the
    way every kernel in this family does."""
    return [t.cpu().float() for t in tensors]


# --- mm -----------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mm(mojo_device, dtype, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_mm)
    a = torch.randn(64, 128).to(dtype)
    b = torch.randn(128, 96).to(dtype)
    got = torch.mm(a.to(mojo_device), b.to(mojo_device)).cpu()
    assert got.dtype == dtype
    ra, rb = _ref(a, b)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close((ra @ rb).to(dtype), got, atol=atol, rtol=rtol)


def test_mm_transposed_operands(mojo_device):
    """A `.t()` operand is a dense transposed layout, which every route reads
    in place — no materialized transpose."""
    a = torch.randn(48, 32)
    bt = torch.randn(64, 32)
    with assert_ran("aten::mm"):
        got = torch.mm(a.to(mojo_device), bt.to(mojo_device).t()).cpu()
    torch.testing.assert_close(got, a @ bt.t(), atol=1e-4, rtol=1e-4)
    with assert_ran("aten::mm"):
        got2 = torch.mm(a.to(mojo_device).t().contiguous().t(), bt.to(mojo_device).t())
    torch.testing.assert_close(got2.cpu(), a @ bt.t(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mm_degenerate_dims(mojo_device, dtype):
    # n == 1 used to segfault the CPU library-matmul route (gemv special case
    # without a DeviceContext); m == 1 / k == 1 pin the library's other
    # special-case routes.
    for m, k, n in [(37, 129, 1), (1, 129, 64), (64, 1, 33), (1, 129, 1)]:
        a = torch.randn(m, k).to(dtype)
        b = torch.randn(k, n).to(dtype)
        got = torch.mm(a.to(mojo_device), b.to(mojo_device)).cpu()
        ra, rb = _ref(a, b)
        torch.testing.assert_close(got, (ra @ rb).to(dtype), atol=5e-2, rtol=5e-2)


def test_mm_unsupported_dtype_raises(mojo_device):
    """Integer matmul has no kernel in this family: the op declines, and the
    decline reaches python as NotImplementedError (never a wrong answer)."""
    a = torch.arange(6, dtype=torch.int64).reshape(2, 3).to(mojo_device)
    b = torch.arange(6, dtype=torch.int64).reshape(3, 2).to(mojo_device)
    with pytest.raises(NotImplementedError):
        torch.mm(a, b)


# --- bmm ----------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bmm(mojo_device, dtype, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_bmm)
    a = torch.randn(3, 64, 128).to(dtype)
    b = torch.randn(3, 128, 96).to(dtype)
    got = torch.bmm(a.to(mojo_device), b.to(mojo_device)).cpu()
    ra, rb = _ref(a, b)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(got, torch.bmm(ra, rb).to(dtype), atol=atol, rtol=rtol)


def test_bmm_expanded_batch(mojo_device):
    """A stride-0 batch dimension is one matrix shared by every item, which
    the batched routes address directly instead of materializing."""
    a = torch.randn(4, 12, 20)
    b = torch.randn(20, 8)
    # as_strided rather than expand(): only the view ops this group's own
    # tests may rely on (`aten::as_strided`) are guaranteed registered here.
    shared = torch.as_strided(b.to(mojo_device), (4, 20, 8), (0, 8, 1))
    with assert_ran("aten::bmm"):
        got = torch.bmm(a.to(mojo_device), shared).cpu()
    ref = torch.bmm(a, torch.as_strided(b, (4, 20, 8), (0, 8, 1)))
    torch.testing.assert_close(got, ref)


def test_bmm_transposed_rhs(mojo_device):
    a = torch.randn(2, 16, 24)
    b = torch.randn(2, 10, 24)
    got = torch.bmm(a.to(mojo_device), b.to(mojo_device).transpose(1, 2)).cpu()
    torch.testing.assert_close(
        got, torch.bmm(a, b.transpose(1, 2)), atol=1e-4, rtol=1e-4
    )


# --- addmm --------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_addmm(mojo_device, dtype, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_addmm)
    bias = torch.randn(96).to(dtype)
    a = torch.randn(64, 128).to(dtype)
    b = torch.randn(128, 96).to(dtype)
    got = torch.addmm(bias.to(mojo_device), a.to(mojo_device), b.to(mojo_device)).cpu()
    rbias, ra, rb = _ref(bias, a, b)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(got, (ra @ rb + rbias).to(dtype), atol=atol, rtol=rtol)


def test_addmm_scaled_declines(mojo_device):
    """beta/alpha scaling is not implemented by this family; the decline is a
    NotImplementedError, not a silently dropped scale."""
    bias = torch.randn(8).to(mojo_device)
    a = torch.randn(4, 6).to(mojo_device)
    b = torch.randn(6, 8).to(mojo_device)
    with pytest.raises(NotImplementedError):
        torch.addmm(bias, a, b, beta=0.5)
    with pytest.raises(NotImplementedError):
        torch.addmm(bias, a, b, alpha=2.0)


# --- linear -------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(64, 128), (2, 64, 128), (128,)])
def test_linear(mojo_device, dtype, shape, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_linear)
    x = torch.randn(*shape).to(dtype)
    w = torch.randn(96, 128).to(dtype)
    bias = torch.randn(96).to(dtype)
    for b in (None, bias):
        got = torch.nn.functional.linear(
            x.to(mojo_device),
            w.to(mojo_device),
            None if b is None else b.to(mojo_device),
        ).cpu()
        rx, rw = _ref(x, w)
        ref = torch.nn.functional.linear(rx, rw, None if b is None else b.cpu().float())
        assert got.shape == ref.shape
        atol, rtol = _tol(dtype)
        torch.testing.assert_close(got, ref.to(dtype), atol=atol, rtol=rtol)


def test_linear_is_not_decomposed_to_addmm(mojo_device):
    """nn.Linear reaches aten::linear, not addmm: that is what keeps its
    backward the fused aten::linear_backward node."""
    x = torch.randn(5, 16)
    layer = torch.nn.Linear(16, 24)
    native.op_counting(True)
    before = native.op_counts()
    got = torch.nn.functional.linear(
        x.to(mojo_device),
        layer.weight.detach().to(mojo_device),
        layer.bias.detach().to(mojo_device),
    ).cpu()
    after = native.op_counts()
    assert after.get("aten::linear", 0) > before.get("aten::linear", 0)
    assert after.get("aten::addmm", 0) == before.get("aten::addmm", 0)
    torch.testing.assert_close(got, layer(x), atol=1e-4, rtol=1e-4)


def test_linear_empty_features(mojo_device):
    """The rank-1 vector route's own edge cases: a zero-width output, and a
    zero-length input whose result is just the bias."""
    x0 = torch.randn(0)
    w0 = torch.randn(7, 0)
    bias = torch.randn(7)
    got = torch.nn.functional.linear(
        x0.to(mojo_device), w0.to(mojo_device), bias.to(mojo_device)
    ).cpu()
    torch.testing.assert_close(got, bias)
    got_nobias = torch.nn.functional.linear(
        x0.to(mojo_device), w0.to(mojo_device)
    ).cpu()
    torch.testing.assert_close(got_nobias, torch.zeros(7))


# --- linear_backward ----------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_linear_backward_through_autograd(mojo_device, dtype):
    """nn.Linear's backward IS aten::linear_backward here: registering
    aten::linear keeps the layer from decomposing to addmm, so the recorded
    node is the fused one."""
    x = torch.randn(8, 32).to(dtype)
    w = torch.randn(16, 32).to(dtype)
    b = torch.randn(16).to(dtype)
    upstream = torch.randn(8, 16).to(dtype)

    def run(device):
        xi = x.to(device).requires_grad_()
        wi = w.to(device).requires_grad_()
        bi = b.to(device).requires_grad_()
        torch.nn.functional.linear(xi, wi, bi).backward(upstream.to(device))
        return xi.grad, wi.grad, bi.grad

    with assert_ran("aten::linear", "aten::linear_backward"):
        got = run(mojo_device)
    ref = run("cpu")
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g.cpu().float(), r.float(), atol=5e-2, rtol=5e-2)


def test_linear_backward_higher_rank_input(mojo_device):
    x = torch.randn(2, 5, 12)
    w = torch.randn(7, 12)
    upstream = torch.randn(2, 5, 7)

    def run(device):
        xi = x.to(device).requires_grad_()
        wi = w.to(device).requires_grad_()
        torch.nn.functional.linear(xi, wi).backward(upstream.to(device))
        return xi.grad, wi.grad

    got = run(mojo_device)
    ref = run("cpu")
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g.cpu(), r, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "mask", [(True, False, False), (False, True, True), (True, True, True)]
)
def test_linear_backward_output_mask(mojo_device, mask, call_checker: CallChecker):
    """Called directly: only the requested gradients have to be right, and
    PyTorch's contract defines both parameter outputs when either is asked
    for."""
    call_checker.register(aten_functions.aten_linear_backward)
    x = torch.randn(6, 10)
    w = torch.randn(4, 10)
    grad = torch.randn(6, 4)
    out = torch.ops.aten.linear_backward(
        x.to(mojo_device), grad.to(mojo_device), w.to(mojo_device), list(mask)
    )
    if mask[0]:
        torch.testing.assert_close(out[0].cpu(), grad @ w, atol=1e-4, rtol=1e-4)
    if mask[1]:
        torch.testing.assert_close(out[1].cpu(), grad.t() @ x, atol=1e-4, rtol=1e-4)
    if mask[2]:
        torch.testing.assert_close(out[2].cpu(), grad.sum(dim=0), atol=1e-4, rtol=1e-4)


# --- addr ---------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_addr(mojo_device, dtype):
    """The fused kernel reproduces CPU's own addr_kernel op order and per-op
    rounding: ATen's composite fallback multiplies in a different order,
    which drifted enough to fail OpInfo conformance for fp16/bf16. beta and
    alpha below are the failing OpInfo sample."""
    self_ = torch.randn(5, 10).to(dtype)
    vec1 = torch.randn(5).to(dtype)
    vec2 = torch.randn(10).to(dtype)
    with assert_ran("aten::addr"):
        got = torch.addr(
            self_.to(mojo_device),
            vec1.to(mojo_device),
            vec2.to(mojo_device),
            beta=0.6,
            alpha=0.2,
        ).cpu()
    # The tolerance is one ULP of the *intermediates* (beta*self, alpha*outer),
    # which a cancelling output element can show in full: on the GPU one of 50
    # elements lands a bf16 ULP away. ATen's composite gets the order itself
    # wrong, which moved a fifth of the elements much further than this.
    atol = 1e-5 if dtype == torch.float32 else 8e-3
    torch.testing.assert_close(
        got, torch.addr(self_, vec1, vec2, beta=0.6, alpha=0.2), atol=atol, rtol=2e-2
    )


def test_addr_default_beta_alpha(mojo_device):
    self_ = torch.randn(4, 6)
    vec1 = torch.randn(4)
    vec2 = torch.randn(6)
    got = torch.addr(
        self_.to(mojo_device), vec1.to(mojo_device), vec2.to(mojo_device)
    ).cpu()
    torch.testing.assert_close(got, torch.addr(self_, vec1, vec2))


def test_addr_beta_zero_ignores_self(mojo_device):
    """beta=0 must ignore `self` entirely, nan included (ATen's own addr
    contract, aten/src/ATen/native/LinearAlgebra.cpp)."""
    self_ = torch.full((3, 4), float("nan"))
    vec1 = torch.randn(3)
    vec2 = torch.randn(4)
    got = torch.addr(
        self_.to(mojo_device),
        vec1.to(mojo_device),
        vec2.to(mojo_device),
        beta=0.0,
        alpha=1.5,
    ).cpu()
    torch.testing.assert_close(got, torch.addr(self_, vec1, vec2, beta=0.0, alpha=1.5))


def test_addr_self_broadcast(mojo_device):
    """`self` broadcastable to (len(vec1), len(vec2)) but not that exact shape
    (here 0-d): the fused kernel's own right-alignment handles it, so this
    does not reach the composite."""
    self_ = torch.randn(())
    vec1 = torch.randn(3)
    vec2 = torch.randn(5)
    got = torch.addr(
        self_.to(mojo_device),
        vec1.to(mojo_device),
        vec2.to(mojo_device),
        beta=0.5,
        alpha=2.0,
    ).cpu()
    torch.testing.assert_close(got, torch.addr(self_, vec1, vec2, beta=0.5, alpha=2.0))


def test_addr_integer_uses_the_composite(mojo_device):
    """A dtype the fused kernel does not cover: the op falls back to ATen's
    own `math_addr` composition instead of declining, so support is
    unchanged."""
    self_ = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    vec1 = torch.arange(3, dtype=torch.int64)
    vec2 = torch.arange(4, dtype=torch.int64)
    got = torch.addr(
        self_.to(mojo_device), vec1.to(mojo_device), vec2.to(mojo_device)
    ).cpu()
    torch.testing.assert_close(got, torch.addr(self_, vec1, vec2))


# --- convolution --------------------------------------------------------------


def test_conv2d_basic(mojo_device, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_convolution)
    x = torch.randn(1, 3, 16, 16)
    w = torch.randn(6, 3, 5, 5)
    bias = torch.randn(6)
    got = torch.nn.functional.conv2d(
        x.to(mojo_device), w.to(mojo_device), bias=bias.to(mojo_device), padding=2
    ).cpu()
    torch.testing.assert_close(
        got,
        torch.nn.functional.conv2d(x, w, bias=bias, padding=2),
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"stride": 2},
        {"padding": 1},
        {"stride": (2, 1), "padding": (1, 2)},
        {"dilation": 2},
        {"stride": 2, "padding": 2, "dilation": 2},
    ],
)
def test_conv2d_geometry(mojo_device, kwargs):
    x = torch.randn(2, 3, 12, 14)
    w = torch.randn(4, 3, 3, 3)
    got = torch.nn.functional.conv2d(x.to(mojo_device), w.to(mojo_device), **kwargs)
    ref = torch.nn.functional.conv2d(x, w, **kwargs)
    assert got.shape == ref.shape
    torch.testing.assert_close(got.cpu(), ref, atol=1e-4, rtol=1e-3)


def test_conv2d_1x1_reuses_the_input_as_the_patch_matrix(mojo_device):
    """A 1x1 stride-1 conv needs no im2col: NCHW already is the col matrix."""
    x = torch.randn(2, 5, 7, 9)
    w = torch.randn(3, 5, 1, 1)
    got = torch.nn.functional.conv2d(x.to(mojo_device), w.to(mojo_device)).cpu()
    torch.testing.assert_close(
        got, torch.nn.functional.conv2d(x, w), atol=1e-4, rtol=1e-3
    )


def test_conv2d_grouped(mojo_device):
    x = torch.randn(2, 6, 10, 10)
    w = torch.randn(8, 3, 3, 3)
    bias = torch.randn(8)
    got = torch.nn.functional.conv2d(
        x.to(mojo_device), w.to(mojo_device), bias=bias.to(mojo_device), groups=2
    ).cpu()
    torch.testing.assert_close(
        got, torch.nn.functional.conv2d(x, w, bias=bias, groups=2), atol=1e-4, rtol=1e-3
    )


def test_conv2d_depthwise(mojo_device):
    x = torch.randn(1, 4, 8, 8)
    w = torch.randn(4, 1, 3, 3)
    got = torch.nn.functional.conv2d(
        x.to(mojo_device), w.to(mojo_device), groups=4, padding=1
    ).cpu()
    torch.testing.assert_close(
        got, torch.nn.functional.conv2d(x, w, groups=4, padding=1), atol=1e-4, rtol=1e-3
    )


def test_conv_transposed_declines(mojo_device):
    """The transposed forward has no kernel here: it must raise, not produce
    a plain convolution."""
    x = torch.randn(1, 3, 8, 8).to(mojo_device)
    w = torch.randn(3, 2, 3, 3).to(mojo_device)
    with pytest.raises(NotImplementedError):
        torch.nn.functional.conv_transpose2d(x, w)


# --- torch.matmul decomposes onto the registered ops --------------------------


def test_matmul_decomposes_to_mm_and_bmm(mojo_device):
    """aten::matmul is CompositeImplicitAutograd: it is not registered here,
    it reaches mm / bmm."""
    a2, b2 = torch.randn(6, 8), torch.randn(8, 4)
    with assert_ran("aten::mm"):
        got2 = torch.matmul(a2.to(mojo_device), b2.to(mojo_device)).cpu()
    torch.testing.assert_close(got2, a2 @ b2, atol=1e-4, rtol=1e-4)

    a3, b3 = torch.randn(3, 6, 8), torch.randn(3, 8, 4)
    with assert_ran("aten::bmm"):
        got3 = torch.matmul(a3.to(mojo_device), b3.to(mojo_device)).cpu()
    torch.testing.assert_close(got3, a3 @ b3, atol=1e-4, rtol=1e-4)


# --- the architecture-gated tensor-core bridges -------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("op", ["mm", "addmm", "linear", "bmm"])
def test_gemm16_entry_points(mojo_h100, dtype, op):
    """Every 16-bit entry point on aligned shapes, where the warp-specialized
    tensor-core routes are the ones that take the call."""
    m, n, k, batch = 128, 192, 256, 3

    def dev(*shape: int) -> torch.Tensor:
        return torch.randn(*shape).to(dtype).to(mojo_h100)

    if op == "mm":
        a, b = dev(m, k), dev(k, n)
        got, ref = torch.mm(a, b), a.cpu().float() @ b.cpu().float()
    elif op == "addmm":
        a, b, c = dev(m, k), dev(k, n), dev(n)
        got = torch.addmm(c, a, b)
        ref = a.cpu().float() @ b.cpu().float() + c.cpu().float()
    elif op == "linear":
        x, w, c = dev(2, m, k), dev(n, k), dev(n)
        got = torch.nn.functional.linear(x, w, c)
        ref = torch.nn.functional.linear(
            x.cpu().float(), w.cpu().float(), c.cpu().float()
        )
    else:
        a, b = dev(batch, m, k), dev(batch, k, n)
        got, ref = torch.bmm(a, b), torch.bmm(a.cpu().float(), b.cpu().float())

    assert got.dtype == dtype
    torch.testing.assert_close(got.cpu().float(), ref, atol=2e-1, rtol=2e-2)


def test_tf32_bridge_opt_in(mojo_h100, monkeypatch):
    """fp32 stays on the strict SIMT path by default (TF32 drops mantissa
    bits) and only the explicit opt-in reaches the tensor-core route."""
    a = torch.randn(128, 256)
    b = torch.randn(256, 192)
    ref = a @ b
    monkeypatch.setenv("TORCH_MOJO_BACKEND_TF32", "1")
    got = torch.mm(a.to(mojo_h100), b.to(mojo_h100)).cpu()
    # TF32 keeps 10 mantissa bits, so the tolerance is bf16-like, not fp32.
    torch.testing.assert_close(got, ref, atol=2e-1, rtol=2e-2)
    monkeypatch.delenv("TORCH_MOJO_BACKEND_TF32")
    assert os.environ.get("TORCH_MOJO_BACKEND_TF32") is None
    strict = torch.mm(a.to(mojo_h100), b.to(mojo_h100)).cpu()
    torch.testing.assert_close(strict, ref, atol=1e-3, rtol=1e-4)
