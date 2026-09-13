"""Tests for the native `unary` op group
(torch_mojo_backend/native/mojo/ops_unary.mojo): abs/neg/sign/relu, the
transcendental unary ops, ceil/floor, gelu(+backward), isnan/logical_not/
bitwise_not and fill.Scalar.

Public torch API only, run against the mojo device: `mojo_gpu`/`mojo_device`
(tests/conftest.py) call `register_mojo_devices()`, which now registers the
*native* backend (`torch_mojo_backend/mojo_device/register.py` ->
`native.register()`) rather than the old Python eager path.
"""

import pytest
import torch
import torch.nn.functional as F

from torch_mojo_backend import aten_functions, native
from torch_mojo_backend.native import device_module


def _native_count(name: str) -> int:
    return native.op_count(f"aten::{name}")


def _reset_native_counts():
    native.op_counting(True)
    native.op_counts_reset()


def _tol(dtype: torch.dtype) -> tuple[float | None, float | None]:
    if dtype in (torch.float16, torch.bfloat16):
        return 3e-2, 3e-2
    return None, None


# One dtype (bfloat16) sweeps every op below for correctness + native-dispatch
# coverage; a handful of ops are re-exercised in float32 by the .out/relu_/
# call_checker tests further down, so float32 kernel specializations for
# those get built too without a second full sweep. float16 shares the exact
# same dtype-gate code path as bfloat16 (`_is_float_dtype`), so it is not
# spot-checked separately.
SWEEP_DTYPE = torch.bfloat16

# (native op name, torch callable, input-domain key)
_UNARY_OPS = [
    ("abs", torch.abs, "signed"),
    ("acos", torch.acos, "unit"),
    ("asinh", torch.asinh, "signed"),
    ("atanh", torch.atanh, "unit"),
    ("cos", torch.cos, "signed"),
    ("cosh", torch.cosh, "signed"),
    ("erf", torch.erf, "signed"),
    ("exp", torch.exp, "signed"),
    ("log", torch.log, "positive"),
    ("log1p", torch.log1p, "above_neg1"),
    ("neg", torch.neg, "signed"),
    ("reciprocal", torch.reciprocal, "positive"),
    ("rsqrt", torch.rsqrt, "positive"),
    ("sigmoid", torch.sigmoid, "signed"),
    ("sign", torch.sign, "signed"),
    ("silu", F.silu, "signed"),
    ("sin", torch.sin, "signed"),
    ("sinh", torch.sinh, "signed"),
    ("sqrt", torch.sqrt, "positive"),
    ("tan", torch.tan, "unit"),
    ("tanh", torch.tanh, "signed"),
    ("relu", torch.relu, "signed"),
]


def _sample(domain: str, shape: tuple[int, ...]) -> torch.Tensor:
    torch.manual_seed(0)
    if domain == "unit":
        return torch.empty(shape, dtype=torch.float64).uniform_(-0.85, 0.85)
    if domain == "positive":
        return torch.empty(shape, dtype=torch.float64).uniform_(0.15, 4.0)
    if domain == "above_neg1":
        # log1p(x) = log(1 + x) needs x > -1; stay well clear of the pole.
        return torch.empty(shape, dtype=torch.float64).uniform_(-0.9, 4.0)
    return torch.randn(shape, dtype=torch.float64) * 2


@pytest.mark.parametrize("op_name,fn,domain", _UNARY_OPS)
def test_unary_matches_cpu(mojo_gpu, op_name, fn, domain):
    x64 = _sample(domain, (3, 5))
    expected = fn(x64).to(SWEEP_DTYPE)
    x = x64.to(SWEEP_DTYPE).to(mojo_gpu)
    _reset_native_counts()
    y = fn(x)
    assert y.device.type == "mojo"
    assert _native_count(op_name) > 0, f"aten::{op_name} did not run natively"
    rtol, atol = _tol(SWEEP_DTYPE)
    torch.testing.assert_close(y.cpu(), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "op_name,fn", [("abs", torch.abs), ("exp", torch.exp), ("sign", torch.sign)]
)
def test_unary_on_cpu_and_gpu_device(mojo_device, op_name, fn):
    """The elementwise spec kernels run on the MAX CPU device too (mojo:cpu),
    not only accelerators — `mojo_device` covers both legs."""
    x64 = torch.randn(3, 5, dtype=torch.float64) * 2
    expected = fn(x64).to(torch.float32)
    x = x64.to(torch.float32).to(mojo_device)
    _reset_native_counts()
    y = fn(x)
    assert _native_count(op_name) > 0
    torch.testing.assert_close(y.cpu(), expected)


def test_unary_noncontiguous_input(mojo_gpu):
    x64 = torch.randn(4, 6, dtype=torch.float64)
    x_cpu = x64.to(torch.float32)
    x = x_cpu.to(mojo_gpu).t()
    assert not x.is_contiguous()
    y = torch.exp(x)
    torch.testing.assert_close(y.cpu(), torch.exp(x_cpu.t()))


_DIRECT_OPS = [
    ("abs", torch.abs),
    ("neg", torch.neg),
    ("sign", torch.sign),
    ("relu", torch.relu),
]
_DIRECT_INT_DTYPES = (torch.int32, torch.uint8)


@pytest.mark.parametrize("dtype", _DIRECT_INT_DTYPES)
@pytest.mark.parametrize("op_name,fn", _DIRECT_OPS)
def test_direct_ops_int_dtypes(mojo_gpu, op_name, fn, dtype):
    lo, hi = (0, 6) if dtype == torch.uint8 else (-6, 6)
    x_cpu = torch.randint(lo, hi, (4, 5)).to(dtype)
    x = x_cpu.to(mojo_gpu)
    _reset_native_counts()
    y = fn(x)
    assert _native_count(op_name) > 0
    assert y.cpu().tolist() == fn(x_cpu).tolist()


@pytest.mark.parametrize(
    "op_name,fn", [("abs", torch.abs), ("exp", torch.exp), ("sigmoid", torch.sigmoid)]
)
def test_out_variant(mojo_gpu, op_name, fn):
    x64 = torch.randn(3, 4, dtype=torch.float64)
    expected = fn(x64).to(torch.float32)
    x = x64.to(torch.float32).to(mojo_gpu)
    out_name = f"{op_name}.out"

    # Ready out tensor (contiguous, right shape/dtype): compute straight into it.
    out1 = torch.empty(3, 4, device=mojo_gpu)
    _reset_native_counts()
    ret1 = fn(x, out=out1)
    assert _native_count(out_name) > 0
    assert ret1.data_ptr() == out1.data_ptr()
    torch.testing.assert_close(out1.cpu(), expected)

    # Non-contiguous out tensor: compute into a temporary then copy_strided_into.
    out2 = torch.empty(4, 3, device=mojo_gpu).t()
    assert not out2.is_contiguous()
    _reset_native_counts()
    fn(x, out=out2)
    assert _native_count(out_name) > 0
    # `.cpu()` on a strided mojo tensor would itself need a strided host
    # destination, which `_copy_from` (ops_core.mojo, outside this group)
    # does not support; materialize contiguous on-device first.
    torch.testing.assert_close(out2.contiguous().cpu(), expected)


def test_relu_inplace(mojo_gpu):
    x64 = torch.randn(3, 4, dtype=torch.float64)
    expected = torch.relu(x64).to(torch.float32)
    x = x64.to(torch.float32).to(mojo_gpu)
    _reset_native_counts()
    ptr_before = x.data_ptr()
    ret = x.relu_()
    assert _native_count("relu_") > 0
    assert ret.data_ptr() == ptr_before
    torch.testing.assert_close(x.cpu(), expected)


def test_relu_inplace_noncontiguous(mojo_gpu):
    x64 = torch.randn(4, 3, dtype=torch.float64)
    expected = torch.relu(x64.t()).to(torch.float32)
    x = x64.to(torch.float32).to(mojo_gpu).t()
    assert not x.is_contiguous()
    ptr_before = x.data_ptr()
    x.relu_()
    assert x.data_ptr() == ptr_before  # in-place: same storage, no reallocation
    # See test_out_variant: `.cpu()` needs a contiguous source for this
    # backend's current `_copy_from` (outside this group).
    torch.testing.assert_close(x.contiguous().cpu(), expected)


@pytest.mark.parametrize("op_name,fn", [("ceil", torch.ceil), ("floor", torch.floor)])
def test_ceil_floor_float(mojo_gpu, op_name, fn):
    x64 = torch.randn(3, 5, dtype=torch.float64) * 3
    expected = fn(x64).to(torch.float32)
    x = x64.to(torch.float32).to(mojo_gpu)
    _reset_native_counts()
    y = fn(x)
    assert _native_count(op_name) > 0
    torch.testing.assert_close(y.cpu(), expected)


@pytest.mark.parametrize("op_name,fn", [("ceil", torch.ceil), ("floor", torch.floor)])
def test_ceil_floor_int_is_identity_copy(mojo_gpu, op_name, fn):
    """ceil/floor of an int tensor is the identity, but still functional: a
    fresh tensor, not the same object (matches aten_fast._int_unary_identity).
    """
    x_cpu = torch.randint(-5, 5, (3, 4), dtype=torch.int64)
    x = x_cpu.to(mojo_gpu)
    y = fn(x)
    assert y.data_ptr() != x.data_ptr()
    assert y.cpu().tolist() == x_cpu.tolist()


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_forward(mojo_gpu, approximate):
    x64 = torch.randn(3, 5, dtype=torch.float64) * 2
    expected = F.gelu(x64, approximate=approximate).to(SWEEP_DTYPE)
    x = x64.to(SWEEP_DTYPE).to(mojo_gpu)
    _reset_native_counts()
    y = F.gelu(x, approximate=approximate)
    assert _native_count("gelu") > 0
    rtol, atol = _tol(SWEEP_DTYPE)
    torch.testing.assert_close(y.cpu(), expected, rtol=rtol, atol=atol)


def test_gelu_out_variant(mojo_gpu):
    x64 = torch.randn(3, 5, dtype=torch.float64) * 2
    expected = F.gelu(x64, approximate="tanh").to(torch.float32)
    x = x64.to(torch.float32).to(mojo_gpu)
    out = torch.empty(3, 5, device=mojo_gpu)
    _reset_native_counts()
    torch.ops.aten.gelu.out(x, approximate="tanh", out=out)
    assert _native_count("gelu.out") > 0
    torch.testing.assert_close(out.cpu(), expected)


def test_gelu_invalid_approximate_declines(mojo_gpu):
    x = torch.randn(3).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        F.gelu(x, approximate="bogus")


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_backward_matches_cpu(mojo_gpu, approximate):
    x64 = (torch.randn(3, 5, dtype=torch.float64) * 2).requires_grad_()
    g64 = torch.randn(3, 5, dtype=torch.float64)
    F.gelu(x64, approximate=approximate).backward(g64)
    assert x64.grad is not None
    expected_grad = x64.grad.to(SWEEP_DTYPE)

    x = x64.detach().to(SWEEP_DTYPE).to(mojo_gpu).requires_grad_()
    g = g64.to(SWEEP_DTYPE).to(mojo_gpu)
    _reset_native_counts()
    F.gelu(x, approximate=approximate).backward(g)
    assert _native_count("gelu_backward") > 0
    assert x.grad is not None
    rtol, atol = _tol(SWEEP_DTYPE)
    torch.testing.assert_close(x.grad.cpu(), expected_grad, rtol=rtol, atol=atol)


def test_gelu_backward_declines_float16(mojo_gpu):
    x = torch.randn(3, 4, dtype=torch.float16).to(mojo_gpu).requires_grad_()
    y = F.gelu(x)
    with pytest.raises(NotImplementedError):
        y.backward(torch.ones_like(y))


def test_gelu_backward_declines_on_cpu_device(mojo_gpu):
    # `mojo_gpu` only guarantees registration + a real accelerator exists;
    # this test deliberately targets the MAX CPU device instead.
    cpu_device = f"mojo:{device_module.device_count() - 1}"
    x = torch.randn(3, 4).to(cpu_device).requires_grad_()
    y = F.gelu(x)
    with pytest.raises(NotImplementedError):
        y.backward(torch.ones_like(y))


def test_missing_backward_kernels_raise_cleanly(mojo_gpu):
    """sigmoid_backward / tanh_backward / threshold_backward (relu's
    backward) have no native kernel (the old eager path preflighted these
    from the forward, see aten_ops/autograd_preflight.py, because a Python
    exception raised inside that backend's autograd engine could abort the
    process). The native backend has no such hazard: an unregistered
    PrivateUse1 op simply raises out of the dispatcher like any missing
    kernel, which this test is here to confirm actually holds rather than
    crashing the interpreter.
    """
    for make_y in (
        lambda x: torch.sigmoid(x),
        lambda x: torch.tanh(x),
        lambda x: torch.relu(x),
    ):
        x = torch.randn(4).to(mojo_gpu).requires_grad_()
        y = make_y(x)
        with pytest.raises((NotImplementedError, RuntimeError)):
            y.backward(torch.ones_like(y))


def test_isnan(mojo_gpu):
    x_cpu = torch.tensor([1.0, float("nan"), -float("inf"), 2.0])
    x = x_cpu.to(mojo_gpu)
    _reset_native_counts()
    y = torch.isnan(x)
    assert _native_count("isnan") > 0
    assert y.cpu().tolist() == torch.isnan(x_cpu).tolist()


def test_logical_not(mojo_gpu):
    x_cpu = torch.tensor([True, False, True, False])
    x = x_cpu.to(mojo_gpu)
    _reset_native_counts()
    y = torch.logical_not(x)
    assert _native_count("logical_not") > 0
    assert y.cpu().tolist() == torch.logical_not(x_cpu).tolist()


@pytest.mark.parametrize("dtype", [torch.int32, torch.uint8, torch.bool])
def test_bitwise_not(mojo_gpu, dtype):
    if dtype is torch.bool:
        x_cpu = torch.tensor([True, False, True])
    else:
        x_cpu = torch.randint(0, 20, (5,)).to(dtype)
    x = x_cpu.to(mojo_gpu)
    _reset_native_counts()
    y = torch.bitwise_not(x)
    assert _native_count("bitwise_not") > 0
    assert y.cpu().tolist() == torch.bitwise_not(x_cpu).tolist()


def test_fill_scalar_functional(mojo_gpu):
    x_cpu = torch.zeros(3, 4)
    x = x_cpu.to(mojo_gpu)
    _reset_native_counts()
    y = torch.fill(x, 7.5)
    assert _native_count("fill.Scalar") > 0
    assert y.data_ptr() != x.data_ptr()  # functional: does not alias self
    assert y.cpu().tolist() == torch.full((3, 4), 7.5).tolist()
    assert x.cpu().tolist() == [[0.0] * 4] * 3  # self left untouched


@pytest.mark.parametrize(
    "aten_fn,torch_fn",
    [
        (aten_functions.aten_abs, torch.abs),
        (aten_functions.aten_exp, torch.exp),
        (aten_functions.aten_sigmoid, torch.sigmoid),
        (aten_functions.aten_relu, torch.relu),
        (aten_functions.aten_isnan, torch.isnan),
    ],
)
def test_call_checker_confirms_native_dispatch(
    mojo_gpu, call_checker, aten_fn, torch_fn
):
    call_checker.register(aten_fn)
    x = torch.randn(3, 4).to(mojo_gpu)
    torch_fn(x)


def test_call_checker_bitwise_not(mojo_gpu, call_checker):
    call_checker.register(aten_functions.aten_bitwise_not)
    x = torch.randint(0, 10, (3, 4), dtype=torch.int32).to(mojo_gpu)
    torch.bitwise_not(x)
