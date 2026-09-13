"""Binary arithmetic on the native mojo device: add/sub/mul/div, pow,
maximum/minimum, remainder, floor_divide, lerp, addcmul/addcdiv, clamp and
the logical/bitwise ops, with their in-place and `out=` variants.

Every check compares against the same computation on CPU torch through the
public API only; `CallChecker` (or `native.op_count` for the ops with no
`aten_functions` twin) asserts the native op actually ran.
"""

import contextlib

import pytest
import torch

from torch_mojo_backend import aten_functions, native, register_mojo_devices


@pytest.fixture(autouse=True)
def _mojo_registered():
    """tests/native has no conftest of its own; the device has to exist
    before any of these tests runs (registration is idempotent)."""
    register_mojo_devices()


@contextlib.contextmanager
def native_ran(*op_names: str):
    """Assert at least one of `op_names` ran as a native boxed kernel."""
    native.op_counting(True)
    before = {name: native.op_count(name) for name in op_names}
    yield
    assert any(native.op_count(name) > before[name] for name in op_names), (
        f"none of {op_names} ran natively"
    )


def _both(shape, dtype, device, *, low=1, high=9):
    """The same tensor on CPU and on the mojo device."""
    if dtype.is_floating_point:
        cpu = torch.randn(shape, dtype=torch.float32).to(dtype)
    else:
        cpu = torch.randint(low, high, shape, dtype=dtype)
    return cpu, cpu.to(device)


# --------------------------------------------------------------------------
# add / sub / mul / div
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_add_sub_mul_tensor(mojo_device, dtype, call_checker):
    call_checker.register(aten_functions.aten_add)
    a_cpu, a = _both((4, 5), dtype, mojo_device)
    b_cpu, b = _both((4, 5), dtype, mojo_device)
    torch.testing.assert_close((a + b).cpu(), a_cpu + b_cpu)
    torch.testing.assert_close((a - b).cpu(), a_cpu - b_cpu)
    torch.testing.assert_close((a * b).cpu(), a_cpu * b_cpu)


def test_div_float(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_div)
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 4), torch.float32, mojo_device)
    torch.testing.assert_close((a / b).cpu(), a_cpu / b_cpu)


def test_div_int_promotes_to_float(mojo_device):
    a_cpu, a = _both((6,), torch.int64, mojo_device)
    b_cpu, b = _both((6,), torch.int64, mojo_device)
    with native_ran("aten::div.Tensor"):
        out = a / b
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), a_cpu / b_cpu)
    with native_ran("aten::div.Tensor"):
        out_scalar = a / 2
    torch.testing.assert_close(out_scalar.cpu(), a_cpu / 2)


def test_add_alpha(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_add)
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 4), torch.float32, mojo_device)
    torch.testing.assert_close(
        torch.add(a, b, alpha=-2.5).cpu(), torch.add(a_cpu, b_cpu, alpha=-2.5)
    )
    torch.testing.assert_close(
        torch.sub(a, b, alpha=3).cpu(), torch.sub(a_cpu, b_cpu, alpha=3)
    )


def test_scalar_operands(mojo_device):
    """`x + 2` reaches the backend as add.Tensor with a wrapped 0-d CPU
    tensor; the scalar routes have to recognise it."""
    a_cpu, a = _both((5,), torch.float32, mojo_device)
    with native_ran("aten::add.Tensor"):
        torch.testing.assert_close((a + 2).cpu(), a_cpu + 2)
    torch.testing.assert_close((2 + a).cpu(), 2 + a_cpu)
    torch.testing.assert_close((a * 1.5).cpu(), a_cpu * 1.5)
    torch.testing.assert_close((a - 0.25).cpu(), a_cpu - 0.25)
    torch.testing.assert_close((a / 4).cpu(), a_cpu / 4)
    torch.testing.assert_close((a**2).cpu(), a_cpu**2)


def test_int_scalar_operands(mojo_device):
    a_cpu, a = _both((5,), torch.int64, mojo_device)
    with native_ran("aten::add.Tensor"):
        torch.testing.assert_close((a + 3).cpu(), a_cpu + 3)
    torch.testing.assert_close((a * 3).cpu(), a_cpu * 3)
    torch.testing.assert_close((a - 3).cpu(), a_cpu - 3)


def test_broadcasting(mojo_device):
    a_cpu, a = _both((3, 1, 5), torch.float32, mojo_device)
    b_cpu, b = _both((4, 5), torch.float32, mojo_device)
    with native_ran("aten::mul.Tensor"):
        out = a * b
    assert tuple(out.shape) == (3, 4, 5)
    torch.testing.assert_close(out.cpu(), a_cpu * b_cpu)


def test_strided_operands(mojo_device):
    a_cpu, a = _both((4, 6), torch.float32, mojo_device)
    b_cpu, b = _both((6, 4), torch.float32, mojo_device)
    with native_ran("aten::add.Tensor"):
        out = a + b.t()
    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu.t())


def test_rank5_equal_shapes(mojo_device):
    """Above rank 4 the kernel takes a flat pass: equal shapes, contiguous."""
    a_cpu, a = _both((2, 2, 2, 2, 3), torch.float32, mojo_device)
    b_cpu, b = _both((2, 2, 2, 2, 3), torch.float32, mojo_device)
    with native_ran("aten::mul.Tensor"):
        out = a * b
    torch.testing.assert_close(out.cpu(), a_cpu * b_cpu)


def test_mixed_dtype_promotion(mojo_device):
    a_cpu, a = _both((4, 4), torch.float32, mojo_device)
    b_cpu, b = _both((4, 4), torch.bfloat16, mojo_device)
    with native_ran("aten::add.Tensor"):
        out = a + b
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu)
    i_cpu, i = _both((4, 4), torch.int32, mojo_device)
    j_cpu, j = _both((4, 4), torch.int64, mojo_device)
    mixed = i + j
    assert mixed.dtype == torch.int64
    torch.testing.assert_close(mixed.cpu(), i_cpu + j_cpu)


def test_non_contiguous_mixed_dtype(mojo_device):
    a_cpu, a = _both((4, 6), torch.float32, mojo_device)
    b_cpu, b = _both((6, 4), torch.bfloat16, mojo_device)
    out = a + b.t()
    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu.t())


def test_unsupported_mix_declines(mojo_device):
    """A promotion the port does not cover declines (NotImplementedError),
    exactly where the old fast path returned NOT_HANDLED."""
    _, a = _both((4,), torch.float32, mojo_device)
    _, b = _both((4,), torch.int64, mojo_device)
    with pytest.raises(NotImplementedError):
        a + b


def test_shape_mismatch_raises(mojo_device):
    _, a = _both((4,), torch.float32, mojo_device)
    _, b = _both((5,), torch.float32, mojo_device)
    with pytest.raises(RuntimeError):
        a + b


# --------------------------------------------------------------------------
# in-place
# --------------------------------------------------------------------------


def test_inplace_tensor(mojo_device):
    a_cpu, a = _both((4, 5), torch.float32, mojo_device)
    b_cpu, b = _both((4, 5), torch.float32, mojo_device)
    with native_ran("aten::add_.Tensor"):
        a.add_(b)
    a_cpu.add_(b_cpu)
    torch.testing.assert_close(a.cpu(), a_cpu)
    with native_ran("aten::mul_.Tensor"):
        a.mul_(b)
    a_cpu.mul_(b_cpu)
    torch.testing.assert_close(a.cpu(), a_cpu)
    with native_ran("aten::sub_.Tensor"):
        a.sub_(b)
    a_cpu.sub_(b_cpu)
    torch.testing.assert_close(a.cpu(), a_cpu)


def test_inplace_scalar(mojo_device):
    a_cpu, a = _both((7,), torch.float32, mojo_device)
    with native_ran("aten::add_.Tensor"):
        a.add_(1.5)
    a_cpu.add_(1.5)
    torch.testing.assert_close(a.cpu(), a_cpu)
    a.mul_(2.0)
    a_cpu.mul_(2.0)
    torch.testing.assert_close(a.cpu(), a_cpu)
    a.sub_(0.5)
    a_cpu.sub_(0.5)
    torch.testing.assert_close(a.cpu(), a_cpu)
    a.add_(b := torch.full((7,), 2.0).to(mojo_device), alpha=3)
    a_cpu.add_(torch.full((7,), 2.0), alpha=3)
    assert b.shape == a.shape
    torch.testing.assert_close(a.cpu(), a_cpu)


def test_inplace_on_a_view(mojo_device):
    a_cpu, a = _both((4, 5), torch.float32, mojo_device)
    b_cpu, b = _both((4,), torch.float32, mojo_device)
    a[:, 1].add_(b)
    a_cpu[:, 1].add_(b_cpu)
    torch.testing.assert_close(a.cpu(), a_cpu)


# --------------------------------------------------------------------------
# out=
# --------------------------------------------------------------------------


def test_out_variants(mojo_device):
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 4), torch.float32, mojo_device)
    dest = torch.empty((3, 4), device=mojo_device)
    with native_ran("aten::add.out"):
        torch.add(a, b, out=dest)
    torch.testing.assert_close(dest.cpu(), a_cpu + b_cpu)
    with native_ran("aten::mul.out"):
        torch.mul(a, b, out=dest)
    torch.testing.assert_close(dest.cpu(), a_cpu * b_cpu)
    with native_ran("aten::sub.out"):
        torch.sub(a, b, out=dest)
    torch.testing.assert_close(dest.cpu(), a_cpu - b_cpu)
    with native_ran("aten::div.out"):
        torch.div(a, b, out=dest)
    torch.testing.assert_close(dest.cpu(), a_cpu / b_cpu)


def test_out_resizes(mojo_device):
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 4), torch.float32, mojo_device)
    dest = torch.empty(0, device=mojo_device)
    torch.add(a, b, out=dest)
    assert tuple(dest.shape) == (3, 4)
    torch.testing.assert_close(dest.cpu(), a_cpu + b_cpu)


def test_out_aliasing_an_input(mojo_device):
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 4), torch.float32, mojo_device)
    torch.add(a, b, out=a)
    torch.testing.assert_close(a.cpu(), a_cpu + b_cpu)


def test_out_strided(mojo_device):
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 4), torch.float32, mojo_device)
    dest = torch.zeros((3, 8), device=mojo_device)[:, ::2]
    torch.add(a, b, out=dest)
    torch.testing.assert_close(dest.cpu(), a_cpu + b_cpu)


def test_div_out_mode(mojo_device):
    a_cpu, a = _both((6,), torch.float32, mojo_device)
    b_cpu, b = _both((6,), torch.float32, mojo_device)
    dest = torch.empty((6,), device=mojo_device)
    with native_ran("aten::div.out_mode"):
        torch.div(a, b, rounding_mode="floor", out=dest)
    torch.testing.assert_close(
        dest.cpu(), torch.div(a_cpu, b_cpu, rounding_mode="floor")
    )


# --------------------------------------------------------------------------
# div rounding modes, floor_divide, remainder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["floor", "trunc"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_div_rounding_modes(mojo_device, mode, dtype):
    a_cpu, a = _both((8,), dtype, mojo_device, low=-9, high=9)
    b_cpu, b = _both((8,), dtype, mojo_device, low=1, high=5)
    with native_ran("aten::div.Tensor_mode"):
        out = torch.div(a, b, rounding_mode=mode)
    assert out.dtype == dtype
    torch.testing.assert_close(out.cpu(), torch.div(a_cpu, b_cpu, rounding_mode=mode))


def test_floor_divide(mojo_device):
    a_cpu, a = _both((8,), torch.int64, mojo_device, low=-9, high=9)
    b_cpu, b = _both((8,), torch.int64, mojo_device, low=1, high=5)
    with native_ran("aten::floor_divide"):
        out = a // b
    torch.testing.assert_close(out.cpu(), a_cpu // b_cpu)
    with native_ran("aten::floor_divide", "aten::floor_divide.Scalar"):
        out_scalar = a // 3
    torch.testing.assert_close(out_scalar.cpu(), a_cpu // 3)


def test_remainder(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_remainder)
    a_cpu, a = _both((8,), torch.float32, mojo_device)
    b_cpu, b = _both((8,), torch.float32, mojo_device, low=1, high=4)
    torch.testing.assert_close(torch.remainder(a, b).cpu(), a_cpu % b_cpu)
    torch.testing.assert_close(torch.remainder(a, 2.0).cpu(), a_cpu % 2.0)
    torch.testing.assert_close(
        torch.remainder(2.0, b).cpu(), torch.remainder(2.0, b_cpu)
    )


# --------------------------------------------------------------------------
# pow / maximum / minimum / clamp
# --------------------------------------------------------------------------


def test_pow(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_pow)
    # Positive base: the tensor-tensor exponent is real-valued only there
    # (and `abs` belongs to another op group, so build it on CPU).
    a_cpu = torch.rand(5) + 0.5
    a = a_cpu.to(mojo_device)
    torch.testing.assert_close(torch.pow(a, 2.0).cpu(), torch.pow(a_cpu, 2.0))
    e_cpu, e = _both((5,), torch.float32, mojo_device)
    with native_ran("aten::pow.Tensor_Tensor"):
        out = torch.pow(a, e)
    torch.testing.assert_close(out.cpu(), torch.pow(a_cpu, e_cpu))


def test_maximum_minimum(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_maximum)
    a_cpu, a = _both((4, 4), torch.float32, mojo_device)
    b_cpu, b = _both((4, 4), torch.float32, mojo_device)
    torch.testing.assert_close(torch.maximum(a, b).cpu(), torch.maximum(a_cpu, b_cpu))
    torch.testing.assert_close(torch.minimum(a, b).cpu(), torch.minimum(a_cpu, b_cpu))


def test_clamp(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_clamp)
    a_cpu, a = _both((10,), torch.float32, mojo_device)
    torch.testing.assert_close(a.clamp(-0.5, 0.5).cpu(), a_cpu.clamp(-0.5, 0.5))
    torch.testing.assert_close(a.clamp(min=0.0).cpu(), a_cpu.clamp(min=0.0))
    torch.testing.assert_close(a.clamp(max=0.0).cpu(), a_cpu.clamp(max=0.0))


# --------------------------------------------------------------------------
# addcmul / addcdiv / lerp
# --------------------------------------------------------------------------


def test_addcmul_addcdiv(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_addcmul)
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 4), torch.float32, mojo_device)
    c_cpu, c = _both((3, 4), torch.float32, mojo_device, low=1, high=5)
    torch.testing.assert_close(
        torch.addcmul(a, b, c, value=0.5).cpu(),
        torch.addcmul(a_cpu, b_cpu, c_cpu, value=0.5),
    )
    torch.testing.assert_close(
        torch.addcdiv(a, b, c, value=2.0).cpu(),
        torch.addcdiv(a_cpu, b_cpu, c_cpu, value=2.0),
    )
    dest = torch.empty((3, 4), device=mojo_device)
    with native_ran("aten::addcmul.out"):
        torch.addcmul(a, b, c, value=0.5, out=dest)
    torch.testing.assert_close(
        dest.cpu(), torch.addcmul(a_cpu, b_cpu, c_cpu, value=0.5)
    )
    with native_ran("aten::addcdiv.out"):
        torch.addcdiv(a, b, c, value=2.0, out=dest)
    torch.testing.assert_close(
        dest.cpu(), torch.addcdiv(a_cpu, b_cpu, c_cpu, value=2.0)
    )


def test_addcmul_broadcast(mojo_device):
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 1), torch.float32, mojo_device)
    c_cpu, c = _both((1, 4), torch.float32, mojo_device)
    with native_ran("aten::addcmul"):
        out = torch.addcmul(a, b, c, value=-1.0)
    torch.testing.assert_close(
        out.cpu(), torch.addcmul(a_cpu, b_cpu, c_cpu, value=-1.0)
    )


@pytest.mark.parametrize("weight", [0.25, 0.75])
def test_lerp(mojo_device, weight):
    a_cpu, a = _both((6,), torch.float32, mojo_device)
    b_cpu, b = _both((6,), torch.float32, mojo_device)
    with native_ran("aten::lerp.Scalar"):
        out = torch.lerp(a, b, weight)
    torch.testing.assert_close(out.cpu(), torch.lerp(a_cpu, b_cpu, weight))
    dest = torch.empty((6,), device=mojo_device)
    with native_ran("aten::lerp.Scalar_out"):
        torch.lerp(a, b, weight, out=dest)
    torch.testing.assert_close(dest.cpu(), torch.lerp(a_cpu, b_cpu, weight))


# --------------------------------------------------------------------------
# logical / bitwise
# --------------------------------------------------------------------------


def test_logical_and_xor(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_logical_and)
    a_cpu = torch.tensor([True, False, True, False])
    b_cpu = torch.tensor([True, True, False, False])
    a, b = a_cpu.to(mojo_device), b_cpu.to(mojo_device)
    torch.testing.assert_close(
        torch.logical_and(a, b).cpu(), torch.logical_and(a_cpu, b_cpu)
    )
    torch.testing.assert_close(
        torch.logical_xor(a, b).cpu(), torch.logical_xor(a_cpu, b_cpu)
    )


def test_logical_mixed_dtypes(mojo_device):
    a_cpu = torch.tensor([0.0, 1.5, 0.0, -2.0])
    b_cpu = torch.tensor([1, 0, 0, 7], dtype=torch.int64)
    a, b = a_cpu.to(mojo_device), b_cpu.to(mojo_device)
    with native_ran("aten::logical_and"):
        out = torch.logical_and(a, b)
    torch.testing.assert_close(out.cpu(), torch.logical_and(a_cpu, b_cpu))


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_bitwise(mojo_device, dtype, call_checker):
    call_checker.register(aten_functions.aten_bitwise_and)
    if dtype == torch.bool:
        a_cpu = torch.tensor([True, False, True, False])
        b_cpu = torch.tensor([True, True, False, False])
    else:
        a_cpu = torch.tensor([1, 2, 3, 12], dtype=dtype)
        b_cpu = torch.tensor([3, 3, 1, 10], dtype=dtype)
    a, b = a_cpu.to(mojo_device), b_cpu.to(mojo_device)
    torch.testing.assert_close((a & b).cpu(), a_cpu & b_cpu)
    torch.testing.assert_close((a | b).cpu(), a_cpu | b_cpu)
    torch.testing.assert_close((a ^ b).cpu(), a_cpu ^ b_cpu)


def test_bitwise_scalar(mojo_device):
    a_cpu = torch.tensor([1, 2, 3, 12], dtype=torch.int64)
    a = a_cpu.to(mojo_device)
    with native_ran("aten::bitwise_and.Scalar"):
        out = a & 3
    torch.testing.assert_close(out.cpu(), a_cpu & 3)


# --------------------------------------------------------------------------
# GPU-only routes
# --------------------------------------------------------------------------


def test_add_f32_bf16_fused_route(mojo_gpu):
    """FP32 + BF16 -> FP32 in one launch, no materialized BF16 operand."""
    a_cpu, a = _both((64, 32), torch.float32, mojo_gpu)
    b_cpu, b = _both((64, 32), torch.bfloat16, mojo_gpu)
    with native_ran("aten::add.Tensor"):
        out = a + b
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu)


def test_autograd_through_binary_ops(mojo_gpu):
    x = torch.randn(4).to(mojo_gpu).requires_grad_()
    y = torch.randn(4).to(mojo_gpu).requires_grad_()
    # `.backward(grad)` rather than `.sum().backward()`: reductions are
    # another op group.
    ((x * y) + x * 2.0).backward(torch.ones(4).to(mojo_gpu))
    torch.testing.assert_close(x.grad.cpu(), (y + 2.0).detach().cpu())
    torch.testing.assert_close(y.grad.cpu(), x.detach().cpu())
