"""Pointwise math and parameterized activations on the native mojo device
(tmb/ops/pointwise.mojo): atan2, hypot, copysign, fmod, frexp, nextafter,
fmax/fmin, gcd/lcm, heaviside, the shifts, logaddexp(2), xlogy, xlog1py,
zeta, the special polynomials, lerp.Tensor, clamp.Tensor, pow.Scalar, and
elu, hardtanh, leaky_relu, softplus, hardshrink, softshrink, hardsigmoid,
hardswish, mish, threshold, logsigmoid, rrelu with their backwards.

Everything is compared with the same computation on CPU torch through the
public API, over edge values (signed zeros, infinities, NaN, huge, tiny,
denormals), broadcasting, strided operands, out= and in-place forms.
"""

import contextlib
import itertools

import pytest
import torch
import torch.nn.functional as F

from torch_mojo_backend import native

FLOATS = [torch.float32, torch.float16, torch.bfloat16]
INTS = [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8]


@contextlib.contextmanager
def ran(*op_names: str):
    native.op_counting(True)
    before = {name: native.op_count(name) for name in op_names}
    yield
    assert any(native.op_count(name) > before[name] for name in op_names), (
        f"none of {op_names} ran natively"
    )


_SPECIAL = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.5,
    -3.75,
    7.0,
    1e-30,
    -1e-30,
    1e30,
    -1e30,
    1e-40,
    float("inf"),
    float("-inf"),
    float("nan"),
]


def _pairs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Every ordered pair of edge values, plus a random bulk."""
    a, b = zip(*itertools.product(_SPECIAL, _SPECIAL))
    torch.manual_seed(0)
    ra = torch.randn(300) * 4
    rb = torch.randn(300) * 4
    x = torch.cat([torch.tensor(a), ra]).to(dtype)
    y = torch.cat([torch.tensor(b), rb]).to(dtype)
    return x, y


def _tol(dtype: torch.dtype, ulps: float = 1.0) -> dict[str, float]:
    if dtype == torch.float32:
        return {"rtol": 1.3e-6 * ulps, "atol": 1e-5}
    if dtype == torch.float16:
        return {"rtol": 1e-3, "atol": 1e-5}
    if dtype == torch.bfloat16:
        return {"rtol": 1.6e-2, "atol": 1e-5}
    return {"rtol": 0.0, "atol": 0.0}


def _close(
    actual: torch.Tensor | None,
    expected: torch.Tensor | None,
    rtol: float | None = None,
    atol: float | None = None,
):
    assert actual is not None and expected is not None
    torch.testing.assert_close(
        actual.cpu(), expected, equal_nan=True, rtol=rtol, atol=atol
    )


# name, torch function, overload names, ulps of float32 tolerance
_BINARY = [
    ("atan2", torch.atan2, ("aten::atan2",), 2),
    ("hypot", torch.hypot, ("aten::hypot",), 1),
    ("copysign", torch.copysign, ("aten::copysign.Tensor",), 0),
    ("fmax", torch.fmax, ("aten::fmax",), 0),
    ("fmin", torch.fmin, ("aten::fmin",), 0),
    ("fmod", torch.fmod, ("aten::fmod.Tensor",), 0),
    ("nextafter", torch.nextafter, ("aten::nextafter",), 0),
    ("logaddexp", torch.logaddexp, ("aten::logaddexp",), 2),
    ("logaddexp2", torch.logaddexp2, ("aten::logaddexp2",), 2),
    ("heaviside", torch.heaviside, ("aten::heaviside",), 0),
    ("xlogy", torch.xlogy, ("aten::xlogy.Tensor",), 2),
    ("xlog1py", torch.special.xlog1py, ("aten::special_xlog1py",), 2),
]


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("name,fn,ops,ulps", _BINARY, ids=[b[0] for b in _BINARY])
def test_binary_math_edges(mojo_gpu, name, fn, ops, ulps, dtype):
    if name == "nextafter" and dtype != torch.float32:
        pytest.skip("CPU torch has no half nextafter to compare with")
    x, y = _pairs(dtype)
    expected = fn(x, y)
    if name == "fmod":
        # CPU's vectorized fmod is x - trunc(x / y) * y, NaN once the quotient
        # overflows (1e30 by 1e-30); CUDA's ::fmod, and ours, is exact, which
        # is what float64 computes for these operands.
        expected = fn(x.double(), y.double()).to(dtype)
    with ran(*ops):
        actual = fn(x.to(mojo_gpu), y.to(mojo_gpu))
    assert actual.dtype == expected.dtype
    _close(actual, expected, **_tol(dtype, ulps))


def test_nextafter_half_bits(mojo_gpu):
    """No CPU half kernel: check the 16-bit patterns against the float32
    ones rounded, which is exact for one ulp steps away from zero."""
    for dtype in (torch.float16, torch.bfloat16):
        x = torch.tensor([1.0, 1.0, -2.0, 0.0, 0.0, 3.0, float("nan")], dtype=dtype)
        y = torch.tensor([2.0, 0.0, 0.0, 1.0, -1.0, 3.0, 1.0], dtype=dtype)
        got = torch.nextafter(x.to(mojo_gpu), y.to(mojo_gpu)).cpu()
        u = torch.int16
        bits = x.view(u)
        step = torch.where((y > x) ^ (x < 0), 1, -1).to(u)
        want = (bits + step).view(dtype)
        want[3] = torch.tensor(0x0001, dtype=u).view(dtype)
        want[4] = torch.tensor(-32767, dtype=u).view(dtype)
        want[5] = 3.0
        want[6] = float("nan")
        _close(got, want, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_binary_math_broadcast_scalar_strided_out(mojo_gpu, dtype):
    torch.manual_seed(1)
    a_cpu = torch.randn(6, 5).to(dtype)
    b_cpu = torch.randn(5).to(dtype)
    a, b = a_cpu.to(mojo_gpu), b_cpu.to(mojo_gpu)
    for fn in (torch.atan2, torch.hypot, torch.copysign, torch.fmod, torch.fmax):
        _close(fn(a, b), fn(a_cpu, b_cpu), **_tol(dtype, 2))
        _close(fn(a.t(), a.t()), fn(a_cpu.t(), a_cpu.t()), **_tol(dtype, 2))
        col = a_cpu[:, :1]
        _close(fn(a, col.to(mojo_gpu)), fn(a_cpu, col), **_tol(dtype, 2))
        zero_d = torch.tensor(0.75, dtype=dtype)
        _close(fn(a, zero_d.to(mojo_gpu)), fn(a_cpu, zero_d), **_tol(dtype, 2))
        out = torch.empty(5, 6, dtype=dtype, device=mojo_gpu).t()
        fn(a, b, out=out)
        _close(out, fn(a_cpu, b_cpu), **_tol(dtype, 2))
    # Python scalars: copysign.Scalar, fmod.Scalar, xlogy's wrapped number
    _close(torch.copysign(a, -1.0), torch.copysign(a_cpu, -1.0))
    _close(torch.fmod(a, 0.7), torch.fmod(a_cpu, 0.7), **_tol(dtype))
    _close(torch.xlogy(a, 2.0), torch.xlogy(a_cpu, 2.0), **_tol(dtype, 2))
    _close(torch.xlogy(2.0, b.abs()), torch.xlogy(2.0, b_cpu.abs()), **_tol(dtype, 2))


def test_binary_math_in_place(mojo_gpu):
    a_cpu = torch.randn(4, 7)
    b_cpu = torch.randn(4, 7)
    for name in ("atan2_", "hypot_", "copysign_", "fmod_", "xlogy_", "nextafter_"):
        x, x_cpu = a_cpu.clone().to(mojo_gpu), a_cpu.clone()
        getattr(x, name)(b_cpu.to(mojo_gpu))
        getattr(x_cpu, name)(b_cpu)
        _close(x, x_cpu, **_tol(torch.float32, 2))


def test_binary_math_int_promotion(mojo_gpu):
    i = torch.arange(-6, 6)
    j = torch.arange(1, 13)
    for fn in (torch.atan2, torch.copysign, torch.xlogy):
        expected = fn(i, j)
        actual = fn(i.to(mojo_gpu), j.to(mojo_gpu))
        assert actual.dtype == expected.dtype == torch.float32
        _close(actual, expected, **_tol(torch.float32, 2))
    for fn in (torch.fmax, torch.fmin, torch.fmod):
        _close(fn(i.to(mojo_gpu), j.to(mojo_gpu)), fn(i, j))
    f = torch.randn(12)
    _close(torch.atan2(i.to(mojo_gpu), f.to(mojo_gpu)), torch.atan2(i, f))


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float16])
def test_heaviside_fmax_fmin_dtypes(mojo_gpu, dtype):
    if dtype == torch.bool:
        x = torch.tensor([True, False, True, False])
        y = torch.tensor([True, True, False, False])
    else:
        x = torch.tensor([-2, 0, 0, 3, 5, -1]).to(dtype)
        y = torch.tensor([7, 1, 0, -2, 5, 4]).to(dtype)
    for fn in (torch.heaviside, torch.fmax, torch.fmin):
        _close(fn(x.to(mojo_gpu), y.to(mojo_gpu)), fn(x, y))


def test_heaviside_rejects_mixed_dtypes(mojo_gpu):
    with pytest.raises(RuntimeError, match="different dtypes"):
        torch.heaviside(
            torch.ones(3, device=mojo_gpu),
            torch.ones(3, dtype=torch.int64, device=mojo_gpu),
        )


@pytest.mark.parametrize("dtype", INTS)
def test_gcd_lcm_shifts_fmod_int(mojo_gpu, dtype):
    torch.manual_seed(2)
    signed = dtype != torch.uint8
    lo = -60 if signed else 0
    a = torch.randint(lo, 60, (200,), dtype=dtype)
    b = torch.randint(lo, 60, (200,), dtype=dtype)
    a[:4] = torch.tensor([0, 0, 12, 7], dtype=dtype)
    b[:4] = torch.tensor([0, 5, 0, 7], dtype=dtype)
    with ran("aten::gcd"):
        _close(torch.gcd(a.to(mojo_gpu), b.to(mojo_gpu)), torch.gcd(a, b))
    _close(torch.lcm(a.to(mojo_gpu), b.to(mojo_gpu)), torch.lcm(a, b))
    nonzero = torch.where(b == 0, torch.ones_like(b), b)
    _close(torch.fmod(a.to(mojo_gpu), nonzero.to(mojo_gpu)), torch.fmod(a, nonzero))
    bits = torch.iinfo(dtype).bits
    shift = torch.randint(0, bits + 4, (200,)).to(dtype)
    if signed:
        shift[:3] = torch.tensor([-1, bits, bits - 1], dtype=dtype)
    with ran("aten::bitwise_left_shift.Tensor", "aten::__lshift__.Tensor"):
        _close(a.to(mojo_gpu) << shift.to(mojo_gpu), a << shift)
    _close(a.to(mojo_gpu) >> shift.to(mojo_gpu), a >> shift)
    _close(
        torch.bitwise_left_shift(a.to(mojo_gpu), shift.to(mojo_gpu)),
        torch.bitwise_left_shift(a, shift),
    )
    _close(a.to(mojo_gpu) << 3, a << 3)
    _close(a.to(mojo_gpu) >> 2, a >> 2)
    x, x_cpu = a.to(mojo_gpu), a.clone()
    x <<= 1
    x_cpu <<= 1
    x >>= shift.to(mojo_gpu)
    x_cpu >>= shift
    _close(x, x_cpu)


@pytest.mark.parametrize("dtype", FLOATS)
def test_frexp(mojo_gpu, dtype):
    x = torch.tensor(_SPECIAL + [3.0, 0.25, -1000.0, 1e-38, 6e-39]).to(dtype)
    x = torch.cat([x, torch.randn(100).to(dtype) * 100])
    m_cpu, e_cpu = torch.frexp(x)
    with ran("aten::frexp.Tensor", "aten::frexp.Tensor_out"):
        m, e = torch.frexp(x.to(mojo_gpu))
    finite = torch.isfinite(x)
    _close(m.cpu()[finite], m_cpu[finite], rtol=0.0, atol=0.0)
    _close(e.cpu()[finite], e_cpu[finite])
    assert e.dtype == torch.int32
    mo = torch.empty(0, dtype=dtype, device=mojo_gpu)
    eo = torch.empty(0, dtype=torch.int32, device=mojo_gpu)
    torch.frexp(x.to(mojo_gpu), out=(mo, eo))
    _close(mo.cpu()[finite], m_cpu[finite], rtol=0.0, atol=0.0)


def test_zeta(mojo_gpu):
    x = torch.tensor([1.0, 0.5, 2.0, 3.5, 2.0, 4.0, 2.0, 1.5, 10.0])
    q = torch.tensor([1.0, 1.0, 1.0, 2.0, -1.0, -2.5, 0.25, 30.0, 0.5])
    torch.manual_seed(3)
    x = torch.cat([x, torch.rand(100) * 6 + 1.01])
    q = torch.cat([q, torch.rand(100) * 5 + 0.1])
    with ran("aten::special_zeta"):
        actual = torch.special.zeta(x.to(mojo_gpu), q.to(mojo_gpu))
    _close(actual, torch.special.zeta(x, q), rtol=4e-6, atol=1e-6)
    _close(
        torch.special.zeta(x.to(mojo_gpu), 2.0),
        torch.special.zeta(x, 2.0),
        rtol=4e-6,
        atol=1e-6,
    )


_POLYS = [
    "chebyshev_polynomial_t",
    "chebyshev_polynomial_u",
    "chebyshev_polynomial_v",
    "chebyshev_polynomial_w",
    "shifted_chebyshev_polynomial_t",
    "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v",
    "shifted_chebyshev_polynomial_w",
    "hermite_polynomial_h",
    "hermite_polynomial_he",
    "laguerre_polynomial_l",
    "legendre_polynomial_p",
]


@pytest.mark.parametrize("name", _POLYS)
def test_special_polynomials(mojo_gpu, name):
    fn = getattr(torch.special, name)
    torch.manual_seed(4)
    x = torch.cat(
        [torch.tensor([-1.0, 1.0, 0.0, 0.5, 2.0]), torch.rand(120) * 2.2 - 1.1]
    )
    n = torch.randint(-1, 14, (x.numel(),)).float()
    n[:5] = torch.tensor([3.0, 4.0, 5.0, 9.0, 3.0])
    with ran(f"aten::special_{name}"):
        actual = fn(x.to(mojo_gpu), n.to(mojo_gpu))
    # n > 6-8 takes cos(n acos x): a few float32 ulps of the argument.
    _close(actual, fn(x, n), rtol=2e-5, atol=2e-5)
    _close(fn(x.to(mojo_gpu), 3), fn(x, 3), rtol=2e-5, atol=2e-5)
    # NaN x: CPU's loop reads garbage there; CUDA (and ours) gives the
    # degree-0 constant, 0 for a negative degree, NaN otherwise.
    nan_x = torch.full((4,), float("nan"), device=mojo_gpu)
    got = fn(nan_x, torch.tensor([0.0, -1.0, 2.0, 7.0], device=mojo_gpu)).cpu()
    _close(
        got, torch.tensor([1.0, 0.0, float("nan"), float("nan")]), rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("dtype", FLOATS)
def test_lerp_tensor(mojo_gpu, dtype):
    torch.manual_seed(5)
    s = torch.randn(40, 3).to(dtype)
    e = torch.randn(40, 3).to(dtype)
    w = (torch.rand(40, 3) * 1.6 - 0.3).to(dtype)
    with ran("aten::lerp.Tensor"):
        actual = torch.lerp(s.to(mojo_gpu), e.to(mojo_gpu), w.to(mojo_gpu))
    _close(actual, torch.lerp(s, e, w), **_tol(dtype, 2))
    wb = w[:1]
    _close(
        torch.lerp(s.to(mojo_gpu), e.to(mojo_gpu), wb.to(mojo_gpu)),
        torch.lerp(s, e, wb),
        **_tol(dtype, 2),
    )
    x, x_cpu = s.to(mojo_gpu), s.clone()
    x.lerp_(e.to(mojo_gpu), w.to(mojo_gpu))
    x_cpu.lerp_(e, w)
    _close(x, x_cpu, **_tol(dtype, 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_clamp_tensor(mojo_gpu, dtype):
    torch.manual_seed(6)
    if dtype.is_floating_point:
        x = torch.cat([torch.tensor(_SPECIAL), torch.randn(40) * 3]).to(dtype)
    else:
        x = torch.randint(-20, 20, (57,), dtype=dtype)
    lo = (torch.arange(x.numel()) % 5 - 3).to(dtype)
    hi = (torch.arange(x.numel()) % 4).to(dtype)
    with ran("aten::clamp.Tensor"):
        actual = torch.clamp(x.to(mojo_gpu), lo.to(mojo_gpu), hi.to(mojo_gpu))
    _close(actual, torch.clamp(x, lo, hi))
    _close(torch.clamp(x.to(mojo_gpu), min=lo.to(mojo_gpu)), torch.clamp(x, min=lo))
    _close(torch.clamp(x.to(mojo_gpu), max=hi.to(mojo_gpu)), torch.clamp(x, max=hi))
    out = torch.empty_like(x, device=mojo_gpu)
    torch.clamp(x.to(mojo_gpu), lo.to(mojo_gpu), hi.to(mojo_gpu), out=out)
    _close(out, torch.clamp(x, lo, hi))
    if dtype.is_floating_point:
        nan_lo = lo.clone()
        nan_lo[3] = float("nan")
        _close(
            torch.clamp(x.to(mojo_gpu), nan_lo.to(mojo_gpu), hi.to(mojo_gpu)),
            torch.clamp(x, nan_lo, hi),
        )


@pytest.mark.parametrize("dtype", FLOATS)
def test_pow_scalar_base(mojo_gpu, dtype):
    e = torch.cat([torch.tensor(_SPECIAL), torch.randn(50) * 3]).to(dtype)
    for base in (2.0, 0.5, 1.0, 10.0, 0.0, -2.0):
        with ran("aten::pow.Scalar"):
            actual = torch.pow(base, e.to(mojo_gpu))
        _close(actual, torch.pow(base, e), **_tol(dtype, 2))
    out = torch.empty_like(e, device=mojo_gpu)
    torch.pow(3.0, e.to(mojo_gpu), out=out)
    _close(out, torch.pow(3.0, e), **_tol(dtype, 2))


# --------------------------------------------------------------------------
# activations
# --------------------------------------------------------------------------


def _in(x: torch.Tensor, value: float) -> float:
    """`value` rounded to x's dtype: CUDA's shrink/threshold kernels compare
    against `value.to<scalar_t>()`, CPU's reduced-float ones against the
    float value; passing the rounded value makes both agree."""
    return torch.tensor(value, dtype=x.dtype).item()


_ACT = [
    ("elu", lambda x: F.elu(x), "aten::elu"),
    ("elu_params", lambda x: torch.ops.aten.elu(x, 0.7, 1.3, 0.8), "aten::elu"),
    ("selu", lambda x: F.selu(x), "aten::elu"),
    ("celu", lambda x: F.celu(x, 1.5), "aten::elu"),
    ("hardtanh", lambda x: F.hardtanh(x, -0.4, 1.7), "aten::hardtanh"),
    ("relu6", lambda x: F.relu6(x), "aten::hardtanh"),
    ("leaky_relu", lambda x: F.leaky_relu(x, 0.2), "aten::leaky_relu"),
    ("softplus", lambda x: F.softplus(x), "aten::softplus"),
    ("softplus_params", lambda x: F.softplus(x, 2.0, 3.0), "aten::softplus"),
    ("hardshrink", lambda x: F.hardshrink(x, _in(x, 0.3)), "aten::hardshrink"),
    ("softshrink", lambda x: F.softshrink(x, _in(x, 0.3)), "aten::softshrink"),
    ("hardsigmoid", lambda x: F.hardsigmoid(x), "aten::hardsigmoid"),
    ("hardswish", lambda x: F.hardswish(x), "aten::hardswish"),
    ("mish", lambda x: F.mish(x), "aten::mish"),
    ("threshold", lambda x: F.threshold(x, _in(x, 0.3), -2.0), "aten::threshold"),
    ("logsigmoid", lambda x: F.logsigmoid(x), "aten::log_sigmoid_forward"),
    (
        "rrelu_eval",
        lambda x: F.rrelu(x, 0.1, 0.3, training=False),
        "aten::rrelu_with_noise",
    ),
]


def _act_input(dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(7)
    edges = torch.tensor(
        [
            0.0,
            -0.0,
            3.0,
            -3.0,
            0.3,
            -0.3,
            20.0,
            25.0,
            -20.0,
            88.0,
            -88.0,
            1e-30,
            float("inf"),
            float("-inf"),
            float("nan"),
        ]
    )
    return torch.cat([edges, torch.randn(200) * 4]).to(dtype)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("name,fn,op", _ACT, ids=[a[0] for a in _ACT])
def test_activation_forward(mojo_gpu, name, fn, op, dtype):
    x = _act_input(dtype)
    with ran(op):
        actual = fn(x.to(mojo_gpu))
    _close(actual, fn(x), **_tol(dtype, 3))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name,fn,op", _ACT, ids=[a[0] for a in _ACT])
def test_activation_autograd(mojo_gpu, name, fn, op, dtype):
    """forward + backward against CPU: the backward kernels (elu_backward,
    hardtanh_backward, ...) run natively."""
    x_cpu = _act_input(dtype)
    x_cpu = x_cpu[torch.isfinite(x_cpu)].clone().requires_grad_()
    x = x_cpu.detach().to(mojo_gpu).requires_grad_()
    g = torch.randn(x_cpu.shape).to(dtype)
    y_cpu = fn(x_cpu)
    y_cpu.backward(g)
    y = fn(x)
    y.backward(g.to(mojo_gpu))
    _close(y.detach(), y_cpu.detach(), **_tol(dtype, 3))
    _close(x.grad, x_cpu.grad, **_tol(dtype, 3))


def test_activation_single_native_ops(mojo_gpu):
    """relu6 -> hardtanh, selu / celu -> elu: one native op each, forward
    and backward."""
    x = torch.randn(33, device=mojo_gpu, requires_grad=True)
    cases = [
        (F.relu6, "aten::hardtanh", "aten::hardtanh_backward"),
        (F.selu, "aten::elu", "aten::elu_backward"),
        (lambda t: F.celu(t, 0.5), "aten::elu", "aten::elu_backward"),
    ]
    for fn, fwd, bwd in cases:
        native.op_counting(True)
        f0, b0 = native.op_count(fwd), native.op_count(bwd)
        y = fn(x)
        assert native.op_count(fwd) == f0 + 1
        y.sum().backward()
        assert native.op_count(bwd) == b0 + 1
        x.grad = None


def test_activation_in_place_and_out(mojo_gpu):
    x_cpu = _act_input(torch.float32)
    for fn in (
        lambda t: F.elu(t, inplace=True),
        lambda t: F.hardtanh(t, inplace=True),
        lambda t: F.leaky_relu(t, 0.1, inplace=True),
        lambda t: F.hardsigmoid(t, inplace=True),
        lambda t: F.hardswish(t, inplace=True),
        lambda t: F.threshold(t, 0.5, 1.0, inplace=True),
        lambda t: F.mish(t, inplace=True),
        lambda t: F.rrelu(t, inplace=True),
    ):
        x, xc = x_cpu.clone().to(mojo_gpu), x_cpu.clone()
        fn(x)
        fn(xc)
        _close(x, xc, **_tol(torch.float32, 3))
    out = torch.empty(0, device=mojo_gpu)
    torch.ops.aten.softplus.out(x_cpu.to(mojo_gpu), 1.0, 20.0, out=out)
    _close(out, F.softplus(x_cpu), **_tol(torch.float32, 3))


def _gelu_grad_input(
    g: torch.Tensor, x: torch.Tensor, approximate: str
) -> torch.Tensor:
    """gelu_backward.grad_input: the pointwise route (the functional
    gelu_backward is the dedicated activation_backward kernel)."""
    out = torch.empty_like(x)
    return torch.ops.aten.gelu_backward.grad_input(
        g, x, approximate=approximate, grad_input=out
    )


_BACKWARD = [
    ("silu_backward", lambda g, x: torch.ops.aten.silu_backward(g, x)),
    ("mish_backward", lambda g, x: torch.ops.aten.mish_backward(g, x)),
    ("hardswish_backward", lambda g, x: torch.ops.aten.hardswish_backward(g, x)),
    ("hardsigmoid_backward", lambda g, x: torch.ops.aten.hardsigmoid_backward(g, x)),
    ("logit_backward", lambda g, x: torch.ops.aten.logit_backward(g, x.sigmoid())),
    (
        "logit_backward_eps",
        lambda g, x: torch.ops.aten.logit_backward(g, x.sigmoid(), 0.2),
    ),
    ("gelu_backward_none", lambda g, x: _gelu_grad_input(g, x, "none")),
    ("gelu_backward_tanh", lambda g, x: _gelu_grad_input(g, x, "tanh")),
    (
        "log_sigmoid_backward",
        lambda g, x: torch.ops.aten.log_sigmoid_backward(
            g, x, torch.ops.aten.log_sigmoid_forward(x)[1]
        ),
    ),
    (
        "elu_backward_result",
        lambda g, x: torch.ops.aten.elu_backward(g, 1.0, 1.0, 1.0, True, F.elu(x)),
    ),
    ("softshrink_backward", lambda g, x: torch.ops.aten.softshrink_backward(g, x, 0.4)),
    ("hardshrink_backward", lambda g, x: torch.ops.aten.hardshrink_backward(g, x, 0.4)),
]


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("name,fn", _BACKWARD, ids=[b[0] for b in _BACKWARD])
def test_backward_ops(mojo_gpu, name, fn, dtype):
    torch.manual_seed(8)
    x = torch.randn(301).to(dtype) * 4
    g = torch.randn(301).to(dtype)
    _close(fn(g.to(mojo_gpu), x.to(mojo_gpu)), fn(g, x), **_tol(dtype, 4))


def test_backward_grad_input_out_forms(mojo_gpu):
    x = torch.randn(40) * 3
    g = torch.randn(40)
    gi = torch.empty(40, device=mojo_gpu)
    torch.ops.aten.silu_backward.grad_input(
        g.to(mojo_gpu), x.to(mojo_gpu), grad_input=gi
    )
    _close(gi, torch.ops.aten.silu_backward(g, x), **_tol(torch.float32, 3))
    torch.ops.aten.gelu_backward.grad_input(
        g.to(mojo_gpu), x.to(mojo_gpu), grad_input=gi
    )
    _close(gi, torch.ops.aten.gelu_backward(g, x), **_tol(torch.float32, 3))
    torch.ops.aten.logit_backward.grad_input(
        g.to(mojo_gpu), x.sigmoid().to(mojo_gpu), grad_input=gi
    )
    _close(gi, torch.ops.aten.logit_backward(g, x.sigmoid()), **_tol(torch.float32, 3))


@pytest.mark.parametrize("dtype", FLOATS)
def test_rrelu_training(mojo_gpu, dtype):
    """Training draws one slope per negative element into `noise`; the
    output is x * noise, the backward grad * noise, and the draws follow the
    device generator (same seed, same slopes)."""
    lower, upper = 0.1, 0.4
    x_cpu = (torch.randn(4097) * 3).to(dtype)
    x_cpu[:3] = torch.tensor([0.0, -0.0, float("nan")])
    torch.manual_seed(11)
    x = x_cpu.to(mojo_gpu).requires_grad_()
    noise = torch.empty_like(x).detach()
    with ran("aten::rrelu_with_noise"):
        y = torch.ops.aten.rrelu_with_noise(x, noise, lower, upper, True)
    n = noise.cpu().float()
    neg = x_cpu.float() <= 0
    assert (n[~neg] == 1).all()
    assert ((n[neg] >= lower - 1e-2) & (n[neg] <= upper + 1e-2)).all()
    assert n[neg].std() > 0.05  # really random
    _close(y.detach(), x_cpu * n.to(dtype), **_tol(dtype, 2))
    y.backward(torch.ones_like(y))
    _close(x.grad, n.to(dtype), **_tol(dtype, 2))
    torch.manual_seed(11)
    noise2 = torch.empty_like(noise)
    torch.ops.aten.rrelu_with_noise(x.detach(), noise2, lower, upper, True)
    _close(noise2, n.to(dtype), rtol=0.0, atol=0.0)
    z = x.detach().clone()
    F.rrelu(z, lower, upper, training=True, inplace=True)
    assert z.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.uint8])
def test_integer_pow(mojo_gpu, dtype):
    torch.manual_seed(9)
    lo = 0 if dtype == torch.uint8 else -5
    base = torch.randint(lo, 6, (300,), dtype=dtype)
    expo = torch.randint(0, 9, (300,), dtype=dtype)
    with ran("aten::pow.Tensor_Tensor"):
        _close(torch.pow(base.to(mojo_gpu), expo.to(mojo_gpu)), torch.pow(base, expo))
    _close(torch.pow(base.to(mojo_gpu), 3), torch.pow(base, 3))
    _close(torch.pow(2, expo.to(mojo_gpu)), torch.pow(2, expo))
    if dtype != torch.uint8:
        neg = torch.tensor([-1, -2, -3, -1, 0], dtype=dtype)
        b = torch.tensor([1, -1, -1, 2, 5], dtype=dtype)
        _close(torch.pow(b.to(mojo_gpu), neg.to(mojo_gpu)), torch.pow(b, neg))
        with pytest.raises(RuntimeError, match="negative integer powers"):
            torch.pow(b.to(mojo_gpu), -2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lerp_scalar_half(mojo_gpu, dtype):
    torch.manual_seed(10)
    s = torch.randn(33, 5).to(dtype)
    e = torch.randn(33, 5).to(dtype)
    for w in (0.3, 0.7, 1.4):
        with ran("aten::lerp.Scalar"):
            actual = torch.lerp(s.to(mojo_gpu), e.to(mojo_gpu), w)
        _close(actual, torch.lerp(s, e, w), **_tol(dtype, 2))
    x, xc = s.to(mojo_gpu), s.clone()
    x.lerp_(e.to(mojo_gpu), 0.25)
    xc.lerp_(e, 0.25)
    _close(x, xc, **_tol(dtype, 2))


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
def test_hardtanh_threshold_integers(mojo_gpu, dtype):
    x = torch.arange(-8, 9, dtype=dtype)
    _close(F.hardtanh(x.to(mojo_gpu), -3, 5), F.hardtanh(x, -3, 5))
    _close(F.relu6(x.to(mojo_gpu)), F.relu6(x))
    _close(F.threshold(x.to(mojo_gpu), 2, 7), F.threshold(x, 2, 7))


def test_clamp_tensor_mixed_dtypes(mojo_gpu):
    x = torch.randn(5, 4)
    lo = torch.randint(-1, 1, (5, 4), dtype=torch.int32)
    hi = torch.randint(0, 2, (4,), dtype=torch.int64)
    _close(torch.clamp(x.to(mojo_gpu), lo.to(mojo_gpu)), torch.clamp(x, lo))
    _close(torch.clamp(x.to(mojo_gpu), None, hi.to(mojo_gpu)), torch.clamp(x, None, hi))
    _close(
        torch.clamp(x.to(mojo_gpu), lo.to(mojo_gpu), hi.to(mojo_gpu)),
        torch.clamp(x, lo, hi),
    )
    xb = x.bfloat16()
    _close(torch.clamp(xb.to(mojo_gpu), lo.to(mojo_gpu)), torch.clamp(xb, lo))


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize(
    "name,fn", (("igamma", torch.igamma), ("igammac", torch.igammac))
)
def test_igamma(mojo_gpu, name, fn, dtype):
    """Every regime of calc_igamma / calc_igammac: the boundaries (a or x at
    0, inf, NaN, negative), the series and the continued fraction on either
    side of x = a, and the uniform asymptotic expansion for large a ~ x."""
    grid = [0.0, 1e-3, 0.3, 0.5, 0.75, 1.0, 1.05, 1.2, 2.5, 7.0, 19.0]
    grid += [21.0, 24.0, 30.0, 150.0, 199.0, 210.0, 1000.0, 1040.0]
    edges = [float("inf"), float("nan"), -1.0]
    values = torch.tensor(grid + edges)
    a = values[:, None].expand(-1, len(values)).to(dtype)
    x = values[None, :].expand(len(values), -1).to(dtype)
    expected = fn(a, x)
    with ran(f"aten::{name}"):
        actual = fn(a.to(mojo_gpu), x.to(mojo_gpu))
    _close(actual, expected, **_tol(dtype, 8))
    # Broadcast against a 0-d tensor, a strided out= and the in-place method.
    x_row = x[3].contiguous()
    scalar = torch.tensor(2.5, dtype=dtype)
    _close(
        fn(x_row.to(mojo_gpu), scalar.to(mojo_gpu)), fn(x_row, scalar), **_tol(dtype, 8)
    )
    out = torch.zeros(len(values), 2 * len(values), dtype=dtype).to(mojo_gpu)
    fn(a.to(mojo_gpu), x.to(mojo_gpu), out=out[:, ::2])
    _close(out[:, ::2], expected, **_tol(dtype, 8))
    inplace = a.contiguous().to(mojo_gpu)
    getattr(inplace, name + "_")(x.to(mojo_gpu))
    _close(inplace, expected, **_tol(dtype, 8))
