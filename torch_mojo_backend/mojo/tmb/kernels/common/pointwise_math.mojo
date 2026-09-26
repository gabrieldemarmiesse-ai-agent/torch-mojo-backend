"""Pointwise SIMD expressions shared by eager (tmb/kernels/pointwise) and
torch.compile (tmb/graph/pointwise.mojo) execution: the binary and ternary
math ops (atan2, hypot, copysign, nextafter, gcd, the special polynomials,
...) and the activations with scalar parameters (elu, softplus, hardtanh,
...) with their backwards.

One entry, `pointwise[kind]`, takes up to three operands of one dtype plus
four scalar parameters and returns the result in `out_dtype`. Each kind
follows the CUDA kernel stock PyTorch runs (named next to it); half-precision
operands are widened to float32 (ATen's opmath type), computed, and rounded
once, except where ATen works on the half value itself (copysign, nextafter,
heaviside, fmax/fmin, clamp).

The exponential and logarithm are the bit-exact libdevice ports
(`nv_expf` / `nv_logf` / `nv_log1pf`, `nv_exp` / `nv_log` / `nv_log1p` for
float64): std.math's GPU `exp` is `ex2.approx` of `x * log2(e)`, whose
relative error grows with |x| (5e-6 near 88) where CUDA's `expf` stays within
2 ulp. They run once per lane.

The GPU build compiles with fast-math flags that let LLVM assume no NaN, so a
comparison with NaN may fold either way: every kind whose result depends on
NaN handling tests `isnan` (bit-based `llvm.is.fpclass`, which survives the
flags) and selects explicitly.
"""

from std.math import acos, cos, erf, floor, pow, sin
from std.collections import Array
from std.memory import bitcast
from std.sys.info import size_of
from std.utils.numerics import inf, isinf, isnan, nan

from tmb.kernels.common.libdevice_port import (
    nv_exp,
    nv_expf,
    nv_log,
    nv_log1p,
    nv_log1pf,
    nv_logf,
)
from tmb.kernels.common.math_utils import ieee_sqrt
from tmb.kernels.common.op_utils import _fmod_narrow_float_exact_scalar
from tmb.kernels.common.special_math import igamma_f, igammac_f


@always_inline
def wide_dtype[dtype: DType]() -> DType:
    """ATen's opmath type of a floating dtype: float64 stays, the rest
    compute in float32 (integers too, for the kinds that promote them)."""
    comptime if dtype == DType.float64:
        return DType.float64
    else:
        return DType.float32


@always_inline
def param_dtype[dtype: DType]() -> DType:
    """The dtype the scalar parameters travel in to the device: float64 only
    for float64 operands (Metal has no double at all)."""
    return wide_dtype[dtype]()


# ---------------------------------------------------------------------------
# accurate exp / log family (per lane, bit-exact libdevice ports)
# ---------------------------------------------------------------------------


@always_inline
def _exp[w: DType, n: Int](x: SIMD[w, n]) -> SIMD[w, n]:
    var r = SIMD[w, n]()
    comptime for i in range(n):
        comptime if w == DType.float64:
            r[i] = nv_exp(x[i].cast[DType.float64]()).cast[w]()
        else:
            r[i] = nv_expf(x[i].cast[DType.float32]()).cast[w]()
    return r


@always_inline
def _log[w: DType, n: Int](x: SIMD[w, n]) -> SIMD[w, n]:
    var r = SIMD[w, n]()
    comptime for i in range(n):
        comptime if w == DType.float64:
            r[i] = nv_log(x[i].cast[DType.float64]()).cast[w]()
        else:
            r[i] = nv_logf(x[i].cast[DType.float32]()).cast[w]()
    return r


@always_inline
def _log1p[w: DType, n: Int](x: SIMD[w, n]) -> SIMD[w, n]:
    var r = SIMD[w, n]()
    comptime for i in range(n):
        comptime if w == DType.float64:
            r[i] = nv_log1p(x[i].cast[DType.float64]()).cast[w]()
        else:
            r[i] = nv_log1pf(x[i].cast[DType.float32]()).cast[w]()
    return r


@always_inline
def _expm1[w: DType, n: Int](x: SIMD[w, n]) -> SIMD[w, n]:
    """expm1 by Kahan's correction, (e^x - 1) * x / log(e^x): a few ulp
    everywhere, exact 0 at 0, -1 below the float range, NaN/inf through."""
    var u = _exp(x)
    var um1 = u - 1
    var corrected = um1 * (x / _log(u))
    var r = um1.eq(0).select(x, corrected)
    r = um1.eq(-1).select(um1, r)
    r = isinf(u).select(u, r)
    return isnan(x).select(x, r)


@always_inline
def _tanh[w: DType, n: Int](x: SIMD[w, n]) -> SIMD[w, n]:
    """tanh(|x|) = e / (e + 2), e = expm1(2|x|); sign restored, saturated to
    1 past 2|x| = 40 where e overflows float32's ratio."""
    var ax = abs(x)
    var e = _expm1(ax + ax)
    var t = e / (e + 2)
    t = ax.gt(20).select(SIMD[w, n](1), t)
    var r = x.lt(0).select(-t, t)
    return isnan(x).select(x, r)


@always_inline
def _sigmoid[w: DType, n: Int](x: SIMD[w, n]) -> SIMD[w, n]:
    return 1 / (1 + _exp(-x))


@always_inline
def _atan_unit[w: DType, n: Int](t: SIMD[w, n]) -> SIMD[w, n]:
    """atan(t) for t in [0, 1]: t * p(t^2), p a degree-9 fit of atan(t)/t
    (relative error 3.1e-9, below float32's half ulp)."""
    var s = t * t
    var p = SIMD[w, n](-0.001744857574148222)
    p = p.fma(s, 0.010694263082360147)
    p = p.fma(s, -0.030762979808513034)
    p = p.fma(s, 0.05753647151596324)
    p = p.fma(s, -0.08378559850299827)
    p = p.fma(s, 0.109435660872342)
    p = p.fma(s, -0.14262576462909438)
    p = p.fma(s, 0.19998390513909448)
    p = p.fma(s, -0.3333329383818141)
    return (p * s).fma(t, t)


@always_inline
def _atan2[w: DType, n: Int](y: SIMD[w, n], x: SIMD[w, n]) -> SIMD[w, n]:
    """atan2 with C99's special values (signed zeros, infinities)."""
    comptime pi = 3.14159265358979323846
    comptime half_pi = 1.57079632679489661923
    var ax = abs(x)
    var ay = abs(y)
    var both_inf = isinf(ax) & isinf(ay)
    var mx = max(ax, ay)
    var mn = min(ax, ay)
    var t = mn / mx
    t = mx.eq(0).select(SIMD[w, n](0), t)
    t = both_inf.select(SIMD[w, n](1), t)
    var r = _atan_unit(t)
    r = ay.gt(ax).select(half_pi - r, r)
    var x_neg = bitcast[_bits[w](), n](x).lt(0)  # -0 counts as negative
    r = x_neg.select(pi - r, r)
    var y_neg = bitcast[_bits[w](), n](y).lt(0)
    r = y_neg.select(-r, r)
    return (isnan(x) | isnan(y)).select(SIMD[w, n](nan[w]()), r)


@always_inline
def _bits[dtype: DType]() -> DType:
    """The signed integer dtype of the same width (sign bit = sign test)."""
    comptime if dtype == DType.float64:
        return DType.int64
    elif dtype == DType.float32:
        return DType.int32
    else:
        return DType.int16


@always_inline
def _pow_f64[
    n: Int
](a: SIMD[DType.float64, n], b: SIMD[DType.float64, n]) -> SIMD[
    DType.float64, n
]:
    """a ** b for a > 0 (the only use: zeta's terms)."""
    return _exp(b * _log(a))


# ---------------------------------------------------------------------------
# special functions
# ---------------------------------------------------------------------------


@always_inline
def _zeta_scalar(x: Float64, q: Float64) -> Float64:
    """Hurwitz zeta, Cephes (aten/src/ATen/native/Math.h `zeta`), in double
    for every input dtype: CPU torch accumulates float in double too."""
    comptime MACHEP = 1.11022302462515654042e-16
    var A = Array[Float64, 12](fill=0.0)
    A[0] = 12.0
    A[1] = -720.0
    A[2] = 30240.0
    A[3] = -1209600.0
    A[4] = 47900160.0
    A[5] = -1.8924375803183791606e9
    A[6] = 7.47242496e10
    A[7] = -2.950130727918164224e12
    A[8] = 1.1646782814350067249e14
    A[9] = -4.5979787224074726105e15
    A[10] = 1.8152105401943546773e17
    A[11] = -7.1661652561756670113e18
    if x == 1.0:
        return inf[DType.float64]()
    if x < 1.0:
        return nan[DType.float64]()
    if q <= 0.0:
        if q == floor(q):
            return inf[DType.float64]()
        if x != floor(x):
            return nan[DType.float64]()
    var s = _zeta_pow(q, -x)
    var a = q
    var i = 0
    var b = 0.0
    while i < 9 or a <= 9.0:
        i += 1
        a += 1.0
        b = _zeta_pow(a, -x)
        s += b
        if -MACHEP * s < b and b < MACHEP * s:
            return s
    var w = a
    s += b * w / (x - 1.0)
    s -= 0.5 * b
    a = 1.0
    var k = 0.0
    for j in range(12):
        a *= x + k
        b /= w
        var t = a * b / A[j]
        s = s + t
        t = abs(t / s)
        if t < MACHEP:
            return s
        k += 1.0
        a *= x + k
        b /= w
        k += 1.0
    return s


@always_inline
def _zeta_pow(a: Float64, b: Float64) -> Float64:
    """C's pow for zeta's bases: negative (non-integral q) bases only meet an
    integral exponent there."""
    if a > 0.0:
        return nv_exp(b * nv_log(a))
    if a == 0.0:
        return inf[DType.float64]() if b < 0.0 else 0.0
    var r = nv_exp(b * nv_log(-a))
    # b is integral here (checked by the caller's x == floor(x)).
    var odd = (b - 2.0 * floor(b * 0.5)) != 0.0
    return -r if odd else r


@always_inline
def _poly_n(n: Float64) -> Int:
    """`static_cast<int64_t>(n)` for the polynomial degree (truncation);
    NaN and out-of-range degrees read as -1, which every polynomial maps
    to 0 (hermite does exactly this; the others would be UB in C++)."""
    if n != n or n >= 9.2e18 or n <= -9.2e18:
        return -1
    return Int(n)


@always_inline
def _polynomial[
    kind: StaticString, w: DType
](x: Scalar[w], n_f: Scalar[w]) -> Scalar[w] where w.is_floating_point():
    """The orthogonal polynomials of aten/src/ATen/native/Math.h
    (`*_polynomial_*_forward`), statement for statement, in the opmath type."""
    var n = _poly_n(n_f.cast[DType.float64]())
    comptime one = Scalar[w](1.0)
    if n < 0:
        return 0
    if isnan(x):
        # Every special case compares x and fails for NaN, so the result is
        # the degree-0 constant or NaN; tested up front because the GPU's
        # fast-math flags may fold those comparisons either way.
        return one if n == 0 else x
    comptime if kind == "chebyshev_polynomial_t":
        if abs(x) == one:
            if x > 0 or n % 2 == 0:
                return one
            return -one
        if n > 6 and abs(x) < one:
            return cos(Scalar[w](n) * acos(x))
        if n == 0:
            return one
        if n == 1:
            return x
        var p = one
        var q = x
        var r = x
        var k = 2
        while k <= n and not isnan(q):
            r = (x + x) * q - p
            p = q
            q = r
            k += 1
        return r
    elif kind == "chebyshev_polynomial_u":
        if abs(x) == one:
            if x > 0 or n % 2 == 0:
                return Scalar[w](n + 1)
            return -Scalar[w](n + 1)
        if n > 8 and abs(x) < one:
            var t = acos(x)
            if sin(t) != 0:
                return sin(Scalar[w](n + 1) * t) / sin(t)
            return Scalar[w](n + 1) * cos(Scalar[w](n + 1) * t) / x
        if n == 0:
            return one
        if n == 1:
            return x + x
        var p = one
        var q = x + x
        var r = q
        var k = 2
        while k <= n and not isnan(q):
            r = (x + x) * q - p
            p = q
            q = r
            k += 1
        return r
    elif kind == "chebyshev_polynomial_v":
        if abs(x) == one:
            if x > 0:
                return one
            if n % 2 == 0:
                return Scalar[w](n + n + 1)
            return -Scalar[w](n + n + 1)
        if n > 8 and abs(x) < one:
            var t = acos(x)
            if sin(t / 2) != one:
                return cos((Scalar[w](n) + 0.5) * t) / cos(t / 2)
            if n % 2 == 0:
                return Scalar[w](n + n + 1)
            return -Scalar[w](n + n + 1)
        if n == 0:
            return one
        if n == 1:
            return x + x - one
        var p = one
        var q = x + x - one
        var r = q
        var k = 2
        while k <= n and not isnan(q):
            r = (x + x) * q - p
            p = q
            q = r
            k += 1
        return r
    elif kind == "chebyshev_polynomial_w":
        if abs(x) == one:
            if x > 0:
                return Scalar[w](n + n + 1)
            if n % 2 == 0:
                return one
            return -one
        if n > 8 and abs(x) < one:
            var t = acos(x)
            if cos(t / 2) != one:
                return sin((Scalar[w](n) + 0.5) * t) / sin(t / 2)
            if x > 0:
                return Scalar[w](n + n + 1)
            if n % 2 == 0:
                return one
            return -one
        if n == 0:
            return one
        if n == 1:
            return x + x + one
        var p = one
        var q = x + x + one
        var r = q
        var k = 2
        while k <= n and not isnan(q):
            r = (x + x) * q - p
            p = q
            q = r
            k += 1
        return r
    elif kind == "hermite_polynomial_h":
        if n == 0:
            return one
        if n == 1:
            return x + x
        comptime limit = 512 if w == DType.float64 else 128
        if n > limit:
            return nan[w]()
        var p = one
        var q = x + x
        var r = Scalar[w](0)
        var k = 2
        while k < n + n:
            r = (x + x) * q - Scalar[w](k) * p
            p = q
            q = r
            k += 2
        return r
    elif kind == "hermite_polynomial_he":
        if n == 0:
            return one
        if n == 1:
            return x
        comptime limit = 512 if w == DType.float64 else 128
        if n > limit:
            return nan[w]()
        var p = one
        var q = x
        var r = Scalar[w](0)
        for k in range(1, n):
            r = x * q - Scalar[w](k) * p
            p = q
            q = r
        return r
    elif kind == "laguerre_polynomial_l":
        if abs(x) == 0:
            return one
        if n == 0:
            return one
        if n == 1:
            return one - x
        var p = one
        var q = one - x
        var r = q
        var k = 1
        while k < n and not isnan(q):
            r = (
                (Scalar[w](k + k) + (one - x)) * q - Scalar[w](k) * p
            ) / Scalar[w](k + 1)
            p = q
            q = r
            k += 1
        return r
    elif kind == "legendre_polynomial_p":
        if abs(x) == one:
            if x > 0 or n % 2 == 0:
                return one
            return -one
        if n == 0:
            return one
        if n == 1:
            return x
        var p = one
        var q = x
        var r = q
        var k = 1
        while k < n and not isnan(q):
            r = (Scalar[w](k + k + 1) * x * q - Scalar[w](k) * p) / Scalar[w](
                k + 1
            )
            p = q
            q = r
            k += 1
        return r
    else:
        # The shifted Chebyshev polynomials: T*(x) = T(2x - 1) and friends,
        # with their own boundary cases at x = 0 and x = 1.
        var y = x + x - one
        comptime if kind == "shifted_chebyshev_polynomial_t":
            if x == one:
                return one
            if x == 0:
                return one if n % 2 == 0 else -one
            if n > 6 and abs(y) < one:
                return cos(Scalar[w](n) * acos(y))
            if n == 0:
                return one
            if n == 1:
                return y
            var p = one
            var q = y
            var r = q
            var k = 2
            while k <= n and not isnan(q):
                r = (y + y) * q - p
                p = q
                q = r
                k += 1
            return r
        elif kind == "shifted_chebyshev_polynomial_u":
            if x == one:
                return Scalar[w](n + 1)
            if x == 0:
                return Scalar[w](n + 1) if n % 2 == 0 else -Scalar[w](n + 1)
            if n > 6 and abs(y) < one:
                var t = acos(y)
                if sin(t) != 0:
                    return sin(Scalar[w](n + 1) * t) / sin(t)
                return Scalar[w](n + 1) * cos(Scalar[w](n + 1) * t) / y
            if n == 0:
                return one
            if n == 1:
                return y + y
            var p = one
            var q = y + y
            var r = q
            var k = 2
            while k <= n and not isnan(q):
                r = (y + y) * q - p
                p = q
                q = r
                k += 1
            return r
        elif kind == "shifted_chebyshev_polynomial_v":
            if x == one:
                return one
            if x == 0:
                return Scalar[w](n + n + 1) if n % 2 == 0 else -Scalar[w](
                    n + n + 1
                )
            if n > 6 and abs(y) < one:
                var t = acos(y)
                if sin(t / 2) != one:
                    return cos((Scalar[w](n) + 0.5) * t) / cos(t / 2)
                return Scalar[w](n + n + 1) if n % 2 == 0 else -Scalar[w](
                    n + n + 1
                )
            if n == 0:
                return one
            if n == 1:
                return y + y - one
            var p = one
            var q = y + y - one
            var r = q
            var k = 2
            while k <= n and not isnan(q):
                r = (y + y) * q - p
                p = q
                q = r
                k += 1
            return r
        else:
            comptime assert (
                kind == "shifted_chebyshev_polynomial_w"
            ), "unknown polynomial kind"
            if x == one:
                return Scalar[w](n + n + 1)
            if x == 0:
                return one if n % 2 == 0 else -one
            if n > 4 and abs(y) < one:
                var t = acos(y)
                if cos(t / 2) != one:
                    return sin((Scalar[w](n) + 0.5) * t) / sin(t / 2)
                return one if n % 2 == 0 else -one
            if n == 0:
                return one
            if n == 1:
                return y + y + one
            var p = one
            var q = y + y + one
            var r = q
            var k = 2
            while k <= n and not isnan(q):
                r = (y + y) * q - p
                p = q
                q = r
                k += 1
            return r


@always_inline
def is_polynomial[kind: StaticString]() -> Bool:
    return (
        kind == "chebyshev_polynomial_t"
        or kind == "chebyshev_polynomial_u"
        or kind == "chebyshev_polynomial_v"
        or kind == "chebyshev_polynomial_w"
        or kind == "shifted_chebyshev_polynomial_t"
        or kind == "shifted_chebyshev_polynomial_u"
        or kind == "shifted_chebyshev_polynomial_v"
        or kind == "shifted_chebyshev_polynomial_w"
        or kind == "hermite_polynomial_h"
        or kind == "hermite_polynomial_he"
        or kind == "laguerre_polynomial_l"
        or kind == "legendre_polynomial_p"
    )


# ---------------------------------------------------------------------------
# kinds computed on the operand dtype itself
# ---------------------------------------------------------------------------


@always_inline
def is_native_kind[kind: StaticString]() -> Bool:
    """Kinds ATen evaluates on the storage value, never widened: they take
    every dtype their op accepts (integers, bool) and keep it."""
    return (
        kind == "copysign"
        or kind == "nextafter"
        or kind == "heaviside"
        or kind == "fmax"
        or kind == "fmin"
        or kind == "fmod"
        or kind == "gcd"
        or kind == "lcm"
        or kind == "lshift"
        or kind == "rshift"
        or kind == "clamp"
        or kind == "maximum"
        or kind == "minimum"
        or kind == "ipow"
        or kind == "frexp_mantissa"
        or kind == "frexp_exponent"
    )


@always_inline
def _uint_of[dtype: DType]() -> DType:
    comptime if size_of[dtype]() == 8:
        return DType.uint64
    elif size_of[dtype]() == 4:
        return DType.uint32
    elif size_of[dtype]() == 2:
        return DType.uint16
    else:
        return DType.uint8


@always_inline
def _mant_bits[dtype: DType]() -> Int:
    comptime if dtype == DType.float64:
        return 52
    elif dtype == DType.float32:
        return 23
    elif dtype == DType.float16:
        return 10
    else:
        return 7


@always_inline
def _nextafter[
    dtype: DType, n: Int
](a: SIMD[dtype, n], b: SIMD[dtype, n]) -> SIMD[dtype, n]:
    """C's nextafter on the bits of `dtype` (c10's Half/BFloat16 overloads
    do the same on their 16-bit patterns)."""
    comptime u = _uint_of[dtype]()
    comptime sign = Scalar[u](1) << Scalar[u](size_of[dtype]() * 8 - 1)
    var ua = bitcast[u, n](a)
    var ub = bitcast[u, n](b)
    var mag_a = ua & ~sign
    var mag_b = ub & ~sign
    var both_zero = mag_a.eq(0) & mag_b.eq(0)
    # From +-0 toward nonzero b: the smallest subnormal with b's sign.
    var from_zero = (ub & sign) | 1
    # Away from zero when a < b for positive a (or a > b for negative a):
    # one step up in magnitude, else one step down.
    var a_neg = (ua & sign).ne(0)
    var up = a.lt(b) ^ a_neg
    var stepped = up.select(ua + 1, ua - 1)
    var r = mag_a.eq(0).select(from_zero, stepped)
    r = a.eq(b).select(ub, r)
    r = both_zero.select(ub, r)
    var res = bitcast[dtype, n](r)
    return (isnan(a) | isnan(b)).select(SIMD[dtype, n](nan[dtype]()), res)


@always_inline
def _frexp_exponent[
    dtype: DType, n: Int
](a: SIMD[dtype, n]) -> SIMD[DType.int32, n]:
    """frexp's exponent: a = m * 2^e with 0.5 <= |m| < 1; 0 for zero, inf
    and NaN (C's frexp leaves those unspecified-but-0 in glibc and CUDA)."""
    comptime w = wide_dtype[dtype]()
    comptime u = _uint_of[w]()
    comptime mb = _mant_bits[w]()
    comptime ebias = 1023 if w == DType.float64 else 127
    comptime emask = Scalar[u](0x7FF if w == DType.float64 else 0xFF)
    var x = a.cast[w]()
    var ux = bitcast[u, n](x)
    var e = ((ux >> Scalar[u](mb)) & emask).cast[DType.int32]()
    # Subnormals (only float32 widened from nothing reaches here: half and
    # bfloat16 subnormals are normal in float32): scale up by 2^64 first.
    var sub = e.eq(0) & x.ne(0)
    var xs = x * Scalar[w](18446744073709551616.0)
    var es = ((bitcast[u, n](xs) >> Scalar[u](mb)) & emask).cast[
        DType.int32
    ]() - 64
    e = sub.select(es, e)
    var res = e - Int32(ebias - 1)
    var special = x.eq(0) | isnan(x) | isinf(x)
    return special.select(SIMD[DType.int32, n](0), res)


@always_inline
def _frexp_mantissa[dtype: DType, n: Int](a: SIMD[dtype, n]) -> SIMD[dtype, n]:
    comptime w = wide_dtype[dtype]()
    var e = _frexp_exponent(a)
    var x = a.cast[w]()
    # 2^-e in two halves so that neither factor overflows (e reaches 1024).
    var h1 = (-e) >> 1
    var h2 = (-e) - h1
    var m = x * _exp2i[w](h1) * _exp2i[w](h2)
    var special = x.eq(0) | isnan(x) | isinf(x)
    return special.select(x, m).cast[dtype]()


@always_inline
def _exp2i[w: DType, n: Int](k: SIMD[DType.int32, n]) -> SIMD[w, n]:
    """2^k for |k| within the normal exponent range, by building the bits."""
    comptime u = _uint_of[w]()
    comptime mb = _mant_bits[w]()
    comptime ebias = 1023 if w == DType.float64 else 127
    return bitcast[w, n]((k + Int32(ebias)).cast[u]() << Scalar[u](mb))


@always_inline
def _gcd_scalar[
    dtype: DType
](x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[dtype]:
    """calc_gcd (aten/src/ATen/native/Math.h): Euclid on the magnitudes."""
    var a = abs(x)
    var b = abs(y)
    while a != 0:
        var c = a
        a = b % a
        b = c
    return b


@always_inline
def _powi[
    dtype: DType
](base: Scalar[dtype], exponent: Scalar[dtype]) -> Scalar[dtype]:
    var a = base
    var b = exponent
    comptime if dtype.is_signed():
        if b < 0:
            if a == 1:
                return 1
            if a == -1:
                return -1 if (b % 2) != 0 else 1
            return 0
    var result = Scalar[dtype](1)
    while b != 0:
        if (b & 1) != 0:
            result *= a
        b = b >> 1
        a *= a
    return result


@always_inline
def _native[
    kind: StaticString, dtype: DType, n: Int
](a: SIMD[dtype, n], b: SIMD[dtype, n], c: SIMD[dtype, n]) -> SIMD[dtype, n]:
    comptime if kind == "copysign":
        # CopysignKernel.cu: the sign bit of b on the magnitude bits of a.
        comptime u = _uint_of[dtype]()
        comptime sign = Scalar[u](1) << Scalar[u](size_of[dtype]() * 8 - 1)
        return bitcast[dtype, n](
            (bitcast[u, n](a) & ~sign) | (bitcast[u, n](b) & sign)
        )
    elif kind == "nextafter":
        return _nextafter(a, b)
    elif kind == "heaviside":
        # StepKernel.cu: a == 0 ? b : (a > 0).
        comptime if dtype == DType.bool:
            return a | b
        else:
            var r = a.eq(0).select(b, a.gt(0).cast[dtype]())
            comptime if dtype.is_floating_point():
                return isnan(a).select(SIMD[dtype, n](0), r)
            else:
                return r
    elif kind == "fmax" or kind == "fmin":
        # MaxMinElementwiseKernel.cu: C's fmax/fmin for floats (a NaN
        # operand yields the other one), maximum/minimum otherwise.
        comptime if dtype == DType.bool:
            comptime if kind == "fmax":
                return a | b
            else:
                return a & b
        else:
            var r = max(a, b) if kind == "fmax" else min(a, b)
            comptime if dtype.is_floating_point():
                r = isnan(a).select(b, r)
                r = isnan(b).select(a, r)
            return r
    elif kind == "fmod":
        # BinaryRemainderKernel.cu: C's fmod (the dividend's sign), exact.
        comptime if dtype.is_floating_point():
            comptime if dtype == DType.float64:
                # The truncated quotient; exact while |a / b| < 2^53.
                var q = a / b
                var tq = q.lt(0).select(-floor(-q), floor(q))
                var r = a - tq * b
                var bad = isnan(a) | isnan(b) | isinf(a) | b.eq(0)
                r = bad.select(SIMD[dtype, n](nan[dtype]()), r)
                return isinf(b).select(a, r)
            else:
                var r = SIMD[dtype, n]()
                comptime for i in range(n):
                    r[i] = _fmod_narrow_float_exact_scalar(
                        a[i].cast[DType.float32](), b[i].cast[DType.float32]()
                    ).cast[dtype]()
                return r
        else:
            # C's %, truncating (Mojo's `%` floors: move a remainder whose
            # sign differs from the dividend's back by one divisor); 0 for
            # a zero divisor (CPU raises, CUDA leaves it undefined).
            var safe = b.eq(0).select(SIMD[dtype, n](1), b)
            var r = a % safe
            comptime if dtype.is_signed():
                var fix = r.ne(0) & (r.lt(0) ^ a.lt(0))
                r = fix.select(r - safe, r)
            return b.eq(0).select(SIMD[dtype, n](0), r)
    elif kind == "gcd" or kind == "lcm":
        var r = SIMD[dtype, n]()
        comptime for i in range(n):
            var g = _gcd_scalar(a[i], b[i])
            comptime if kind == "gcd":
                r[i] = g
            else:
                # GcdLcmKernel.cu: (g == 0) ? 0 : abs(a / g * b), with C++'s
                # promotion of the narrow integers to int before the abs.
                comptime wide = DType.int64 if size_of[
                    dtype
                ]() == 8 else DType.int32
                var prod = (a[i] // g).cast[wide]() * b[i].cast[wide]()
                r[i] = 0 if g == 0 else abs(prod).cast[dtype]()
        return r
    elif kind == "lshift" or kind == "rshift":
        # BinaryShiftOpsKernels.cu: a negative or too-wide shift gives 0
        # (left) or the sign fill (right) instead of C's UB.
        comptime width_bits = size_of[dtype]() * 8
        comptime u = _uint_of[dtype]()
        comptime if kind == "lshift":
            var bad = bitcast[u, n](b).ge(Scalar[u](width_bits))
            comptime if dtype.is_signed():
                bad = bad | b.lt(0)
            var sh = bad.select(SIMD[u, n](0), bitcast[u, n](b))
            var r = bitcast[dtype, n](bitcast[u, n](a) << sh)
            return bad.select(SIMD[dtype, n](0), r)
        else:
            comptime max_shift = width_bits - (1 if dtype.is_signed() else 0)
            var bad = bitcast[u, n](b).ge(Scalar[u](max_shift))
            comptime if dtype.is_signed():
                bad = bad | b.lt(0)
            var sh = bad.select(SIMD[dtype, n](max_shift), b)
            return a >> sh
    elif kind == "clamp":
        # ClampKernel (clamp_kernel_cuda): NaN value, then NaN bounds win,
        # else min(max(v, lo), hi).
        var r = min(max(a, b), c)
        comptime if dtype.is_floating_point():
            r = isnan(c).select(c, r)
            r = isnan(b).select(b, r)
            r = isnan(a).select(a, r)
        return r
    elif kind == "maximum" or kind == "minimum":
        # MaxMinElementwiseKernel.cu: or/and for bool, NaN propagating.
        comptime if dtype == DType.bool:
            return (a | b) if kind == "maximum" else (a & b)
        else:
            var r = max(a, b) if kind == "maximum" else min(a, b)
            comptime if dtype.is_floating_point():
                r = isnan(b).select(b, r)
                r = isnan(a).select(a, r)
            return r
    elif kind == "ipow":
        # Pow.h powi: square-and-multiply; a negative exponent gives 1 for
        # base 1, +-1 for base -1 by parity, 0 otherwise.
        comptime if dtype.is_integral():
            var r = SIMD[dtype, n]()
            comptime for i in range(n):
                r[i] = _powi(a[i], b[i])
            return r
        else:
            comptime assert False, "ipow takes integers"
    elif kind == "frexp_mantissa":
        return _frexp_mantissa(a)
    else:
        comptime assert False, "unknown native pointwise kind"


# ---------------------------------------------------------------------------
# kinds computed in the opmath type
# ---------------------------------------------------------------------------


@always_inline
def _wide[
    kind: StaticString, w: DType, n: Int
](a: SIMD[w, n], b: SIMD[w, n], c: SIMD[w, n], p: SIMD[w, 4]) -> SIMD[
    w, n
] where w.is_floating_point():
    comptime zero = SIMD[w, n](0)
    comptime one = SIMD[w, n](1)
    # --- binary math -------------------------------------------------------
    comptime if kind == "atan2":
        return _atan2(a, b)
    elif kind == "hypot":
        # BinaryGeometricKernels.cu (::hypot). Float32 squares in float64,
        # which is exact, and rounds the root once; float64 scales by the
        # larger magnitude.
        comptime if w == DType.float32:
            var ad = a.cast[DType.float64]()
            var bd = b.cast[DType.float64]()
            var r = ieee_sqrt(ad * ad + bd * bd).cast[w]()
            return (isinf(a) | isinf(b)).select(SIMD[w, n](inf[w]()), r)
        else:
            var ax = abs(a)
            var bx = abs(b)
            var mx = max(ax, bx)
            var mn = min(ax, bx)
            var t = mn / mx
            var r = mx * ieee_sqrt(t.fma(t, 1))
            r = mx.eq(0).select(zero, r)
            r = (isnan(a) | isnan(b)).select(SIMD[w, n](nan[w]()), r)
            return (isinf(a) | isinf(b)).select(SIMD[w, n](inf[w]()), r)
    elif kind == "logaddexp" or kind == "logaddexp2":
        # LogAddExpKernel.cu: inf == inf keeps it, else
        # m + log1p(exp(-|a - b|)) (exp2 and a 1/ln2 factor for base 2).
        var m = max(a, b)
        var d = -abs(a - b)
        var r: SIMD[w, n]
        comptime if kind == "logaddexp":
            r = m + _log1p(_exp(d))
        else:
            comptime ln2 = 0.693147180559945309417232121458
            comptime inv_ln2 = 1.44269504088896340735992468100
            r = m + _log1p(_exp(d * ln2)) * inv_ln2
        r = (isinf(a) & a.eq(b)).select(a, r)
        return (isnan(a) | isnan(b)).select(SIMD[w, n](nan[w]()), r)
    elif kind == "xlogy" or kind == "xlog1py":
        # BinaryMiscOpsKernels.cu: NaN y -> NaN, x == 0 -> 0, else x log(y).
        var l = _log(b) if kind == "xlogy" else _log1p(b)
        var r = a * l
        r = a.eq(0).select(zero, r)
        r = isnan(a).select(a, r)
        return isnan(b).select(SIMD[w, n](nan[w]()), r)
    elif kind == "igamma" or kind == "igammac":
        # IGammaKernel.cu calc_igamma / calc_igammac, float accscalar_t
        # (special_math.mojo; stock torch has no float64 route to match here).
        var r = SIMD[w, n]()
        comptime for i in range(n):
            var ai = a[i].cast[DType.float32]()
            var xi = b[i].cast[DType.float32]()
            comptime if kind == "igamma":
                r[i] = igamma_f(ai, xi).cast[w]()
            else:
                r[i] = igammac_f(ai, xi).cast[w]()
        return r
    elif kind == "zeta":
        var r = SIMD[w, n]()
        comptime for i in range(n):
            r[i] = _zeta_scalar(
                a[i].cast[DType.float64](), b[i].cast[DType.float64]()
            ).cast[w]()
        return (isnan(a) | isnan(b)).select(SIMD[w, n](nan[w]()), r)
    elif is_polynomial[kind]():
        var r = SIMD[w, n]()
        comptime for i in range(n):
            r[i] = _polynomial[kind, w](a[i], b[i])
        return r
    elif kind == "lerp_scalar":
        # lerp.Scalar: the weight is a parameter in opmath (Lerp.cu's
        # lerp_scalar_kernel converts it with weight.to<opmath_t>()).
        var diff = b - a
        var weight = SIMD[w, n](p[0])
        var small = abs(weight).lt(0.5)
        return small.select(weight.fma(diff, a), b - diff * (one - weight))
    elif kind == "lerp":
        # Lerp.h: self + weight * (end - self) for |weight| < 0.5, else
        # end - (end - self) * (1 - weight).
        var diff = b - a
        var small = abs(c).lt(0.5)
        return small.select(c.fma(diff, a), b - diff * (one - c))
    elif kind == "pow_scalar_base":
        # pow(Scalar base, Tensor exponent): PowKernel.cu's cpu-scalar base,
        # base ** exponent in opmath (the base rounded to it, not to the
        # tensor dtype). Computed in float64 through the libdevice exp/log,
        # with C's pow special cases: std.math.pow's GPU lowering returns 1
        # for an infinite exponent and inf for a NaN one.
        return _pow_scalar_base(
            p[0].cast[DType.float64](), a.cast[DType.float64]()
        ).cast[w]()
    # --- activations (forward: a = x) ----------------------------------------
    elif kind == "elu":
        # ActivationEluKernel.cu, p = (alpha, scale, input_scale):
        # x > 0 ? x * scale : expm1(x * input_scale) * alpha * scale.
        var negcoef = p[0] * p[1]
        var r = a.gt(0).select(a * p[1], _expm1(a * p[2]) * negcoef)
        return isnan(a).select(a, r)
    elif kind == "hardshrink":
        # p0 = lambd rounded to the input dtype by the host.
        var keep = a.ge(-p[0]) & a.le(p[0])
        return keep.select(zero, a)
    elif kind == "softshrink":
        var r = a.gt(p[0]).select(a - p[0], a.lt(-p[0]).select(a + p[0], zero))
        return isnan(a).select(a, r)
    elif kind == "hardsigmoid":
        var r = min(max(a + 3, zero), SIMD[w, n](6)) * (1.0 / 6.0)
        return isnan(a).select(a, r)
    elif kind == "hardswish":
        var r = a * min(max(a + 3, zero), SIMD[w, n](6)) * (1.0 / 6.0)
        return isnan(a).select(a, r)
    elif kind == "hardtanh":
        # hardtanh is clamp(x, min_val, max_val); NaN passes through.
        var r = min(max(a, SIMD[w, n](p[0])), SIMD[w, n](p[1]))
        return isnan(a).select(a, r)
    elif kind == "leaky_relu":
        var r = a.gt(0).select(a, a * p[0])
        return isnan(a).select(a, r)
    elif kind == "softplus":
        # p = (beta, threshold): x * beta > threshold ? x
        #                       : log1p(exp(x * beta)) / beta.
        var xb = a * p[0]
        var r = xb.gt(p[1]).select(a, _log1p(_exp(xb)) / p[0])
        return isnan(a).select(a, r)
    elif kind == "mish":
        # ActivationMishKernel.cu: x * tanh(log1p(exp(x))).
        return a * _tanh(_log1p(_exp(a)))
    elif kind == "threshold":
        # p = (threshold, value), both rounded to the input dtype by the
        # host (ActivationThresholdKernel.cu computes in scalar_t).
        var r = a.le(p[0]).select(SIMD[w, n](p[1]), a)
        return isnan(a).select(a, r)
    elif kind == "log_sigmoid":
        # ActivationLogSigmoidKernel.cu: min(0, x) - log1p(exp(-|x|)).
        var r = min(zero, a) - _log1p(_exp(-abs(a)))
        return isnan(a).select(a, r)
    elif kind == "rrelu_train" or kind == "rrelu_noise":
        # RreluWithNoise.cu, a = x, b = the uniform draw already in the
        # input dtype; p = (lower, upper - lower). The slope r = b * range +
        # lower is rounded to the input dtype (by the caller's final cast for
        # the noise, explicitly here for the product).
        var r = b.fma(p[1], p[0])
        var keep = a.le(0)
        comptime if kind == "rrelu_noise":
            return keep.select(r, one)
        else:
            return keep.select(a * r, a)
    # --- activation backwards ------------------------------------------------
    elif kind == "elu_backward":
        # a = grad, b = self or result; p = (alpha, scale, input_scale,
        # is_result).
        var negcoef = p[0] * p[1]
        var neg: SIMD[w, n]
        if p[3] != 0:
            neg = a * p[2] * (b + negcoef)
        else:
            neg = a * p[2] * negcoef * _exp(b * p[2])
        return b.le(0).select(neg, a * p[1])
    elif kind == "shrink_backward":
        # a = grad, b = self; hardshrink_backward and softshrink_backward.
        var zeroed = b.ge(-p[0]) & b.le(p[0])
        return zeroed.select(zero, a)
    elif kind == "hardsigmoid_backward":
        return (b.gt(-3) & b.lt(3)).select(a * (1.0 / 6.0), zero)
    elif kind == "hardswish_backward":
        var mid = a * (b / 3 + 0.5)
        return b.le(-3).select(zero, b.lt(3).select(mid, a))
    elif kind == "hardtanh_backward":
        return (b.le(p[0]) | b.ge(p[1])).select(zero, a)
    elif kind == "leaky_relu_backward":
        # a = self, b = grad (the CUDA iterator's operand order).
        return a.gt(0).select(b, b * p[0])
    elif kind == "softplus_backward":
        # a = grad, b = self; p = (beta, threshold).
        var xb = b * p[0]
        var z = _exp(xb)
        return xb.gt(p[1]).select(a, a * z / (z + 1))
    elif kind == "mish_backward":
        var s = _sigmoid(b)
        var t = _tanh(_log1p(_exp(b)))
        return a * (t + b * s * (1 - t * t))
    elif kind == "silu_backward":
        var s = _sigmoid(b)
        return a * s * (1 + b * (1 - s))
    elif kind == "log_sigmoid_backward":
        # a = self, b = grad.
        var neg = a.lt(0)
        var max_deriv = neg.select(one, zero)
        var sign = neg.select(one, -one)
        var z = _exp(-abs(a))
        return b * (max_deriv - sign * (z / (1 + z)))
    elif kind == "logit_backward":
        # a = grad, b = self; p0 = eps, < 0 for none.
        var d = a / (b * (1 - b))
        if p[0] < 0:
            return (b.lt(0) | b.gt(1)).select(SIMD[w, n](nan[w]()), d)
        return (b.lt(p[0]) | b.gt(1 - p[0])).select(zero, d)
    elif kind == "gelu_backward_none" or kind == "gelu_backward_tanh":
        # ActivationGeluKernel.cu, a = grad, b = self.
        comptime if kind == "gelu_backward_none":
            comptime kAlpha = 0.70710678118654752440
            comptime kBeta = 0.39894228040143267794
            var cdf = 0.5 * (1 + _erf(b * kAlpha))
            var pdf = _exp(-0.5 * b * b) * kBeta
            return a * (cdf + b * pdf)
        else:
            comptime kBeta = 0.79788456080286535588
            comptime kKappa = 0.044715
            var x_sq = b * b
            var x_cube = x_sq * b
            var inner = kBeta * (b + kKappa * x_cube)
            var t = _tanh(inner)
            var left = 0.5 * b
            var right = 1 + t
            var left_derivative = 0.5 * right
            var right_derivative = (
                left * (1 - t * t) * kBeta * (1 + 3 * kKappa * x_sq)
            )
            return a * (left_derivative + right_derivative)
    else:
        comptime assert False, "unknown pointwise kind"


@always_inline
def _pow_scalar_base[
    n: Int
](base: Float64, e: SIMD[DType.float64, n]) -> SIMD[DType.float64, n]:
    """C's pow(base, e) for a scalar base."""
    var r = _exp(e * _log(SIMD[DType.float64, n](abs(base))))
    var integral = e.eq(floor(e))
    # Parity of an integral e; every |e| >= 2^53 is even.
    var half = e * 0.5
    var odd = integral & half.ne(floor(half)) & abs(e).lt(9007199254740992.0)
    if base < 0:
        r = integral.select(
            odd.select(-r, r), SIMD[DType.float64, n](nan[DType.float64]())
        )
    if base == 0:
        # 0 ** e: +0 for e > 0, +inf for e < 0, signed for -0 and odd e.
        var neg_zero = bitcast[DType.int64](base) < 0
        var mag = e.lt(0).select(
            SIMD[DType.float64, n](inf[DType.float64]()),
            SIMD[DType.float64, n](0),
        )
        r = odd.select(-mag, mag) if neg_zero else mag
    if abs(base) == 1:
        # (+-1) ** +-inf is 1; -1 ** e otherwise follows the parity above.
        r = isinf(e).select(SIMD[DType.float64, n](1), r)
    r = isnan(e).select(e, r)
    r = e.eq(0).select(SIMD[DType.float64, n](1), r)
    if base == 1:
        return SIMD[DType.float64, n](1)
    return r


@always_inline
def _erf[
    w: DType, n: Int
](x: SIMD[w, n]) -> SIMD[w, n] where w.is_floating_point():
    return erf(x)


@always_inline
def _pow_general[
    w: DType, n: Int
](base: SIMD[w, n], e: SIMD[w, n]) -> SIMD[w, n] where w.is_floating_point():
    return pow(base, e)


@always_inline
def pointwise[
    kind: StaticString, dtype: DType, out_dtype: DType, n: Int
](
    a: SIMD[dtype, n],
    b: SIMD[dtype, n],
    c: SIMD[dtype, n],
    p: SIMD[param_dtype[dtype](), 4],
) -> SIMD[out_dtype, n]:
    """`kind` of (a, b, c) with scalar parameters `p`, rounded once into
    `out_dtype`. Unused operands and parameters are ignored."""
    comptime if kind == "frexp_exponent":
        return _frexp_exponent(a).cast[out_dtype]()
    elif kind == "hardtanh" and dtype.is_integral():
        # hardtanh is clamp_out(min_val, max_val): on integers, in scalar_t.
        var lo = SIMD[dtype, n](p[0].cast[dtype]())
        var hi = SIMD[dtype, n](p[1].cast[dtype]())
        return min(max(a, lo), hi).cast[out_dtype]()
    elif kind == "threshold" and dtype.is_integral():
        # ActivationThresholdKernel.cu on integers: in scalar_t.
        var thr = SIMD[dtype, n](p[0].cast[dtype]())
        return (
            a.le(thr)
            .select(SIMD[dtype, n](p[1].cast[dtype]()), a)
            .cast[out_dtype]()
        )
    elif is_native_kind[kind]():
        return _native[kind](a, b, c).cast[out_dtype]()
    else:
        comptime if dtype == DType.float64:
            return _wide[kind](
                a.cast[DType.float64](),
                b.cast[DType.float64](),
                c.cast[DType.float64](),
                p.cast[DType.float64](),
            ).cast[out_dtype]()
        else:
            return _wide[kind](
                a.cast[DType.float32](),
                b.cast[DType.float32](),
                c.cast[DType.float32](),
                p.cast[DType.float32](),
            ).cast[out_dtype]()
