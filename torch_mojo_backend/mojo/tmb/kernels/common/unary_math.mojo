"""Unary SIMD expressions shared by eager and torch.compile execution."""

from std.math import (
    acos,
    atanh,
    ceil,
    cos,
    cosh,
    copysign,
    erf,
    exp,
    floor,
    log,
    log1p,
    log2,
    sin,
    sinh,
    sqrt,
    tanh,
)
from std.memory import bitcast
from std.sys.info import is_apple_gpu, is_nvidia_gpu
from std.utils.numerics import isnan, max_or_inf, nan
from tmb.kernels.common.math_utils import custom_tan, ieee_sqrt


@always_inline
def _log1p_nonneg[
    dtype: DType, width: SIMDLength
](t: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """log1p for finite t >= 0 (special values are the caller's to select).

    float32 is CUDA's log1pf (libdevice): 1 + t = 2^i (1 + m) by exponent
    arithmetic on the bits, m in about [-1/4, 1/2], then a degree-9
    polynomial -- no division and no hardware lg2 approximation, whose
    absolute error near one the compensated log1p kind has to repair.
    """
    comptime if dtype == DType.float32:
        var x = rebind[SIMD[DType.float32, width]](t)
        var e = (bitcast[DType.int32](x + 1) - 0x3F400000) & -8388608
        var m = bitcast[DType.float32](bitcast[DType.int32](x) - e)
        var s = bitcast[DType.float32](0x40800000 - e)
        m += SIMD[DType.float32, width](0.25).fma(s, -1)
        var i = e.cast[DType.float32]() * 1.1920928955078125e-7
        var p = SIMD[DType.float32, width](-0.04534861445426941)
        p = p.fma(m, 0.10546888411045074)
        p = p.fma(m, -0.13229703903198242)
        p = p.fma(m, 0.14491446316242218)
        p = p.fma(m, -0.16641564667224884)
        p = p.fma(m, 0.199888676404953)
        p = p.fma(m, -0.2500019669532776)
        p = p.fma(m, 0.33333510160446167)
        p = p.fma(m, -0.5)
        var r = (p * m).fma(m, m)
        return rebind[SIMD[dtype, width]](i.fma(0.6931471824645996, r))
    else:
        return log1p(t)


@always_inline
def _acosh[
    dtype: DType, width: SIMDLength, //, exact_sqrt: Bool
](a: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """acosh as CUDA's acoshf computes it (libdevice __nv_acoshf, what stock
    torch runs): log1p(d + sqrt(d (x + 1))) with d = x - 1, exact near one.

    Past d = 2^23 acosh(x) = log(2x) = log1p(d) + ln 2 instead, where
    d (x + 1) would overflow float32. `exact_sqrt=False` takes the hardware
    approximation (~2^-22 relative on NVIDIA), for float16/bfloat16 inputs
    whose result is rounded to 11 or 8 bits anyway: the correctly rounded
    root cost ~10 instructions per element. std.math.acosh is libm/CPU-only.
    """
    var d = a - 1
    var large = ~(d.ge(0) & d.le(8388608.0))
    var prod = a.fma(d, d)
    comptime if exact_sqrt:
        prod = ieee_sqrt(prod)
    else:
        prod = sqrt(prod)
    var res = _log1p_nonneg(large.select(d, d + prod)) + large.select(
        SIMD[dtype, width](0.69314718055994530942), 0
    )
    # One mask for every special value: NaN below one and for NaN, +inf
    # at +inf; only finite x >= 1 keeps the computed value.
    var ge1 = a.ge(1)
    return (ge1 & a.lt(max_or_inf[dtype]())).select(
        res, ge1.select(a, SIMD[dtype, width](nan[dtype]()))
    )


@always_inline
def _float_unary[
    kind: StaticString, dtype: DType, width: SIMDLength
](a: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """The float-only unary math, evaluated in `dtype` (float32 or float64).

    Only instantiated for float32/float64 (half inputs are promoted before
    the call), so every std.math call below sees a supported dtype.
    """
    comptime assert (
        kind == "exp"
        or kind == "tanh"
        or kind == "ceil"
        or kind == "floor"
        or kind == "acos"
        or kind == "acosh"
        or kind == "asinh"
        or kind == "atanh"
        or kind == "cos"
        or kind == "cosh"
        or kind == "erf"
        or kind == "log"
        or kind == "log1p"
        or kind == "log2"
        or kind == "reciprocal"
        or kind == "rsqrt"
        or kind == "sigmoid"
        or kind == "silu"
        or kind == "sin"
        or kind == "sinh"
        or kind == "sqrt"
        or kind == "tan"
        or kind == "gelu_none"
        or kind == "gelu_tanh"
    ), "unsupported elementwise kind"
    var res = a
    comptime if kind == "exp":
        res = exp(a)
    comptime if kind == "tanh":
        res = tanh(a)
    comptime if kind == "ceil":
        res = ceil(a)
    comptime if kind == "floor":
        res = floor(a)
    comptime if kind == "acos":
        res = acos(a)
        # std.math.acos clamps outside [-1, 1]; ATen requires NaN.
        comptime if is_nvidia_gpu() and dtype == DType.float32:
            # Integer masking keeps NVIDIA's packed float32-to-half conversion;
            # a floating select lets LLVM split it into scalar conversions.
            var magnitude = bitcast[DType.uint32](a) & 0x7FFFFFFF
            var invalid = magnitude.gt(0x3F800000)
            var nan_bits = invalid.cast[DType.uint32]() * 0x7FC00000
            res = bitcast[dtype](bitcast[DType.uint32](res) | nan_bits)
        else:
            res = (abs(a).gt(1) | isnan(a)).select(
                SIMD[dtype, width](nan[dtype]()), res
            )
    comptime if kind == "acosh":
        res = _acosh[exact_sqrt=True](a)
    comptime if kind == "asinh":
        # asinh(x) = log(x + sqrt(x^2 + 1)); std.math.asinh is libm/CPU-only.
        res = log(a + ieee_sqrt(a * a + 1))
        res = (abs(a).eq(max_or_inf[dtype]()) | a.eq(0)).select(a, res)
    comptime if kind == "atanh":
        res = atanh(a)
    comptime if kind == "cos":
        res = cos(a)
    comptime if kind == "cosh":
        res = cosh(a)
    comptime if kind == "erf":
        res = erf(a)
    comptime if kind == "log":
        # Preserve +inf, which the stdlib CPU approximation treats as finite.
        res = a.eq(max_or_inf[dtype]()).select(a, log(a))
    comptime if kind == "log2":
        res = log2(a)
        comptime if dtype == DType.float64 or is_apple_gpu():
            # The double approximation and Metal's float log2 omit +inf.
            res = a.eq(max_or_inf[dtype]()).select(a, res)
    comptime if kind == "log1p":
        comptime if is_apple_gpu() or (
            is_nvidia_gpu() and dtype == DType.float32
        ):
            # std.log1p promotes to float64: unsupported on Metal and costly
            # on NVIDIA. Reuse the compensated float32 Metal algorithm;
            # the NVIDIA near-zero polynomial below handles lg2.approx error.
            var xp1 = 1 + a
            var rc = log(xp1)
            var corrected = rc * (a / (xp1 - 1))
            rc = (a.gt(-0.5) & a.lt(0.5)).select(corrected, rc)
            res = xp1.eq(1).select(a, rc)
            comptime if is_nvidia_gpu() and dtype == DType.float32:
                # Avoid lg2.approx's absolute error near one. The degree-eight
                # Taylor remainder is below 7e-9 relative for |x| < 1/8.
                var p = SIMD[dtype, width](-1 / 8)
                p = p.fma(a, 1 / 7)
                p = p.fma(a, -1 / 6)
                p = p.fma(a, 1 / 5)
                p = p.fma(a, -1 / 4)
                p = p.fma(a, 1 / 3)
                p = p.fma(a, -1 / 2)
                var small = a.fma(a * p, a)
                res = abs(a).lt(0.125).select(small, res)
        else:
            res = log1p(a)
        res = a.eq(max_or_inf[dtype]()).select(a, res)
    comptime if kind == "reciprocal":
        res = 1 / a
    comptime if kind == "rsqrt":
        res = 1 / ieee_sqrt(a)
    comptime if kind == "sigmoid":
        res = 1 / (1 + exp(-a))
    comptime if kind == "silu":
        res = a / (1 + exp(-a))
    comptime if kind == "sin":
        res = sin(a)
    comptime if kind == "sinh":
        res = sinh(a)
        # std.sinh's expm1 approximation saturates large CPU inputs and
        # computes inf/inf at infinity. exp(|x|/2)^2/2 keeps the valid
        # finite range before overflow and restores the signed infinities.
        var half_exp = exp(abs(a) * 0.5)
        var large = copysign((0.5 * half_exp) * half_exp, a)
        res = abs(a).gt(20).select(large, res)
    comptime if kind == "sqrt":
        res = ieee_sqrt(a)
    comptime if kind == "tan":
        # `custom_tan` picks libm, the argument-reduced polynomial or
        # `sin / cos` from the compilation target and `dtype` on its own.
        res = custom_tan(a)
    comptime if kind == "gelu_none":
        # 0.5 * x * (1 + erf(x / sqrt(2)))
        comptime inv_sqrt2 = 0.70710678118654752440
        res = 0.5 * a * (1 + erf(a * inv_sqrt2))
    comptime if kind == "gelu_tanh":
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        comptime sqrt_2_over_pi = 0.79788456080286535588
        var inner = sqrt_2_over_pi * (a + 0.044715 * a * a * a)
        res = 0.5 * a * (1 + tanh(inner))
    # CPU stdlib approximations omit NaNs for exp/tanh/cosh. NVIDIA's float32
    # math preserves them already, so avoid an extra select on every lane.
    # Retain the existing correction on other backends and float64.
    comptime if not (is_nvidia_gpu() and dtype == DType.float32):
        return isnan(a).select(a, res)
    else:
        return res


@always_inline
def elementwise_unary[
    kind: StaticString, dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime if kind == "relu":
        # Return the input for NaN and signed zero, as ATen relu does.
        return x.lt(0).select(SIMD[dtype, width](0), x)
    elif kind == "abs":
        return abs(x)
    elif kind == "neg":
        return -x
    elif kind == "sign":
        comptime if dtype == DType.bool:
            return x
        else:
            # Both comparisons are false for NaN, matching ATen's zero result.
            return x.gt(0).cast[dtype]() - x.lt(0).cast[dtype]()
    elif (kind == "ceil" or kind == "floor") and dtype.is_integral():
        return x
    elif kind == "acosh" and (
        dtype == DType.float16 or dtype == DType.bfloat16
    ):
        return _acosh[exact_sqrt=False](x.cast[DType.float32]()).cast[dtype]()
    elif dtype == DType.float16 or dtype == DType.bfloat16:
        # Compute once in float32 and round only the final result.
        return _float_unary[kind](x.cast[DType.float32]()).cast[dtype]()
    else:
        comptime assert (
            dtype.is_floating_point()
        ), "floating point input required"
        return _float_unary[kind](x)


@always_inline
def elementwise_predicate[
    kind: StaticString, dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
    comptime if kind == "isnan":
        # Bit-based fpclass survives fast-math and is false for integers.
        return isnan(x)
    elif kind == "logical_not":
        return x.eq(SIMD[dtype, width](0))
    else:
        comptime assert False, "unsupported elementwise predicate"
