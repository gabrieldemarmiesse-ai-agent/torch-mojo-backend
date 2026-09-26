# ===----------------------------------------------------------------------=== #
# Scalar ports of the CUDA float math routines ATen's unary kernels call.
#
# Stock torch on CUDA runs `::asinf`, `::erfcf`, `::lgammaf`, `::sinf`, ...
# (UnaryOpsKernel.cu, UnarySpecialOpsKernel.cu, and the jiterator strings of
# cuda/Math.cuh). Each body below is a literal translation of the matching
# `__nv_*` function of libdevice.10.bc with every `__nvvm_reflect` branch taken
# at value 0 (PyTorch is built without -ftz / fast math), the same convention
# as `libdevice_port.mojo`, whose `nv_logf` / `nv_expf` / `nv_tanf` these
# reuse. `llvm.nvvm.fma.rn` is `fma`; `mul.rn` / `add.rn` stay unfused
# (`_mul_rn` / `_add_rn`) because Mojo would contract them into an fma.
#
# The PTX approximation instructions libdevice uses (rcp/lg2/rsqrt.approx.ftz,
# ex2.approx, sqrt.approx) are emitted verbatim on NVIDIA and replaced by the
# IEEE operation elsewhere, so every other target (and the CPU, which runs the
# torch.compile graph on CPU tensors) computes the same algorithm to within
# an ulp rather than bit for bit.
# ===----------------------------------------------------------------------=== #

from std.math import exp2, fma, log2, sqrt
from std.sys import llvm_intrinsic
from std.sys._assembly import inlined_assembly
from std.sys.info import is_nvidia_gpu

from tmb.kernels.common.libdevice_port import (
    _f2i_rn,
    _trig_reduction_slowpath_f,
    nv_logf,
)


@always_inline
def _f(bits: UInt32) -> Float32:
    return Float32(from_bits=bits)


comptime _INF = Float32(from_bits=UInt32(0x7F800000))


@always_inline
def _abs(x: Float32) -> Float32:
    return llvm_intrinsic["llvm.fabs", Float32, has_side_effect=False](x)


@always_inline
def _trunc(x: Float32) -> Float32:
    return llvm_intrinsic["llvm.trunc", Float32, has_side_effect=False](x)


@always_inline
def _round_away(x: Float32) -> Float32:
    """llvm.nvvm.round.f: nearest integer, halfway cases away from zero."""
    return llvm_intrinsic["llvm.round", Float32, has_side_effect=False](x)


@always_inline
def _floor(x: Float32) -> Float32:
    return llvm_intrinsic["llvm.floor", Float32, has_side_effect=False](x)


@always_inline
def _copysign(mag: Float32, sgn: Float32) -> Float32:
    return Float32(
        from_bits=(mag.to_bits[DType.uint32]() & UInt32(0x7FFFFFFF))
        | (sgn.to_bits[DType.uint32]() & UInt32(0x80000000))
    )


@always_inline
def _mul_rn(a: Float32, b: Float32) -> Float32:
    """mul.rn.f32: a rounded product Mojo may not fuse into a later add."""
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "mul.rn.f32 $0, $1, $2;",
            Float32,
            constraints="=f,f,f",
            has_side_effect=False,
        ](a, b)
    else:
        return a * b


@always_inline
def _add_rn(a: Float32, b: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "add.rn.f32 $0, $1, $2;",
            Float32,
            constraints="=f,f,f",
            has_side_effect=False,
        ](a, b)
    else:
        return a + b


@always_inline
def _rcp_approx_ftz(x: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "rcp.approx.ftz.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return 1 / x


@always_inline
def _sqrt_approx(x: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "sqrt.approx.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return sqrt(x)


@always_inline
def _rsqrt_approx_ftz(x: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "rsqrt.approx.ftz.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return 1 / sqrt(x)


@always_inline
def _lg2_approx_ftz(x: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "lg2.approx.ftz.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return log2(x)


@always_inline
def _ex2_approx(x: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "ex2.approx.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return exp2(x)


@always_inline
def _ex2_approx_ftz(x: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "ex2.approx.ftz.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return exp2(x)


@always_inline
def ieee_sqrtf(x: Float32) -> Float32:
    """`::sqrtf` under nvcc's default -prec-sqrt=true: sqrt.rn.f32."""
    return llvm_intrinsic["llvm.sqrt", Float32, has_side_effect=False](x)


# --------------------------------------------------------------------------- #
# exp2f / log10f / expm1f
# --------------------------------------------------------------------------- #


@always_inline
def nv_exp2f(a: Float32) -> Float32:
    """`__nv_exp2f`: one ex2.approx.f32 (full range, non-ftz)."""
    return _ex2_approx(a)


@always_inline
def nv_log10f(a: Float32) -> Float32:
    """`__nv_log10f`: the `__nv_logf` body scaled by log10(e) (its zero case
    already reads -inf through the product)."""
    return nv_logf(a) * _f(0x3EDE5BD9)


@always_inline
def nv_expm1f(a: Float32) -> Float32:
    """`__nv_expm1f`."""
    var t = Float32(0.0)
    if not (_abs(a) < _f(0x3ED1EB85)):  # 0.41
        t = _round_away(a * _f(0x3FB8AA3B))
    var j = Float32(127.0) if t == Float32(128.0) else t
    var e = _ex2_approx(j)
    var r = fma(-t, _f(0x3F317200), a)
    r = fma(-t, _f(0x35BFBE8E), r)
    var p = fma(_f(0x3AB5EBE6), r, _f(0x3C095663))
    p = fma(p, r, _f(0x3D2AABE3))
    p = fma(p, r, _f(0x3E2AA9F6))
    p = fma(p, r, _f(0x3EFFFFFE))
    p = r * p
    p = fma(p, r, r)
    var u = fma(p, e, e + Float32(-1.0))
    if t == Float32(128.0):
        u = u + u
    if j > Float32(128.0):
        u = _INF
    if j < Float32(-25.0):
        u = Float32(-1.0)
    if a == Float32(0.0):
        u = a + a
    return u


# --------------------------------------------------------------------------- #
# asinf / atanf
# --------------------------------------------------------------------------- #


@always_inline
def nv_asinf(a: Float32) -> Float32:
    """`__nv_asinf`."""
    var x = _abs(a)
    var big = x > _f(0x3F11EB85)  # 0.57
    var s = _sqrt_approx((Float32(1.0) - x) * Float32(0.5))
    var d = s if big else x
    var d2 = d * d
    var p = fma(_f(0x3D53F941), d2, _f(0x3C94D2E9))
    p = fma(p, d2, _f(0x3D3F841F))
    p = fma(p, d2, _f(0x3D994929))
    p = fma(p, d2, _f(0x3E2AAB94))
    p = d2 * p
    var r = fma(p, d, d)
    var t = fma(Float32(-2.0), r, _f(0x3FC90FDB)) if big else r
    if t == t:  # ordered: copy the sign of the input
        return _copysign(t, a)
    return t


@always_inline
def nv_atanf(a: Float32) -> Float32:
    """`__nv_atanf`."""
    var x = _abs(a)
    var big = x > Float32(1.0)
    var t = Float32(1.0) / x if big else x
    var t2 = _mul_rn(t, t)
    var p = fma(t2, _f(0xBF52C7EA), _f(0xC0B59883))
    p = fma(p, t2, _f(0xC0D21907))
    p = t2 * p
    p = t * p
    var q = t2 + _f(0x41355DC0)
    q = fma(q, t2, _f(0x41E6BD60))
    q = fma(q, t2, _f(0x419D92C8))
    var r = fma(p, Float32(1.0) / q, t)
    if big:
        r = _f(0x3FC90FDB) - r
    if x == x:
        return _copysign(r, a)
    return r


# --------------------------------------------------------------------------- #
# erfcf / erfinvf
# --------------------------------------------------------------------------- #


@always_inline
def nv_erfcf(a: Float32) -> Float32:
    """`__nv_erfcf`."""
    var x = _abs(a)
    var num = x + Float32(-4.0)
    var rden = _rcp_approx_ftz(x + Float32(4.0))
    var q = _mul_rn(num, rden)
    var e = fma(Float32(-4.0), q + Float32(1.0), x)
    e = fma(-q, x, e)
    q = fma(rden, e, q)
    var p = fma(_f(0x3A69A091), q, _f(0x3BE6E05B))
    p = fma(p, q, _f(0xBC81FB4B))
    p = fma(p, q, _f(0x3D15373B))
    p = fma(p, q, _f(0xBD887C5A))
    p = fma(p, q, _f(0x3DC021D5))
    p = fma(p, q, _f(0xBDCED424))
    p = fma(p, q, _f(0x3D8B74DE))
    p = fma(p, q, _f(0x3C7BF170))
    p = fma(p, q, _f(0xBE0EF8D4))
    p = fma(p, q, _f(0x3F9DD2C9))
    var rd2 = _rcp_approx_ftz(fma(Float32(2.0), x, Float32(1.0)))
    var y = _mul_rn(p, rd2)
    var w = fma(x, y * Float32(-2.0), p)
    var f = fma(w - y, rd2, y)
    # exp(-x^2), with the rounding error of -x^2 folded back in.
    var nx2 = x * (-x)
    var jt = _trunc(_mul_rn(nx2, _f(0x3FB8AA3B)))
    var j = _copysign(Float32(126.0), jt) if _abs(jt) > Float32(126.0) else jt
    var r = fma(j, _f(0xBF317218), nx2)
    r = fma(j, _f(0x3102E308), r)
    var scale = Float32(
        from_bits=(j + _f(0x4B40007F)).to_bits[DType.uint32]() << UInt32(23)
    )
    var ex = _ex2_approx_ftz(r * _f(0x3FB8AA3B)) * scale
    var err = fma(-x, x, -nx2)
    var res = fma(ex, err, ex) * f
    if x > _f(0x4120E148):  # 10.055
        res = Float32(0.0)
    if a < Float32(0.0):
        res = Float32(2.0) - res
    return res


@always_inline
def nv_erfinvf(a: Float32) -> Float32:
    """`__nv_erfinvf`."""
    var w = _lg2_approx_ftz(fma(a, -a, Float32(1.0)))
    if w < _f(0xC1033333):  # -8.2
        var s = _rsqrt_approx_ftz(-w)
        var p = fma(_f(0xBF1704A1), s, _f(0xBF29BAA5))
        p = fma(p, s, _f(0x3FCC6ADC))
        p = fma(p, s, _f(0xBF2CDAED))
        p = fma(p, s, _f(0xBDC30537))
        p = fma(p, s, _f(0x3F55D9B9))
        return _copysign((Float32(1.0) / s) * p, a)
    var t = -w
    var p = fma(_f(0xAF8A6370), t, _f(0x3221F645))
    p = fma(p, t, _f(0xB4016FDA))
    p = fma(p, t, _f(0x3468F846))
    p = fma(p, t, _f(0x370742AA))
    p = fma(p, t, _f(0xB804DB4D))
    p = fma(p, t, _f(0xBA4AFEA1))
    p = fma(p, t, _f(0x3BB5C027))
    p = fma(p, t, _f(0x3E24AE0F))
    p = fma(p, t, _f(0x3F62DFC4))
    return a * p


# --------------------------------------------------------------------------- #
# sinf / cosf
# --------------------------------------------------------------------------- #


@always_inline
def _trig_reduce_f(a: Float32) -> Tuple[Float32, Int32]:
    """The argument reduction `__nv_sinf` / `__nv_cosf` share: a * 2/pi to the
    nearest quadrant, a three-term Cody-Waite residual, Payne-Hanek from
    |a| >= 105615 on."""
    var j = _f2i_rn(a * _f(0x3F22F983))
    var jf = j.cast[DType.float32]()
    var t = fma(jf, _f(0xBFC90FDA), a)
    t = fma(jf, _f(0xB3A22168), t)
    t = fma(jf, _f(0xA7C234C5), t)
    if _abs(a) >= Float32(105615.0):
        if _abs(a) == _INF:
            return (_mul_rn(a, Float32(0.0)), Int32(0))
        return _trig_reduction_slowpath_f(a)
    return (t, j)


@always_inline
def _sin_poly(t: Float32, i: Int32) -> Float32:
    """`__internal_sin_cos_kernel`: sin(t) for even quadrants, cos(t) for odd
    ones, negated in quadrants 2 and 3."""
    var t2 = _mul_rn(t, t)
    var even = (i & Int32(1)) == Int32(0)
    var s = t if even else Float32(1.0)
    var u = fma(t2, s, Float32(0.0))
    var c_odd = fma(_f(0x37CBAC00), t2, _f(0xBAB607ED))
    var c2 = _f(0xBE2AAAA8) if even else _f(0xBEFFFFFF)
    var c1 = _f(0x3C0885E4) if even else _f(0x3D2AAABB)
    var c0 = _f(0xB94D4153) if even else c_odd
    var p = fma(c0, t2, c1)
    p = fma(p, t2, c2)
    var z = fma(p, u, s)
    if (i & Int32(2)) != Int32(0):
        z = fma(z, Float32(-1.0), Float32(0.0))
    return z


@always_inline
def nv_sinf(a: Float32) -> Float32:
    """`__nv_sinf`."""
    var r = _trig_reduce_f(a)
    return _sin_poly(r[0], r[1])


@always_inline
def nv_cosf(a: Float32) -> Float32:
    """`__nv_cosf`: the sine kernel one quadrant on."""
    var r = _trig_reduce_f(a)
    return _sin_poly(r[0], r[1] + Int32(1))


# --------------------------------------------------------------------------- #
# lgammaf
# --------------------------------------------------------------------------- #


@always_inline
def _lgammaf_pos(x: Float32) -> Float32:
    """`__internal_lgammaf_pos`: lgamma(|a|)."""
    if x < Float32(3.0):
        if x < Float32(1.5):
            if x < _f(0x3F333333):  # 0.7
                var p = fma(_f(0x3B6B1C86), x, _f(0xBBB34878))
                p = fma(p, x, _f(0xBD36CAEF))
                p = fma(p, x, _f(0x3E2B5555))
                p = fma(p, x, _f(0xBD2C96C7))
                p = fma(p, x, _f(0xBF27E6EB))
                p = fma(p, x, _f(0x3F13C463))
                p = x * p
                var g = fma(p, x, x)  # 1 / Gamma(x)
                return -nv_logf(g)
            var y = Float32(1.0) - x
            var p = fma(_f(0x3D3BEF76), y, _f(0x3DD47577))
            p = fma(p, y, _f(0x3DFB8079))
            p = fma(p, y, _f(0x3E0295B5))
            p = fma(p, y, _f(0x3E12A765))
            p = fma(p, y, _f(0x3E2D6867))
            p = fma(p, y, _f(0x3E5462BF))
            p = fma(p, y, _f(0x3E8A8A72))
            p = fma(p, y, _f(0x3ECD26A4))
            p = fma(p, y, _f(0x3F528D32))
            p = fma(p, y, _f(0x3F13C468))
            return y * p
        var y = x + Float32(-2.0)
        var p = fma(_f(0x385007FA), y, _f(0xB967A002))
        p = fma(p, y, _f(0x3A0DE6FC))
        p = fma(p, y, _f(0xBA9DE0E2))
        p = fma(p, y, _f(0x3B3D05B7))
        p = fma(p, y, _f(0xBBF1EB10))
        p = fma(p, y, _f(0x3CA89A28))
        p = fma(p, y, _f(0xBD89F01A))
        p = fma(p, y, _f(0x3EA51A66))
        p = fma(p, y, _f(0x3ED87730))
        return y * p
    if x < _f(0x40F9999A):  # 7.8
        var y = x + Float32(-3.0)
        var n = fma(_f(0xC43B38FB), y, _f(0xC640F6F8))
        n = fma(n, y, _f(0xC7206560))
        n = fma(n, y, _f(0xC73CB6AA))
        n = fma(n, y, _f(0xC80BAE5A))
        var d = y + _f(0xC381A020)
        d = fma(d, y, _f(0xC62864B8))
        d = fma(d, y, _f(0xC7B50686))
        d = fma(d, y, _f(0xC8498465))
        return fma(n, _rcp_approx_ftz(d), y)
    # Stirling: (x - 1/2) log x - x + log(2 pi)/2 + series(1/x).
    var r = _rcp_approx_ftz(x)
    var r2 = r * r
    var s = fma(_f(0x3A4BE755), r2, _f(0xBB360953))
    s = fma(s, r2, _f(0x3DAAAAA3))
    s = fma(s, r, _f(0x3F6B3F8E))
    var hl = nv_logf(x) * Float32(0.5)
    var a = _mul_rn(hl, x + Float32(-0.5))
    var res = (a - x) + _add_rn(a, s)
    if x == _INF:
        res = _INF
    return res


@always_inline
def nv_lgammaf(a: Float32) -> Float32:
    """`__nv_lgammaf`: the positive kernel, and the reflection
    log(pi / |x sin(pi x)|) - lgamma(|x|) for negative non-integers."""
    var x = _abs(a)
    var t = _lgammaf_pos(x)
    if not (a < Float32(0.0)):
        return t
    if x == _floor(x):
        return _INF
    if x < _f(0x1FEC1E4A):  # 1e-19
        return -nv_logf(x)
    # sin(pi x) by the sine kernel on a quadrant reduction of 2x.
    var q = _round_away(x * Float32(2.0))
    var qi = q.cast[DType.int32]()
    var r = fma(-q, Float32(0.5), x) * _f(0x40490FDB)
    var s = _abs(_sin_poly(r, qi))
    return (_f(0x3F928682) - nv_logf(x * s)) - t
