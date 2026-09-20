"""SIMD math shared by native kernels and graph custom operations."""

from std.math import cos, floor, sin, sqrt, tan
from std.math.polynomial import polynomial_evaluate
from std.sys import llvm_intrinsic
from std.sys.info import is_gpu, is_nvidia_gpu


# ===========================================================================
# Correctly rounded square root
# ===========================================================================
#
# `std.math.sqrt` is not IEEE-754 on NVIDIA. Its NVIDIA arm routes every float
# dtype through `_sqrt_nvvm`, i.e. `llvm.nvvm.sqrt.approx.ftz.f`
# (`mojo/stdlib/std/math/math.mojo`), and that is a property of the stdlib,
# not of any fast-math flag we could turn off. PTX documents `sqrt.approx` at
# up to 2 ulp, and `.ftz` flushes denormals to zero on input AND output, so a
# value whose true root is denormal comes back as exactly 0. Neither is a
# precision preference: a zeroed AdamW denominator is a wrong answer, and a
# 1-2 ulp drift means no eager op that returns a square root can be compared
# bit-for-bit against ATen on CPU or CUDA.
#
# `llvm.sqrt` lowers to `sqrt.rn.f32` / `sqrt.rn.f64` on NVPTX -- correctly
# rounded and denormal preserving -- so the override is a plain intrinsic
# swap; no inline PTX is needed. Every other target already reaches the right
# instruction through `std.math.sqrt` itself, which is why the fast path stays
# gated on `is_nvidia_gpu()`: AMD expands `llvm.sqrt` to `v_sqrt_f32` plus the
# denormal rescale and the +-1 ulp fma correction (verified in the emitted
# gfx942 assembly), and Apple uses `llvm.air.sqrt`.
@always_inline
def ieee_sqrt[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Elementwise square root that is correctly rounded on every backend.

    Use this, not `std.math.sqrt`, wherever the root reaches a user-visible
    result. `std.math.sqrt` remains the right call only where the value feeds
    a heuristic that never leaves the kernel.

    Parameters:
        dtype: Element type of the input and output vector.
        width: SIMD width of the input and output vector.

    Args:
        x: Vector to take the square root of.

    Returns:
        The elementwise square root of `x`.
    """
    comptime if is_nvidia_gpu() and dtype.is_floating_point():
        comptime if dtype in (DType.float16, DType.bfloat16):
            # Widening is exact, and f32 carries at least 2p+2 bits for both
            # 16-bit formats (24 >= 2*11+2 for f16, 24 >= 2*8+2 for bf16), so
            # rounding a correctly rounded f32 root back down is itself
            # correctly rounded -- the classic no-double-rounding bound.
            return llvm_intrinsic[
                "llvm.sqrt", SIMD[DType.float32, width], has_side_effect=False
            ](x.cast[DType.float32]()).cast[dtype]()
        else:
            return llvm_intrinsic[
                "llvm.sqrt", SIMD[dtype, width], has_side_effect=False
            ](x)
    else:
        return sqrt(x)


# Tangent: CPU -> libm; float32 on GPU -> the polynomial below; anything
# else -> sin(x) / cos(x).
#
# float32 GPU needs the polynomial because on NVIDIA `sin`/`cos` lower to a
# fixed ~1e-6-absolute-error approx instruction, and dividing them near a pole
# (cos(x) ~ 0) turns that into a large relative error -- 8.8% of a 20x20 randn
# sample failed conformance. AMD and Apple do not use that instruction, but
# take the polynomial too: it is what they already ran, and it needs no
# per-vendor accuracy claim.
@always_inline
def custom_tan[
    dtype: DType, width: SIMDLength, //, *, exact: Bool = True
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Elementwise tangent that keeps its relative accuracy on every backend.

    Use this, not `std.math.tan` (which refuses to compile for a GPU target)
    and not a hand-written `sin / cos` (which loses relative accuracy near the
    poles on NVIDIA).

    Parameters:
        dtype: Element type of the input and output vector.
        width: SIMD width of the input and output vector.
        exact: Whether float32 on GPU takes the argument-reduced polynomial
            path. Passing `False` asks for the raw hardware approximation
            instead -- two instructions and a divide, but wrong by a large
            relative factor near every pole -- and is only appropriate where
            the tangent feeds a heuristic that never leaves the kernel. Every
            other target/dtype combination ignores it: there is no cheaper
            route to drop to.

    Args:
        x: Vector to take the tangent of, in radians.

    Returns:
        The elementwise tangent of `x`.
    """
    comptime if not is_gpu():
        # `sin`/`cos` here would be the accurate LLVM intrinsics rather than
        # the approx PTX instructions, so neither GPU route below buys
        # anything over libm's own tangent.
        return tan(x)
    elif exact and dtype == DType.float32:
        # Reduce x to the nearest multiple of pi/2 (Cody-Waite, exact in
        # float32 for any |k| this op will realistically see) so the residual
        # r is always in [-pi/4, pi/4], away from every pole, then evaluate
        # tan(r) with a dedicated least-squares polynomial fit (fit against a
        # float64 reference; <2e-7 relative error in float32 arithmetic over
        # the polynomial's own domain). Near a pole the residual r is itself
        # small and well-conditioned, so -1/tan(r) reproduces the blow-up
        # without ever dividing two independently-rounded hardware trig
        # results against each other -- though for x within ~0.05 rad of an
        # exact pole, the end-to-end relative error can still exceed
        # torch.testing's default float32 rtol; that is inherent to
        # representing tan's unbounded derivative there in finite precision,
        # not specific to this polynomial.
        var af = x.cast[DType.float32]()
        comptime PIO2_HI = Float32(1.5703125)
        comptime PIO2_LO = Float32(0.00048382679233327506)
        comptime INV_PIO2 = Float32(0.63661977236758134308)

        var k = floor(af.fma(INV_PIO2, 0.5))
        var r = (af - k * PIO2_HI) - k * PIO2_LO
        var z = r * r
        var poly = polynomial_evaluate[
            [
                Float32(0.33333312008738518),
                0.13334750477592851,
                0.053745989665389061,
                0.023242133948206902,
                0.0050118292279914021,
                0.0082744075469672680,
            ],
        ](z)
        var tan_r = r + r * z * poly
        var is_odd = (k.cast[DType.int32]() & 1).cast[DType.bool]()
        return is_odd.select(-1 / tan_r, tan_r).cast[dtype]()
    else:
        return sin(x) / cos(x)
