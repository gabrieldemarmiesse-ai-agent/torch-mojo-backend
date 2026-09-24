"""Division SIMD expressions shared by eager and torch.compile execution."""

from std.memory import bitcast
from std.utils.numerics import inf


@always_inline
def true_div[
    dtype: DType, width: SIMDLength
](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """`torch.div`: IEEE division in the (floating) result dtype."""
    return a / b


@always_inline
def floor_div[
    dtype: DType, width: SIMDLength
](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """`torch.floor_divide`, ATen's div_floor_floating / div_floor_integer.

    Floats divide in at least float32: a bf16/fp16 quotient would round onto
    the wrong side of an integer before the floor (5.985 -> 6.0). ATen floors
    a nonzero numerator over an opposite-sign divisor to -1 even when the
    quotient underflows to -0 or the divisor is infinite, where `//` gives
    -0. A zero quotient already rules out a zero or NaN divisor. The
    numerator's bits decide nonzero, so a backend that flushes subnormals
    (Metal, the CPU `elementwise` lowering) still gets -1; its redundant
    finite test keeps LLVM from folding the pair into a float compare, which
    flushes too (float32 1e-38 // -1e10 gave 0 on a CPU graph without it).
    Integers take Mojo's `//`, which gives 0 for a zero divisor.
    """
    comptime if dtype.is_floating_point():
        comptime wide = DType.float64 if dtype == DType.float64 else DType.float32
        comptime bits = DType.uint64 if wide == DType.float64 else DType.uint32
        comptime sign = bitcast[bits](Scalar[wide](-0.0))
        comptime inf_bits = bitcast[bits](inf[wide]())
        var aw = a.cast[wide]()
        var bw = b.cast[wide]()
        var q = aw // bw
        var abits = bitcast[bits, width](aw)
        var bbits = bitcast[bits, width](bw)
        var amag = abits & ~sign
        var negative_zero = (
            q.eq(0)
            & amag.ne(0)
            & amag.lt(inf_bits)
            & ((abits ^ bbits) & sign).ne(0)
        )
        return negative_zero.select(SIMD[wide, width](-1), q).cast[dtype]()
    else:
        return a // b


@always_inline
def trunc_div[
    dtype: DType, width: SIMDLength
](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """`torch.div(rounding_mode="trunc")`, ATen's div_trunc_kernel.

    Floats divide at their own precision, rounding the quotient before the
    trunc, as ATen's tensor/tensor kernel does (bf16 -6.3125 / -1.0547 is
    6); callers reproduce its float32 scalar-divisor path by widening the
    operands first. Integers correct floor's `//` toward zero when the signs
    differ and the division is inexact; a zero divisor stays at `//`'s 0.
    """
    comptime if dtype.is_floating_point():
        return (a / b).__trunc__()
    else:
        var q = a // b
        var r = a - q * b
        var zero = SIMD[dtype, width](0)
        var opposite_signs = a.lt(zero) ^ b.lt(zero)
        var needs_adjust = opposite_signs & r.ne(zero) & b.ne(zero)
        return needs_adjust.select(q + 1, q)
