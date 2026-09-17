"""SIMD math shared by native device kernels and MAX graph custom ops."""

from std.math import acos
from std.utils.numerics import isnan, nan


@always_inline
def acos_value[
    dtype: DType, width: SIMDLength
](x: SIMD[dtype, width],) -> SIMD[dtype, width]:
    comptime assert (
        dtype.is_floating_point()
    ), "acos requires floating point input"
    comptime if dtype == DType.float16 or dtype == DType.bfloat16:
        return acos_value(x.cast[DType.float32]()).cast[dtype]()
    else:
        # Mojo 1.0's float32 acos clamps abs(x) to [0, 1]. Restore ATen's
        # NaN result outside the domain, including infinities and NaN inputs.
        var invalid = abs(x).gt(1) | isnan(x)
        return invalid.select(SIMD[dtype, width](nan[dtype]()), acos(x))
