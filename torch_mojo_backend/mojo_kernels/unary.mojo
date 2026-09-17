import extensibility as compiler
from extensibility import ElementwiseUnaryOp

from .unary_math import acos_value


@compiler.register("tmb_acos")
struct AcosKernel(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType, width: SIMDLength
    ](x: SIMD[dtype, width],) -> SIMD[dtype, width]:
        return acos_value(x)
