import extensibility as compiler
from extensibility import ElementwiseBinaryOp

from tmb.kernels.common.div_math import floor_div, trunc_div, true_div


@compiler.register("div_true")
struct DivTrueKernel(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype.is_floating_point(), "div_true takes floats"
        return true_div(lhs, rhs)


@compiler.register("div_floor")
struct DivFloorKernel(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return floor_div(lhs, rhs)


@compiler.register("div_trunc")
struct DivTruncKernel(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return trunc_div(lhs, rhs)
