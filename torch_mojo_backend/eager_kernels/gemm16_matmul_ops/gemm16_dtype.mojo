# ===----------------------------------------------------------------------=== #
# Which 16-bit tensor-core dtype this build of the GEMM family carries.
#
# bfloat16 and float16 are the same thing to Hopper's tensor cores: identical
# operand width, identical WGMMA tile shapes, identical FP32 accumulator, and
# only the operand-type token of the instruction differs.  Every tile size,
# pipeline depth, TMA descriptor and wave-aware grid in this directory is a
# function of the operand WIDTH, so one source serves both dtypes and the
# choice is made once, here, at compile time.
#
# The selector is a compile-time define, the same mechanism every other kernel
# family in this package uses: the loader already compiles one .so per
# (OP, dtype, flags) tuple and `DTYPE_ARG_0` already carries the operand dtype
# of the call, so nothing new travels at runtime.
#
# bfloat16 is the default for EVERY other value of the define, including the
# absent one.  That is deliberate rather than incidental: it is what makes a
# build that never heard of this define -- `mojo build` with no `-D`, or
# scripts/compare_kernel_asm.py's float32 pass -- emit exactly the bfloat16
# kernels it emitted before this family was parametrized, which is how the
# bfloat16 half of the family stays byte-invariant under the parametrization.
# ===----------------------------------------------------------------------=== #

from variant_gates import _dtype_arg_on


@always_inline
def _gemm16_dtype() -> DType:
    """The 16-bit operand dtype this specialization was compiled for."""
    comptime if _dtype_arg_on[0, DType.float16]():
        return DType.float16
    else:
        return DType.bfloat16


@always_inline
def _gemm16_tag() -> StaticString:
    """The dtype token every kernel of this family carries in its name.

    Kernel names are what CUPTI, Nsight and torch.profiler print, so a user
    profiling a float16 model has to read "f16" there and never "bf16".
    """
    comptime if _gemm16_dtype() == DType.float16:
        return "f16"
    else:
        return "bf16"


comptime _GEMM16_DT = _gemm16_dtype()
comptime _GEMM16_TAG = _gemm16_tag()
