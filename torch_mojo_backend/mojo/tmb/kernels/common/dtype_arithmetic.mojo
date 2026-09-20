"""Input-dtype products that preserve Half operator rounding."""

from std.sys import inlined_assembly, llvm_intrinsic, is_nvidia_gpu, is_amd_gpu


@always_inline
def _product[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float16 and is_nvidia_gpu():
        # Explicit .rn also prevents ptxas from contracting c10::Half operators.
        return inlined_assembly[
            "mul.rn.f16 $0, $1, $2;",
            Scalar[dt],
            constraints="=h,h,h",
            has_side_effect=False,
        ](a, b)
    elif dt == DType.float16 and is_amd_gpu():
        return llvm_intrinsic[
            "llvm.arithmetic.fence", Scalar[dt], has_side_effect=False
        ](a * b)
    elif dt == DType.float16:
        var value = a * b
        var ptr = Pointer(to=value)
        ptr.unsafe_store[volatile=True](0, value)
        return ptr.unsafe_load[volatile=True](0)
    else:
        return a * b
