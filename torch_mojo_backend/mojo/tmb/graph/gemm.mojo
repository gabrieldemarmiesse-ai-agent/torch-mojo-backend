"""The eager GEMM routes as MAX graph custom ops.

`torch.compile(backend=mojo_backend)` builds a MAX graph, and MAX's own
`matmul` would run Modular's kernels there while the same model in eager
mode runs this repository's (`tmb/ops/matmul.mojo`). These
registrations put the eager routes behind `F.custom`:

  `native_gemm`       C = A @ op(B)          one dense row-major 2-D operand each
  `native_gemm_bias`  C = A @ op(B) + bias   bias is one row of n
  `native_bmm`        C[i] = A[i] @ op(B[i]) dense strided batches

`op(B)` is `B` (k, n) or, with the `transpose_b` parameter, `B.T` for a `B`
stored (n, k) -- the `input @ weight.T` of every Linear. MAX materializes a
transposed producer before an opaque custom op (measured: the op always
sees row-major strides), so the Python caller
(`aten_functions._native_matmul`) looks through the `aten.t` node and
passes the stored weight plus the flag instead of `W.T`.

The tier ladder is the eager one -- gemm16 / TF32 tensor-core bridges on an
H100, then the generic SIMT routes -- reached through its comptime-dtype
entry points (`_gemm_transb_dispatch[dt]`, `_gemv_launch[dt]`,
`_matmul_bias_launch[dt]`): a MAX-compiled package carries no `-D`
defines, and `variant_gates.mojo`'s runtime-dtype dispatchers gate
everything OFF without them. `tf32` is the numerics decision torch owns
(`get_float32_matmul_precision() != "highest"`), taken in Python exactly as
`tmb/ops/matmul.mojo`'s `_tf32_enabled` takes it for eager mode.
"""
import extensibility as compiler
from extensibility import InputTensor, OutputTensor
from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys.info import _accelerator_arch

from tmb.kernels.gemm16_matmul.gemm16_candidate_dispatch import (
    try_enqueue_candidate_nt_bias_addr,
)
from tmb.kernels.gemm16_matmul.gemm16_dtype import _GEMM16_DT
from tmb.kernels.gemm16_matmul.gemm16_v3_kernels import (
    enqueue_gemm16_bmm_addr,
    enqueue_gemm16_gemm_addr,
)
from tmb.kernels.matmul.entry import (
    _bias_add_row,
    _gemm_transb_dispatch,
    _gemv_launch,
    _matmul_bias_launch,
)
from tmb.kernels.tf32_matmul.tf32_gemm_kernels import (
    enqueue_tf32_bmm_f32,
    enqueue_tf32_gemm_f32,
)

# The H100-class target both tensor-core bridges are written for (their
# eager gate is `tmb/ops/matmul.mojo`'s `_sm90_cuda`; here the target is a
# compile-time fact, MAX compiles the op for the device it runs on).
comptime _SM90 = "nvidia:sm_90a"


@always_inline
def _ptr[dtype: DType](addr: Int) -> Pointer[Scalar[dtype], MutAnyOrigin]:
    """A typed pointer over a raw address, in the origin the bridge entry
    points take (the `_make_ptr(...).as_unsafe_any_origin()` spelling of the
    eager bridges does not convert when the dtype is the gemm16 alias)."""
    return Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=addr)


def _gemm16_2d[
    transpose_b: Bool
](
    c: Int,
    a: Int,
    b: Int,
    bias: Int,
    has_bias: Bool,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """`tmb/ops/matmul.mojo`'s `_addmm_route` / `_mm_route` for bfloat16 on an
    H100: the fused NT+bias candidate first, then -- when the shape could
    reach a tensor-core route, which declines a bias -- the unbiased GEMM
    plus a separate row add, else gemm16's own bias epilogue."""
    var biasp = bias if has_bias else c
    if has_bias:
        comptime if transpose_b:
            if try_enqueue_candidate_nt_bias_addr(
                c, a, b, biasp, m, n, k, True, ctx
            ):
                return
        # `_alignment_favors_split`: every tile any tensor-core route uses
        # is a multiple of 64.
        if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
            enqueue_gemm16_gemm_addr(
                c, a, b, biasp, m, n, k, False, transpose_b, False, ctx
            )
            _bias_add_row[_GEMM16_DT](c, bias, m * n, n, ctx)
            return
    enqueue_gemm16_gemm_addr(
        c, a, b, biasp, m, n, k, False, transpose_b, has_bias, ctx
    )


def _gemm_2d[
    dtype: DType, transpose_b: Bool, tf32: Bool
](
    c: Int,
    a: Int,
    b: Int,
    bias: Int,
    has_bias: Bool,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """One dense 2-D GEMM through the eager tier ladder."""
    if m == 0 or n == 0:
        return
    if k == 0:
        # An empty reduction: ATen defines the product as zeros (+ bias).
        ctx.enqueue_memset(
            DeviceBuffer[dtype](ctx, _ptr[dtype](c), m * n, owning=False),
            Scalar[dtype](0),
        )
        if has_bias:
            _bias_add_row[dtype](c, bias, m * n, n, ctx)
        return
    comptime if _accelerator_arch() == _SM90:
        comptime if dtype == _GEMM16_DT:
            _gemm16_2d[transpose_b](c, a, b, bias, has_bias, m, n, k, ctx)
            return
        comptime if dtype == DType.float32 and tf32:
            enqueue_tf32_gemm_f32(
                _ptr[DType.float32](c),
                _ptr[DType.float32](a),
                _ptr[DType.float32](b),
                _ptr[DType.float32](bias if has_bias else c),
                m,
                n,
                k,
                False,
                transpose_b,
                has_bias,
                ctx,
            )
            return
    comptime TB = 1 if transpose_b else 0
    if has_bias:
        _matmul_bias_launch[dtype](c, a, b, bias, m, n, k, TB, ctx)
    elif m == 1:
        _gemv_launch[dtype](c, a, b, m, n, k, TB, ctx)
    else:
        _gemm_transb_dispatch[dtype](c, a, b, 1, m, n, k, m * k, TB, ctx)


def _bmm_3d[
    dtype: DType, transpose_b: Bool, tf32: Bool
](
    c: Int,
    a: Int,
    b: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """One dense batched GEMM through the eager tier ladder (`_bmm_route`)."""
    if batch == 0 or m == 0 or n == 0:
        return
    if k == 0:
        ctx.enqueue_memset(
            DeviceBuffer[dtype](
                ctx, _ptr[dtype](c), batch * m * n, owning=False
            ),
            Scalar[dtype](0),
        )
        return
    comptime if _accelerator_arch() == _SM90:
        comptime if dtype == _GEMM16_DT:
            enqueue_gemm16_bmm_addr(
                c,
                a,
                b,
                batch,
                m,
                n,
                k,
                m * n,
                m * k,
                k * n,
                False,
                transpose_b,
                ctx,
            )
            return
        comptime if dtype == DType.float32 and tf32:
            enqueue_tf32_bmm_f32(
                _ptr[DType.float32](c),
                _ptr[DType.float32](a),
                _ptr[DType.float32](b),
                batch,
                m,
                n,
                k,
                m * n,
                m * k,
                k * n,
                False,
                transpose_b,
                ctx,
            )
            return
    _gemm_transb_dispatch[dtype](
        c, a, b, batch, m, n, k, m * k, 1 if transpose_b else 0, ctx
    )


@compiler.register("native_gemm")
struct NativeGemm:
    @staticmethod
    def execute[
        dtype: DType,
        //,
        transpose_b: Bool,
        tf32: Bool,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        a: InputTensor[dtype=dtype, rank=2, ...],
        b: InputTensor[dtype=dtype, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if target == "gpu":
            var m = a.dim_size(0)
            var k = a.dim_size(1)
            var n = b.dim_size(0) if transpose_b else b.dim_size(1)
            _gemm_2d[dtype, transpose_b, tf32](
                Int(output.unsafe_ptr()),
                Int(a.unsafe_ptr()),
                Int(b.unsafe_ptr()),
                0,
                False,
                m,
                n,
                k,
                ctx,
            )
        else:
            raise Error("native_gemm runs on an accelerator only")


@compiler.register("native_gemm_bias")
struct NativeGemmBias:
    @staticmethod
    def execute[
        dtype: DType,
        //,
        transpose_b: Bool,
        tf32: Bool,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        a: InputTensor[dtype=dtype, rank=2, ...],
        b: InputTensor[dtype=dtype, rank=2, ...],
        bias: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if target == "gpu":
            var m = a.dim_size(0)
            var k = a.dim_size(1)
            var n = b.dim_size(0) if transpose_b else b.dim_size(1)
            _gemm_2d[dtype, transpose_b, tf32](
                Int(output.unsafe_ptr()),
                Int(a.unsafe_ptr()),
                Int(b.unsafe_ptr()),
                Int(bias.unsafe_ptr()),
                True,
                m,
                n,
                k,
                ctx,
            )
        else:
            raise Error("native_gemm_bias runs on an accelerator only")


@compiler.register("native_bmm")
struct NativeBmm:
    @staticmethod
    def execute[
        dtype: DType,
        //,
        transpose_b: Bool,
        tf32: Bool,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        a: InputTensor[dtype=dtype, rank=3, ...],
        b: InputTensor[dtype=dtype, rank=3, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if target == "gpu":
            var batch = a.dim_size(0)
            var m = a.dim_size(1)
            var k = a.dim_size(2)
            var n = b.dim_size(1) if transpose_b else b.dim_size(2)
            _bmm_3d[dtype, transpose_b, tf32](
                Int(output.unsafe_ptr()),
                Int(a.unsafe_ptr()),
                Int(b.unsafe_ptr()),
                batch,
                m,
                n,
                k,
                ctx,
            )
        else:
            raise Error("native_bmm runs on an accelerator only")
