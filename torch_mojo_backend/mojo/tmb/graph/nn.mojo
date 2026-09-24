"""The eager softmax, layer-norm and embedding kernels as MAX custom ops.

  `native_softmax_rows`  softmax over the trailing dim of a (rows, cols) operand
  `native_layer_norm`    layer norm over the trailing dim of a (rows, cols)
                         operand, with the float32 per-row mean and rstd ATen
                         returns (the CUDA contract: `at::toAccumulateType`)
  `native_embedding`     one row of the table per index, indices flattened

Python (`aten_functions.py`) flattens the leading dimensions before the
call and restores them after, so every op here is rank-2 and the kernels
are exactly the ones `tmb/ops/nn.mojo` launches
(`SoftmaxSpec` -> `_softmax_rows`, `LayerNormForward` -> `enqueue_norm_rows`,
`Gather0` -> `_gather0`), by their comptime-dtype entry points.
"""
import extensibility as compiler
from extensibility import InputTensor, OutputTensor
from max.gpu.host import DeviceContext
from std.memory import bitcast

from tmb.kernels.nn.gather_kernels import _gather0
from tmb.kernels.nn.softmax_rows_kernels import _softmax_rows
from tmb.kernels.normalization_forward.kernels import (
    AFFINE_COL,
    enqueue_norm_rows,
)


@compiler.register("native_softmax_rows")
struct NativeSoftmaxRows:
    @staticmethod
    def execute[
        dtype: DType, //, target: StaticString
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if target == "gpu":
            var rows = x.dim_size(0)
            var cols = x.dim_size(1)
            if rows == 0 or cols == 0:
                return
            # scale 1, no causal mask, q_len 1: the plain aten::_softmax.
            _softmax_rows[dtype](
                Int(output.unsafe_ptr()),
                Int(x.unsafe_ptr()),
                rows,
                cols,
                Float32(1),
                0,
                1,
                ctx,
            )
        else:
            raise Error("native_softmax_rows runs on an accelerator only")


@compiler.register("native_layer_norm")
struct NativeLayerNorm:
    @staticmethod
    def execute[
        dtype: DType,
        //,
        # ATen's `float eps`, passed as the bit pattern of its float64: MAX
        # custom-op parameters are bool / int / str / dtype.
        eps_bits: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        mean: OutputTensor[dtype=DType.float32, rank=1, ...],
        rstd: OutputTensor[dtype=DType.float32, rank=1, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        weight: InputTensor[dtype=dtype, rank=1, ...],
        bias: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if target == "gpu":
            var eps = Float32(bitcast[DType.float64](Int64(eps_bits)))
            # hxw / cpg / group are read only by the group-norm affine.
            enqueue_norm_rows[dtype, AFFINE_COL](
                Int(output.unsafe_ptr()),
                Int(mean.unsafe_ptr()),
                Int(rstd.unsafe_ptr()),
                Int(x.unsafe_ptr()),
                Int(weight.unsafe_ptr()),
                Int(bias.unsafe_ptr()),
                x.dim_size(0),
                x.dim_size(1),
                eps,
                1,
                1,
                1,
                True,
                True,
                ctx,
            )
        else:
            raise Error("native_layer_norm runs on an accelerator only")


@compiler.register("native_embedding")
struct NativeEmbedding:
    @staticmethod
    def execute[
        dtype: DType, idx_dtype: DType, //, target: StaticString
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        indices: InputTensor[dtype=idx_dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if target == "gpu":
            var num_indices = indices.dim_size(0)
            var row_len = weight.dim_size(1)
            if num_indices == 0 or row_len == 0:
                return
            _gather0[dtype, idx_dtype](
                Int(output.unsafe_ptr()),
                Int(weight.unsafe_ptr()),
                Int(indices.unsafe_ptr()),
                num_indices,
                row_len,
                weight.dim_size(0),
                ctx,
            )
        else:
            raise Error("native_embedding runs on an accelerator only")
