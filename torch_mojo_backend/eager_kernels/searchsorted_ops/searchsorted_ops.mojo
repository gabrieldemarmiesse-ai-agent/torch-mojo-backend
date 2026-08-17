# ===----------------------------------------------------------------------=== #
# Dynamic CPU/GPU binary search for aten::searchsorted and aten::bucketize.
#
# One logical worker searches one input value.  A 1-D boundary is shared by
# every value; an N-D boundary selects the matching flattened prefix row.
# Optional sorter entries are relative indices within the final dimension,
# matching ATen.  Python validates shapes, devices, dtypes, and sorter bounds
# before this raw-pointer bridge is called.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator
from std.utils.coord import Coord
from std.utils.numerics import isnan
from std.utils.static_tuple import StaticTuple

from op_utils import (
    GS_THREADS,
    _enqueue_cached,
    _gs_blocks,
    _make_ptr,
    _parallel_for,
    _raw_ctx,
    _raw_dtype_int,
    _raw_int,
    _raw_ret_none,
    _spec_dispatcher13,
    _spec_unsupported,
)
from variant_gates import (
    _dtype_arg_on,
    _dtype_out_on,
    _op_on,
    _register_call,
)


comptime SEARCH_DTYPES = [
    DType.float32,
    DType.bfloat16,
    DType.float16,
    DType.int32,
    DType.int64,
]
comptime OUTPUT_DTYPES = [DType.int32, DType.int64]


@always_inline
def _binary_search_position[
    dtype: DType
](
    boundaries: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    values: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    sorter: UnsafePointer[Scalar[DType.int64], ImmutAnyOrigin],
    value_index: Int,
    boundary_size: Int,
    values_per_batch: Int,
    boundaries_are_1d: Bool,
    has_sorter: Bool,
    right: Bool,
) -> Int:
    var boundary_base = 0
    if not boundaries_are_1d:
        boundary_base = (value_index // values_per_batch) * boundary_size

    var value = SIMD[dtype, 1](values[value_index])
    var value_is_nan = isnan(value)
    var low = 0
    var high = boundary_size
    while low < high:
        var mid = low + ((high - low) >> 1)
        var boundary_index = mid
        if has_sorter:
            boundary_index = Int(sorter[boundary_base + mid])
        var boundary = SIMD[dtype, 1](
            boundaries[boundary_base + boundary_index]
        )
        var advance = (
            value_is_nan | (boundary.le(value) if right else boundary.lt(value))
        )[0]
        # Conditional expressions lower to selects: the data-dependent update
        # is branchless even though every lane takes a different search path.
        low = mid + 1 if advance else low
        high = high if advance else mid
    return low


@__name(t"binary_search_global_{dtype}_{out_dtype}_t{GS_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GS_THREADS))
)
def _binary_search_kernel[
    dtype: DType, out_dtype: DType
](
    out_ptr: UnsafePointer[Scalar[out_dtype], MutAnyOrigin],
    boundaries: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    values: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    sorter: UnsafePointer[Scalar[DType.int64], ImmutAnyOrigin],
    num_values_arg: Int64,
    boundary_size_arg: Int64,
    values_per_batch_arg: Int64,
    boundaries_are_1d_arg: Int64,
    has_sorter_arg: Int64,
    right_arg: Int64,
):
    var num_values = Int(num_values_arg)
    var boundary_size = Int(boundary_size_arg)
    var values_per_batch = Int(values_per_batch_arg)
    var boundaries_are_1d = Bool(boundaries_are_1d_arg)
    var has_sorter = Bool(has_sorter_arg)
    var right = Bool(right_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < num_values:
        out_ptr[i] = _binary_search_position[dtype](
            boundaries,
            values,
            sorter,
            i,
            boundary_size,
            values_per_batch,
            boundaries_are_1d,
            has_sorter,
            right,
        ).cast[out_dtype]()
        i += stride


@always_inline
def _searchsorted[
    dtype: DType, out_dtype: DType
](
    out_addr: Int,
    boundaries_addr: Int,
    values_addr: Int,
    sorter_addr: Int,
    num_values: Int,
    boundary_size: Int,
    values_per_batch: Int,
    boundaries_are_1d: Bool,
    has_sorter: Bool,
    right: Bool,
    ctx: DeviceContext,
) raises:
    var out = _make_ptr[out_dtype](out_addr).as_unsafe_any_origin()
    var boundaries = (
        _make_ptr[dtype](boundaries_addr).as_unsafe_any_origin().as_immutable()
    )
    var values = (
        _make_ptr[dtype](values_addr).as_unsafe_any_origin().as_immutable()
    )
    var sorter = (
        _make_ptr[DType.int64](sorter_addr)
        .as_unsafe_any_origin()
        .as_immutable()
    )

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(
            out,
            boundaries,
            values,
            sorter,
            boundary_size,
            values_per_batch,
            boundaries_are_1d,
            has_sorter,
            right,
        )
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            out[i] = _binary_search_position[dtype](
                boundaries,
                values,
                sorter,
                i,
                boundary_size,
                values_per_batch,
                boundaries_are_1d,
                has_sorter,
                right,
            ).cast[out_dtype]()

        _parallel_for[func](num_values, ctx)
        return

    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        _enqueue_cached[_binary_search_kernel[dtype, out_dtype]](
            ctx,
            String(t"binary_search_{dtype}_{out_dtype}"),
            _gs_blocks(num_values),
            1,
            1,
            GS_THREADS,
            out,
            boundaries,
            values,
            sorter,
            Int64(num_values),
            Int64(boundary_size),
            Int64(values_per_batch),
            Int64(boundaries_are_1d),
            Int64(has_sorter),
            Int64(right),
        )


def _searchsorted_go(
    out_obj: PyObjectPtr,
    boundaries_obj: PyObjectPtr,
    values_obj: PyObjectPtr,
    sorter_obj: PyObjectPtr,
    num_values_obj: PyObjectPtr,
    boundary_size_obj: PyObjectPtr,
    values_per_batch_obj: PyObjectPtr,
    boundaries_are_1d_obj: PyObjectPtr,
    has_sorter_obj: PyObjectPtr,
    right_obj: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    out_dtype_obj: PyObjectPtr,
    ctx_obj: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_dtype = _raw_dtype_int(out_dtype_obj)
    var handled = False
    comptime for dt in SEARCH_DTYPES:
        comptime if _dtype_arg_on[0, dt]() and _dtype_arg_on[1, dt]():
            if dtype == dt:
                comptime for odt in OUTPUT_DTYPES:
                    comptime if _dtype_out_on[0, odt]():
                        if out_dtype == odt:
                            _searchsorted[dt, odt](
                                _raw_int(out_obj),
                                _raw_int(boundaries_obj),
                                _raw_int(values_obj),
                                _raw_int(sorter_obj),
                                _raw_int(num_values_obj),
                                _raw_int(boundary_size_obj),
                                _raw_int(values_per_batch_obj),
                                Bool(_raw_int(boundaries_are_1d_obj)),
                                Bool(_raw_int(has_sorter_obj)),
                                Bool(_raw_int(right_obj)),
                                _raw_ctx(ctx_obj),
                            )
                            handled = True
    if not handled:
        raise Error(
            "unsupported dtype specialization for searchsorted: "
            + String(dtype)
            + " -> "
            + String(out_dtype)
        )


@export
def PyInit_searchsorted_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("searchsorted_ops")
        comptime if _op_on["Searchsorted"]():
            _register_call(
                b,
                _spec_dispatcher13[_searchsorted_go, "Searchsorted"],
                docstring=(
                    "(out, boundaries, values, sorter pointers; dynamic search "
                    "geometry, flags, dtypes, context)"
                ),
            )
        return b.finalize()
    except e:
        abort(t"failed to create searchsorted_ops python module: {e}")
