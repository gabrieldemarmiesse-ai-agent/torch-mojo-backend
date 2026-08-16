# ===----------------------------------------------------------------------=== #
# Thin eager-mode bridge for the on-device random generators of mojo_device.
#
# Kernel bodies live in `uniform_kernels.mojo`. This Python-visible module only
# unpacks the raw pointer/scalar ABI, rebuilds the full-width Philox seed and
# counter from their 32-bit halves, and enqueues work on the caller's
# DeviceContext. It performs no host reads and no synchronization, so a draw
# stays asynchronous: the generator state it needs was reserved on the host
# before the call (`_reserve_philox_state`).
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.python import PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.python.bindings import PythonModuleBuilder

from op_utils import (
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _spec_dispatcher10,
)
from uniform_kernels import enqueue_uniform

from variant_gates import _dtype_out_on, _op_on, _register_call


# The dtypes `aten::uniform_` is defined for, minus complex (unsupported
# device-wide): ATen dispatches it over the floating-point types plus the two
# 16-bit floats (`AT_DISPATCH_FLOATING_TYPES_AND2` in `uniform_impl_`).
comptime UNIFORM_DTYPES = [
    DType.float32,
    DType.bfloat16,
    DType.float16,
    DType.float64,
]


@always_inline
def _join_u64(lo: Int, hi: Int) -> UInt64:
    """Rejoin a 64-bit value split into 32-bit halves by the Python caller.

    Keeping every Python integer below 2**32 avoids `Py_ssize_t` overflow in
    the raw CPython bridge while preserving all 64 bits of seed and counter.
    """
    return UInt64(lo) | (UInt64(hi) << 32)


def _uniform_go(
    dst_ptr_obj: PyObjectPtr,
    from_obj: PyObjectPtr,
    to_obj: PyObjectPtr,
    numel_obj: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    seed_lo_obj: PyObjectPtr,
    seed_hi_obj: PyObjectPtr,
    offset_lo_obj: PyObjectPtr,
    offset_hi_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dst_addr = _raw_int(dst_ptr_obj)
    var from_value = _raw_f64(from_obj)
    var to_value = _raw_f64(to_obj)
    var size = _raw_int(numel_obj)
    var dtype = _raw_dtype_int(dtype_obj)
    var seed = _join_u64(_raw_int(seed_lo_obj), _raw_int(seed_hi_obj))
    var base_offset = _join_u64(
        _raw_int(offset_lo_obj), _raw_int(offset_hi_obj)
    )
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False

    comptime for dt in UNIFORM_DTYPES:
        comptime if _dtype_out_on[0, dt]():
            if dtype == dt:
                enqueue_uniform[dt](
                    dst_addr, from_value, to_value, size, seed, base_offset, ctx
                )
                handled = True
    if not handled:
        # A miss means Python selected the wrong immutable specialization.
        raise Error("unsupported dtype for on-device uniform_: ", dtype)


@export
def PyInit_random_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("random_ops")
        comptime if _op_on["UniformFill"]():
            _register_call(
                b,
                _spec_dispatcher10[_uniform_go, "UniformFill"],
                docstring=(
                    "(dst_ptr, from, to, numel, dtype, seed_lo, seed_hi,"
                    " offset_lo, offset_hi, context_ptr); in-place uniform"
                    " [from, to) fill of a contiguous buffer"
                ),
            )
        return b.finalize()
    except e:
        abort(t"failed to create random_ops python module: {e}")
