"""Entry point of the native backend: `tmb_native_init` registers the device
hooks with the C++ shim and every aten op with torch's dispatcher.

Build: `mojo build backend.mojo --emit shared-lib -I native/mojo -I eager_kernels`
(native/__init__.py does it, cached like every other on-demand build).
"""
from std.ffi import c_char, external_call

from abi import set_shim_error
from device import hooks_table, init_backend
from kernels import init_loader
from ops_attention import register_attention
from ops_binary import register_binary
from ops_compare import register_compare
from ops_core import (
    op_as_strided,
    op_copy_from,
    op_empty_memory_format,
    op_empty_strided,
    op_fill_scalar_,
    op_local_scalar_dense,
    op_record_stream,
    op_reshape_alias,
    op_view,
    op_zero_,
)
from ops_data_movement import register_data_movement
from ops_factories import register_factories
from ops_foreach import register_foreach
from ops_matmul import register_matmul
from ops_nn import register_nn
from ops_reductions import register_reductions
from ops_unary import register_unary
from registry import Lib, impl


def _register_ops(lib: Lib) raises:
    # core (ops_core.mojo)
    impl[op_empty_memory_format](lib, "empty.memory_format")
    impl[op_empty_strided](lib, "empty_strided")
    impl[op_copy_from](lib, "_copy_from")
    impl[op_view](lib, "view")
    impl[op_view](lib, "_unsafe_view")
    impl[op_reshape_alias](lib, "_reshape_alias")
    impl[op_as_strided](lib, "as_strided")
    impl[op_local_scalar_dense](lib, "_local_scalar_dense")
    impl[op_fill_scalar_](lib, "fill_.Scalar")
    impl[op_zero_](lib, "zero_")
    impl[op_record_stream](lib, "record_stream")
    # one file per group; each group registers its own ops
    register_unary(lib)
    register_binary(lib)
    register_compare(lib)
    register_data_movement(lib)
    register_factories(lib)
    register_reductions(lib)
    register_matmul(lib)
    register_nn(lib)
    register_attention(lib)
    register_foreach(lib)


@export
def tmb_native_init(
    kernels_dir: Pointer[c_char, MutUntrackedOrigin],
    cache_dir: Pointer[c_char, MutUntrackedOrigin],
    mojo_exe: Pointer[c_char, MutUntrackedOrigin],
    toolchain: Pointer[c_char, MutUntrackedOrigin],
    trace: Int32,
) abi("C") -> Int32:
    """Returns the mojo device count, or -1 with the error in tmb_get_error."""
    try:
        var n = init_backend()
        init_loader(
            String(unsafe_from_utf8_ptr=kernels_dir.unsafe_bitcast[UInt8]()),
            String(unsafe_from_utf8_ptr=cache_dir.unsafe_bitcast[UInt8]()),
            String(unsafe_from_utf8_ptr=mojo_exe.unsafe_bitcast[UInt8]()),
            String(unsafe_from_utf8_ptr=toolchain.unsafe_bitcast[UInt8]()),
            trace != 0,
        )
        var table = hooks_table()
        if external_call["tmb_backend_register", Int32](table) != 0:
            raise Error("tmb_backend_register failed")
        var ns = String("aten")
        var key = String("PrivateUse1")
        var lib = external_call["tmb_library_new", Lib](
            ns.as_c_string_slice().unsafe_ptr(),
            key.as_c_string_slice().unsafe_ptr(),
        )
        if Int(lib) == 0:
            raise Error("tmb_library_new failed")
        _register_ops(lib)
        return Int32(n)
    except e:
        set_shim_error(String(e))
        return -1
