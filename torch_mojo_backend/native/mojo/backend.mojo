"""Entry point of the native backend: `tmb_native_init` registers the device
hooks with the C++ shim and every aten op with torch's dispatcher.

Build: `mojo build backend.mojo --emit shared-lib -I native/mojo -I eager_kernels`
(native/__init__.py does it, cached like every other on-demand build).
"""
from std.ffi import c_char, external_call

from abi import op_address, set_shim_error
from device import hooks_table, init_backend
from loader import Loader
from ops_core import (
    op_as_strided,
    op_copy_from,
    op_empty_memory_format,
    op_empty_strided,
    op_fill_scalar_,
    op_local_scalar_dense,
    op_reshape_alias,
    op_view,
    op_zero_,
)
from ops_binary import op_add_tensor, op_mul_tensor
from kernels import init_loader


def _impl(
    lib: OpaquePointer[MutUntrackedOrigin], name: StaticString, addr: Int
) raises:
    var s = String(name)
    var rc = external_call["tmb_library_impl", Int32](
        lib, s.as_c_string_slice().unsafe_ptr(), addr, 0
    )
    if rc != 0:
        raise Error("registering ", name, " failed")


def _register_ops(lib: OpaquePointer[MutUntrackedOrigin]) raises:
    _impl(lib, "empty.memory_format", op_address[op_empty_memory_format]())
    _impl(lib, "empty_strided", op_address[op_empty_strided]())
    _impl(lib, "_copy_from", op_address[op_copy_from]())
    _impl(lib, "view", op_address[op_view]())
    _impl(lib, "_unsafe_view", op_address[op_view]())
    _impl(lib, "_reshape_alias", op_address[op_reshape_alias]())
    _impl(lib, "as_strided", op_address[op_as_strided]())
    _impl(lib, "_local_scalar_dense", op_address[op_local_scalar_dense]())
    _impl(lib, "fill_.Scalar", op_address[op_fill_scalar_]())
    _impl(lib, "zero_", op_address[op_zero_]())
    _impl(lib, "add.Tensor", op_address[op_add_tensor]())
    _impl(lib, "mul.Tensor", op_address[op_mul_tensor]())


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
        var lib = external_call[
            "tmb_library_new", OpaquePointer[MutUntrackedOrigin]
        ](
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
