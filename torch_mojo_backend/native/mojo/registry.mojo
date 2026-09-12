"""Registration helper shared by every ops_*.mojo file."""
from std.ffi import external_call

from abi import OpFn, op_address

comptime Lib = OpaquePointer[MutUntrackedOrigin]


def impl[op: OpFn](lib: Lib, name: StaticString) raises:
    """Register `op` as the PrivateUse1 kernel of aten::<name> ("add.Tensor",
    "view", "fill_.Scalar", ...)."""
    var s = String(name)
    var rc = external_call["tmb_library_impl", Int32](
        lib, s.as_c_string_slice().unsafe_ptr(), op_address[op](), 0
    )
    if rc != 0:
        raise Error("registering aten::", name, " failed")
