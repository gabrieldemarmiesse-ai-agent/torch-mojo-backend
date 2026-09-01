# ptxas static-shared-memory probe, built by `_ptxas_supports_big_static_smem`
# (eager_kernels/__init__.py). The kernel writes, barriers and reads back its
# `PROBE_SIZE`-byte static `.shared` buffer so no pass can drop it. The entry
# is never called; `@export` is what makes `mojo build` elaborate the kernel.
# It is not named `PyInit_...` so compare_kernel_asm.py does not pick it up.

from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from std.gpu import thread_idx
from std.memory import (
    AddressSpace,
    OpaquePointer,
    UnsafePointer,
    stack_allocation,
)
from std.os import abort
from std.sys.defines import get_defined_string

comptime _PROBE_SIZE_STR = get_defined_string["PROBE_SIZE", "49152"]()


@always_inline
def _probe_size() -> Int:
    # Any size other than the two in `_PTXAS_PROBE_SIZES` takes the control leg.
    comptime if _PROBE_SIZE_STR == "131072":
        return 131072
    else:
        return 49152


comptime _SIZE = _probe_size()


def _ptxas_probe_kernel(
    out_ptr: UnsafePointer[Scalar[DType.uint32], MutUntrackedOrigin]
):
    var tile = stack_allocation[
        _SIZE, DType.uint8, address_space=AddressSpace.SHARED
    ]()
    tile[0] = thread_idx.x.cast[DType.uint8]()
    barrier()
    out_ptr[0] = tile[0].cast[DType.uint32]()


@export
def _ptxas_probe_entry() abi("C"):
    try:
        var ctx = DeviceContext(
            OpaquePointer[MutUntrackedOrigin](unsafe_from_address=1)
        )
        ctx.enqueue_function[_ptxas_probe_kernel](
            UnsafePointer[Scalar[DType.uint32], MutUntrackedOrigin](
                unsafe_from_address=1
            ),
            grid_dim=(1,),
            block_dim=(1,),
        )
    except e:
        abort(t"unreachable: the ptxas probe entry is never executed ({e})")
