# ===----------------------------------------------------------------------=== #
# ptxas static-shared-memory capability probe.
#
# Not a kernel: `_ptxas_probe_kernel` exists only so `mojo build` elaborates a
# GPU entry with a *static* `.shared` buffer of `PROBE_SIZE` bytes (a `-D`
# define, see eager_kernels/__init__.py) the same way a real gated GEMM
# kernel would. Its thread-index write, barrier, and readback into a
# caller-visible pointer keep the buffer live so no optimizer pass can prove
# it dead and drop the `.shared` declaration before ptxas ever sees it.
#
# `_ptxas_probe_entry` is `@export`ed and never called (the address it hands
# `DeviceContext`/`enqueue_function` is a placeholder, not a real buffer):
# exporting it is what makes `mojo build` fully elaborate everything it
# reaches, the same mechanism `scripts/compare_kernel_asm.py` relies on for
# real entry modules. It deliberately is NOT named `PyInit_...` -- that name
# is how `compare_kernel_asm.py` finds real Python-extension entry points,
# and this file is neither loaded as an extension nor built by that script.
#
# Built with `--emit shared-lib` and no `--target-accelerator` override, the
# same way `eager_kernels._extension_cmd` builds real kernels: passing an
# explicit accelerator makes `mojo build` defer to PTX-only embedding and
# skip assembling through the toolchain entirely (confirmed empirically --
# see the probe's Python-side docstring), so this file always builds for
# whatever accelerator is actually attached, exactly like a real kernel does.
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from std.gpu import thread_idx
from std.memory import AddressSpace, OpaquePointer, UnsafePointer, stack_allocation
from std.os import abort
from std.sys.defines import get_defined_string

comptime _PROBE_SIZE_STR = get_defined_string["PROBE_SIZE", "49152"]()


@always_inline
def _probe_size() -> Int:
    # Only the two sizes eager_kernels._PTXAS_PROBE_SIZES ever sends are
    # meaningful; anything else silently takes the control (always-must-pass)
    # leg rather than failing this comptime binding.
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
