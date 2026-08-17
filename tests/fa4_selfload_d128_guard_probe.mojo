# Probe for tests/test_fa4_selfload_ptx_ordering.py::test_selfload_rejects_d128.
#
# Ported from agent A2's review artifact (d128_guard_v7.mojo,
# /scratch/fa4-fwd-harness-2c): instantiating the self-loading kernel at
# head_dim=128 must fail the BUILD with the comptime assert in
# fa4_fwd_selfload_kernel.mojo, not silently compute the wrong thing. The
# self-loading body assumes one 128-thread MMA warpgroup; d128 needs
# BM=BN=128 with two warpgroups and 224 KiB of smem, which is the
# phase-2b kernel's job (fa4_fwd_kernel.mojo), not this module's.
from max.gpu.host import DeviceContext
from std.memory.unsafe_pointer import pointer_to_int

from fa4_fwd_selfload_launch import launch_fwd_fa4_selfload


def main() raises:
    var ctx = DeviceContext()
    var ctx_addr = pointer_to_int(ctx._handle)
    comptime batch: Int = 1
    comptime heads: Int = 8
    comptime seq: Int = 512
    comptime head_dim: Int = 128
    comptime n: Int = batch * heads * seq * head_dim
    var q = ctx.enqueue_create_buffer[DType.bfloat16](n)
    var k = ctx.enqueue_create_buffer[DType.bfloat16](n)
    var v = ctx.enqueue_create_buffer[DType.bfloat16](n)
    var o = ctx.enqueue_create_buffer[DType.bfloat16](n)
    var lse = ctx.enqueue_create_buffer[DType.float32](batch * heads * seq)
    launch_fwd_fa4_selfload[
        DType.bfloat16,
        head_dim,
        False,
        True,
        1,
        0,
    ](
        batch,
        seq,
        heads,
        Float32(0.088),
        Int(q.unsafe_ptr()),
        Int(k.unsafe_ptr()),
        Int(v.unsafe_ptr()),
        Int(o.unsafe_ptr()),
        Int(lse.unsafe_ptr()),
        0,
        ctx_addr,
    )
    _ = q^
    _ = k^
    _ = v^
    _ = o^
    _ = lse^
