# Probe binary for tests/test_fa4_selfload_ptx_ordering.py.
#
# Never run: `mojo build --emit asm` only COMPILES this file, it does not
# execute `main()`, so no GPU is needed to produce the PTX sidecar this
# probe exists for (see AGENTS.md's cross-compilation section). Building
# it forces monomorphization of `fwd_fa4_selfload_kernel` at the same
# comptime parameters the production dense-causal-bhsd d64 entry points
# use, which is what makes the sidecar comparable to what actually ships.
#
# Build (never under the GPU lock -- this never touches a GPU):
#   <mojo> build tests/fa4_selfload_ptx_probe.mojo \
#       -I torch_mojo_backend/eager_flash_attention \
#       --emit asm --target-accelerator sm_90a -o <out>.s
from max.gpu.host import DeviceContext
from std.memory.unsafe_pointer import pointer_to_int

from fa4_fwd_selfload_launch import launch_fwd_fa4_selfload


def main() raises:
    var ctx = DeviceContext()
    var ctx_addr = pointer_to_int(ctx._handle)
    comptime batch: Int = 8
    comptime heads: Int = 12
    comptime seq: Int = 1024
    comptime head_dim: Int = 64
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
        Float32(0.125),
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
