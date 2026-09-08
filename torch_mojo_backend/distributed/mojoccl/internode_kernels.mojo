# The three device kernels the inter-node hop needs on top of
# collectives_kernels.mojo, which is untouched.
#
# None of them synchronizes with anything: stream order does it. Each runs
# after the host callback that put the network data in place (internode.mojo)
# and before the intra-node collective that consumes the result, so there is
# no flag protocol here and no peer pointer -- every address is inside this
# rank's own region or its own user buffers.

from std.atomic import Atomic, Ordering
from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx, grid_dim
from std.time import global_perf_counter_ns
from std.sys import size_of
from std.utils import StaticTuple
from max.gpu.host import DeviceContext, DeviceStream

from collectives_kernels import BLOCK, MAX_WORLD, _copy_bytes, _enqueue_cached

comptime _UNROLL = 4
comptime _MAX_BLOCKS = 432
# (nnodes - 1) inbox slots; 16 nodes x 8 GPUs is far past what this library
# is built for, and the array below is what makes it a compile-time bound.
comptime MAX_NODES = 16


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_internode_inbox_add_{dtype}")
def _inbox_add_kernel[
    dtype: DType, W: Int
](
    shard: Pointer[Scalar[dtype], MutAnyOrigin],
    inbox: Pointer[UInt8, MutAnyOrigin],
    count: Int64,
    slot_bytes: Int64,
    npeers_i: Int32,
):
    """`shard += sum of the npeers inbox slots`, elementwise.

    The node-local sum of this rank's shard is already in `shard` (that is
    what `reduce_scatter_stage` left there); each remote node's node-local
    sum of the SAME shard has landed in one inbox slot. Adding them makes
    `shard` the global sum, which `allgather_finish` then spreads.

    Accumulated in the wire dtype, like the rest of the hierarchical path:
    the remote sums arrived rounded already, so a wider accumulator here
    would buy nothing.
    """
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    var n = Int(count)
    var npeers = Int(npeers_i)
    var sb = Int(slot_bytes)
    var nv = n // W

    for v in range(tid, nv, stride):
        var acc = shard.unsafe_load[width=W, alignment=16](v * W)
        for j in range(npeers):
            var src = inbox.unsafe_offset(j * sb).unsafe_bitcast[Scalar[dtype]]()
            acc += src.unsafe_load[width=W, alignment=16](v * W)
        shard.unsafe_store[width=W, alignment=16](v * W, acc)

    for i in range(nv * W + tid, n, stride):
        var acc = shard[unsafe_offset=i]
        for j in range(npeers):
            var src = inbox.unsafe_offset(j * sb).unsafe_bitcast[Scalar[dtype]]()
            acc += src[unsafe_offset=i]
        shard[unsafe_offset=i] = acc


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_internode_proxy_request")
def _proxy_request_kernel(
    mailbox: Pointer[UInt64, MutAnyOrigin], seq: UInt64
):
    """Hand exchange `seq` to the progress thread.

    A release store into pinned host memory, so everything the stream did
    before this kernel -- the reduce-scatter that produced the shard the
    thread is about to send -- is visible to the CPU that acquires it.
    """
    if global_idx.x == 0:
        Atomic[DType.uint64].store[ordering = Ordering.RELEASE](mailbox, seq)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_internode_proxy_wait")
def _proxy_wait_kernel(
    mailbox: Pointer[UInt64, MutAnyOrigin],
    error_word: Pointer[UInt64, MutAnyOrigin],
    seq: UInt64,
    timeout_ns: UInt64,
):
    """Hold the stream until the progress thread reports exchange `seq` done.

    One thread spinning on pinned host memory. This is the whole reason the
    proxy exists: the same rendezvous through `cuLaunchHostFunc` cost about
    480 us per exchange on this cluster (measured, job 234035 -- 1 MiB
    allreduce 496 us against 24 us on one node), because the driver has to
    stop the stream, wake a thread and restart it. A spin kernel and a
    spinning CPU thread cost a launch each.

    On the deadline it writes the region's error word and gives up rather
    than hanging the stream forever; the add kernel then runs on stale
    inbox bytes, which `ncclCommGetAsyncError` reports.
    """
    if global_idx.x == 0:
        var t0 = global_perf_counter_ns()
        while (
            Atomic[DType.uint64].load[ordering = Ordering.ACQUIRE](mailbox)
            < seq
        ):
            if global_perf_counter_ns() - t0 > timeout_ns:
                Atomic[DType.uint64].store[ordering = Ordering.RELEASE](
                    error_word, UInt64(9) * 1_000_000
                )
                return


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_internode_copy_bytes")
def _copy_kernel(
    dst: Pointer[UInt8, MutAnyOrigin],
    src: Pointer[UInt8, MutAnyOrigin],
    nbytes: Int64,
):
    """Staging copy: a user buffer is not in the registered region, so
    broadcast and allgather have to move their payload in and out of it."""
    _copy_bytes[_UNROLL](
        dst,
        src,
        Int(nbytes),
        Int(global_idx.x),
        Int(grid_dim.x) * BLOCK,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_internode_place_blocks")
def _place_blocks_kernel(
    dst: Pointer[UInt8, MutAnyOrigin],
    src: Pointer[UInt8, MutAnyOrigin],
    dst_offsets: StaticTuple[Int64, MAX_WORLD],
    block_bytes: Int64,
    nblocks_i: Int32,
):
    """Scatter one node's allgather block into the output by global rank.

    A node's block holds its `local_world` contributions in local-rank order;
    the output wants them at their global ranks, which torchrun's numbering
    makes contiguous but which this library reads out of the bootstrap table
    instead of assuming. One launch per node beats `local_world` launches of
    a few bytes each.
    """
    var nb = Int(block_bytes)
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    for l in range(Int(nblocks_i)):
        _copy_bytes[_UNROLL](
            dst.unsafe_offset(Int(dst_offsets[l])),
            src.unsafe_offset(l * nb),
            nb,
            tid,
            stride,
        )


# ===-------------------------------------------------------------------=== #
# Launchers
# ===-------------------------------------------------------------------=== #


def _blocks_for(nbytes: Int) -> Int:
    return min(_MAX_BLOCKS, max(1, (nbytes // 16 + BLOCK - 1) // BLOCK))


def inbox_add[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    shard_ptr: Int,
    inbox_ptr: Int,
    count: Int,
    slot_bytes: Int,
    npeers: Int,
) raises:
    """Enqueue `shard += sum(inbox slots)` on `stream`."""
    if count <= 0 or npeers <= 0:
        return
    comptime W = 16 // size_of[dtype]()
    _enqueue_cached[_inbox_add_kernel[dtype, W]](
        ctx,
        stream,
        String(t"ib_add_{dtype}"),
        _blocks_for(count * size_of[dtype]()),
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=shard_ptr),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=inbox_ptr),
        Int64(count),
        Int64(slot_bytes),
        Int32(npeers),
    )


def copy_bytes(
    ctx: DeviceContext, stream: DeviceStream, dst: Int, src: Int, nbytes: Int
) raises:
    if nbytes <= 0:
        return
    _enqueue_cached[_copy_kernel](
        ctx,
        stream,
        "ib_copy",
        _blocks_for(nbytes),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=dst),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=src),
        Int64(nbytes),
    )


def place_blocks(
    ctx: DeviceContext,
    stream: DeviceStream,
    dst: Int,
    src: Int,
    dst_offsets: StaticTuple[Int64, MAX_WORLD],
    block_bytes: Int,
    nblocks: Int,
) raises:
    if block_bytes <= 0 or nblocks <= 0:
        return
    _enqueue_cached[_place_blocks_kernel](
        ctx,
        stream,
        "ib_place",
        _blocks_for(block_bytes),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=dst),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=src),
        dst_offsets,
        Int64(block_bytes),
        Int32(nblocks),
    )


def proxy_request(
    ctx: DeviceContext, stream: DeviceStream, mailbox: Int, seq: Int
) raises:
    _enqueue_cached[_proxy_request_kernel](
        ctx,
        stream,
        "ib_req",
        1,
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=mailbox),
        UInt64(seq),
    )


def proxy_wait(
    ctx: DeviceContext,
    stream: DeviceStream,
    mailbox: Int,
    error_word: Int,
    seq: Int,
    timeout_ns: Int,
) raises:
    _enqueue_cached[_proxy_wait_kernel](
        ctx,
        stream,
        "ib_wait",
        1,
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=mailbox),
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=error_word),
        UInt64(seq),
        UInt64(timeout_ns),
    )
