# The inter-node hop: GPUDirect RDMA over libibverbs, no vendor collective
# library. One RC queue pair per remote node, to the rank holding the SAME
# local_rank there -- so the 8 ranks of a node drive 8 independent NICs and
# each rank only ever exchanges its own 1/local_world shard.
#
# Ordering. The GPU cannot post verbs and the NIC cannot wait on a kernel,
# so a CPU thread stands between them and the stream is what sequences it:
#
#     [reduce_scatter_stage]  my shard is final in my stage_out
#     [proxy_request]         one thread releases the exchange counter into
#                             a pinned mailbox
#       ~ progress thread ~   post one RDMA_WRITE_WITH_IMM per peer, poll
#                             the CQ until every peer's shard has landed in
#                             my inbox, flush (below), release `done`
#     [proxy_wait]            one thread spins until `done` catches up
#     [inbox_add]             shard += the peers' shards
#     [allgather_finish]      spread the global sum
#
# The obvious alternative -- one `cuLaunchHostFunc` doing all of it inline
# -- was written first, shipped, and measured: it costs about 480 us per
# exchange on this cluster, because the driver has to stop the stream, wake
# a thread and restart it, and that delay does NOT cancel between the two
# nodes (each side ends up measuring the other's dispatch jitter; both
# reported ~390 us of "waiting for the peer" on a transfer worth 3 us). It
# survives behind `MOJOCCL_IB_PROXY=0`: same exchange body, one fewer core
# burned, several hundred microseconds slower.
#
# The GPUDirect flush. Seeing the RDMA_WRITE_WITH_IMM completion does NOT
# mean the payload is visible in GPU memory: the completion lands in host
# memory and the payload in the GPU's BAR, two different PCIe destinations
# with no ordering between them. A read from the GPU BAR flushes the posted
# writes ahead of it, so an exchange finishes with a 4-byte RDMA_READ of
# the inbox over a self-connected QP -- exactly NCCL's `gpuFlush` QP
# (nccl:src/transport/net_ib/p2p.cc:589-602). Measured 1.9 us.
#
# Flow control: EXPLICIT CREDITS. The inbox is carved once into `nslots`
# fixed slot groups and exchange e lands in group `e % nslots`, so peer B
# writes the same bytes at e and e+nslots. What makes reuse safe is a credit,
# not stream order: after my consumer kernel has read group g, my proxy
# RDMA-writes every peer a cumulative "I have consumed through exchange c"
# counter, and a peer may post exchange e only once every receiver has
# released the group e will land in (`c >= e - nslots`). That is NCCL's
# head/tail pair in miniature (nccl:src/transport/net.cc, the
# recvNetHead/step counters; nccl:src/device/prims_simple.h for the device
# side of the same idea).
#
# It replaces an earlier double-buffer-by-parity argument that derived
# reuse safety from stream order -- "B cannot reach e+2 before receiving my
# e+1 data, which I send only after my own add kernel for e". That chain
# holds only when exactly one exchange is in flight; the pipelined schedule
# (mojoccl.mojo `_do_allreduce`) issues the reduce-scatter of later chunks
# before the add of earlier ones and breaks it. The credit is the
# replacement proof, and it is a proof rather than a timing margin.
#
# Two things survive from the old argument unchanged. Every exchange must
# still be all-to-all, because an arrival tally of `nrecv` is what completes
# one -- a rank with nothing to contribute sends EMPTY_SHARD_BYTES
# (mojoccl.mojo). And every slot group must be a FIXED byte range for every
# exchange alike, never sized from the message in flight (`_inbox_base` in
# mojoccl.mojo carries the case that broke). The immediate carries the
# exchange counter and a credit bit, so an arrival is tallied against its own
# exchange and a credit is never mistaken for data.
#
# `credit_upto` is how the host tells the engine what has been consumed
# without a further kernel: it is the number of consumer kernels already
# enqueued on the stream ahead of this exchange's request kernel. When the
# proxy observes the request, that kernel has run, so every kernel enqueued
# before it has completed -- the same stream-order argument that makes the
# shard final at that point.
#
# The inbox lives in the region's own network area, never aliased onto the
# intra-node staging: a peer node writes it as soon as ITS reduce-scatter is
# done, which is ordered against neither mine nor a local peer still reading
# the previous generation (see ncclCommInitRank).

from std.ffi import OwnedDLHandle, external_call
from std.memory.alloc import unsafe_alloc
from std.sys import size_of
from std.os import getenv
from std.time import perf_counter_ns, sleep
from max.gpu.host import DeviceContext, DeviceStream

from std.atomic import Atomic, Ordering

from driver import (
    alloc_host,
    device_pci_bus_id,
    free_host,
    host_device_ptr,
    launch_host_func,
    open_driver,
)
from internode_kernels import proxy_request, proxy_wait
from ibverbs import (
    IbPort,
    IBV_ACCESS_LOCAL_WRITE,
    IBV_ACCESS_REMOTE_READ,
    IBV_ACCESS_REMOTE_WRITE,
    IBV_WC_RDMA_READ,
    IBV_WC_RDMA_WRITE,
    IBV_WC_RECV_RDMA_WITH_IMM,
    IBV_WC_SUCCESS,
    MR_LKEY,
    MR_RKEY,
    P8,
    SZ_RECV_WR,
    SZ_SEND_WR,
    SZ_SGE,
    SZ_WC,
    WC_IMM_DATA,
    WC_OPCODE,
    WC_QP_NUM,
    WC_STATUS,
    WC_VENDOR_ERR,
    WC_WR_ID,
    Ibv,
    alloc_bytes,
    be32,
    build_read_wr,
    build_recv_wr,
    build_write_wr,
    create_rc_qp,
    ld32,
    ld64,
    ldu32,
    list_ib_ports,
    poll_cq,
    post_recv,
    post_send,
    qp_number,
    qp_to_init,
    qp_to_rtr,
    qp_to_rts,
)

# Recv WRs kept posted per peer QP. Each RDMA_WRITE_WITH_IMM consumes one --
# data and credits alike; the engine reposts every one it consumes, so the
# depth only has to cover the burst a peer can produce while this rank is
# elsewhere: `nslots` data messages plus `nslots` credits, times a wide
# margin.
comptime RECV_DEPTH = 64
comptime CQ_SIZE = 1024
comptime SEND_WR_DEPTH = 64
comptime WORK_SLOTS = 512
comptime MAX_NODES = 16
comptime DEFAULT_IB_TIMEOUT_S: Float64 = 60.0
# ncclCommAbort's bound on waiting for the progress thread: long enough for
# it to notice MB_STOP (at most one idle-backoff quantum plus one engine
# step -- `ib_drive` never blocks), short enough that abort's documented
# "don't wait" contract still holds even if the thread is wedged somewhere
# this bound does not anticipate.
comptime IB_ABORT_JOIN_TIMEOUT_S: Float64 = 2.0
# Default idle-wait quantum for the progress thread between exchanges (see
# `_proxy_main`). An exchange takes ~280 us at the DDP bucket, so a few tens
# of us of wake-up latency between exchanges is cheap; measured end to end
# (job 234072, 2x8 H100) an unconditional hot spin here cost nanoGPT DDP
# ~20% of its steady-state tok/s against real NCCL, competing for a core/SMT
# sibling with the ~760-aten-op-per-step host dispatch of the training loop.
comptime DEFAULT_IB_PROXY_IDLE_US: Int = 20
# Bytes of the inbox read back by the flush; any read of the destination
# device flushes the writes ahead of it, the size is irrelevant.
comptime FLUSH_BYTES = 4

# Largest inbox slot-group count the engine will accept; bounds the arrival
# tally and, through it, how many exchanges may be outstanding at once.
comptime PIPE_MAX_SLOTS = 16
# Immediate layout: bit 31 marks a credit, bits 0..30 carry the exchange
# counter. 2^31 exchanges is ~10^4 DDP training runs, and the counter never
# wraps within one communicator.
comptime IMM_CREDIT_BIT: UInt32 = 0x8000_0000
comptime IMM_SEQ_MASK: UInt32 = 0x7FFF_FFFF
# Credit landing pad: one 64-byte line per sender in the region's credit
# area, so two peers' credits never share a cache line. The bytes are never
# read -- the immediate is the message -- but a real address is needed
# because a zero-length RDMA write is not worth relying on across HCAs.
comptime CREDIT_SLOT_BYTES = 64
comptime CREDIT_AREA_BYTES = 4096
comptime CREDIT_PAYLOAD_BYTES = 4

# The proxy mailbox: two 64-bit words the GPU and the progress thread
# pass the exchange counter through, plus a stop word the host sets at
# teardown. A cache line apart so the GPU's writes to REQUEST never
# invalidate the line the CPU is writing DONE into.
comptime MB_REQUEST = 0
comptime MB_DONE = 64
comptime MB_STOP = 128
comptime MB_BYTES = 192


struct IbPeer(Copyable, Movable):
    var node: Int
    var qp: Int
    var qpn: UInt32
    var remote_base: Int
    var remote_rkey: UInt32

    def __init__(
        out self,
        node: Int,
        qp: Int,
        qpn: UInt32,
        remote_base: Int,
        remote_rkey: UInt32,
    ):
        self.node = node
        self.qp = qp
        self.qpn = qpn
        self.remote_base = remote_base
        self.remote_rkey = remote_rkey


struct IbWork(Copyable, Movable):
    """One inter-node exchange, described by the host for the engine.

    Lives in a ring inside `IbState` indexed by `(seq-1) % WORK_SLOTS`: the
    counter is dense and starts at 1, so a slot is implied by the sequence
    number and no queue is needed. A slot is refilled only once the engine
    has set `status` away from 0.
    """

    var state: Int
    var send_addr: Int
    var send_bytes: Int
    var inbox_base: Int  # region offset of this exchange's inbox slot group
    var slot_bytes: Int
    var do_send: Int
    var nrecv: Int
    var flush_addr: Int
    var seq: Int
    var status: Int  # 0 running, 1 done, 2 failed
    # Highest exchange whose consumer kernel is already enqueued ahead of
    # this one's request on the stream -- the credit the engine publishes
    # when it picks this exchange up. See the header's flow-control note.
    var credit_upto: Int
    # 1 once this exchange's data write is posted to every peer. The exchange
    # is not done until each of those sends has completed on ITS queue pair
    # (`IbState.send_done`), or the consumer kernel could overwrite a buffer
    # the NIC is still reading for a slower peer.
    var sent: Int
    var t0: Int  # perf_counter_ns when it was posted, for the trace

    def __init__(out self):
        self.state = 0
        self.send_addr = 0
        self.send_bytes = 0
        self.inbox_base = 0
        self.slot_bytes = 0
        self.do_send = 0
        self.nrecv = 0
        self.flush_addr = 0
        self.seq = 0
        self.status = 1
        self.credit_upto = 0
        self.sent = 0
        self.t0 = 0


struct IbState(Movable):
    """Everything the inter-node hop owns, per communicator."""

    var ibv: Ibv
    var hca: String
    var ctx: Int
    var port: Int
    var pd: Int
    var mr: Int
    var lkey: UInt32
    var rkey: UInt32
    var cq: Int
    var peers: List[IbPeer]
    var flush_qp: Int
    var flush_mr: Int
    var flush_host: Int
    var flush_lkey: UInt32
    var region: Int
    var my_node: Int
    var nnodes: Int
    var exchanges: Int
    var error: Int
    var timeout_ns: Int
    # --- the progress engine (see `ib_drive`) ---
    var nslots: Int  # inbox slot groups; exchange e lands in `e % nslots`
    var credit_off: Int  # region offset of the credit landing pad
    var request_seq: Int  # highest exchange handed to the engine
    var posted_seq: Int  # highest exchange whose data writes are posted
    var done_seq: Int  # highest exchange fully arrived, sent and flushed
    var flush_seq: Int  # exchange whose flush read is outstanding (0: none)
    var flush_done: Int  # flush-read completions seen and not yet consumed
    var flush_t0: Int
    # Per peer, the highest exchange whose data write has completed on that
    # peer's queue pair (the wr_id of the completion is the exchange number).
    # Per QP, not one global count: RC completes in order on ONE queue pair
    # only, so with three or more nodes a completion for e+1 towards a fast
    # peer can land before the completion for e towards a slow one, and a
    # global tally would call e's sends finished while the NIC still reads
    # e's source for the slow peer.
    var send_done: List[Int]
    var credit_sent: Int  # highest credit published to the peers
    var tally: List[Int]  # arrivals, indexed by `seq % nslots`
    var credit_recv: List[Int]  # per peer, highest credit it published
    var last_progress_ns: Int
    var stall_seq: Int  # exchange the last credit stall was counted against
    var n_credit_stalls: Int
    # Host-side: highest exchange whose consumer kernel has been ENQUEUED.
    # Snapshotted into each work item as `credit_upto` (see
    # `ib_note_consumed`); never touched by the engine thread.
    var consumed_enqueued: Int
    # Scratch buffers and the work ring are held as raw addresses: a
    # `Pointer[..., MutAnyOrigin]` cannot be a struct field, and these
    # outlive every borrow anyway (allocated once, freed never -- a few
    # hundred bytes per communicator).
    var wr: Int
    var sge: Int
    var rwr: Int
    var bad: Int
    var wc: Int
    var ts: Int  # struct timespec scratch for the idle nanosleep
    var works: Int
    var work_next: Int
    var mailbox: Int  # pinned host address
    var mailbox_dev: Int  # the same memory as a kernel addresses it
    var error_word: Int  # region + error_offset, for the wait kernel
    var proxy: Bool
    var thread_id: Int
    var trace: Bool
    var t_post_ns: Int
    var t_wait_ns: Int
    var t_flush_ns: Int
    var n_exchanges: Int

    def __init__(
        out self,
        var ibv: Ibv,
        region: Int,
        my_node: Int,
        nnodes: Int,
        nslots: Int,
        credit_off: Int,
    ):
        self.ibv = ibv^
        self.hca = String("")
        self.ctx = 0
        self.port = 0
        self.pd = 0
        self.mr = 0
        self.lkey = 0
        self.rkey = 0
        self.cq = 0
        self.peers = List[IbPeer]()
        self.flush_qp = 0
        self.flush_mr = 0
        self.flush_host = 0
        self.flush_lkey = 0
        self.region = region
        self.my_node = my_node
        self.nnodes = nnodes
        self.exchanges = 0
        self.error = 0
        self.timeout_ns = Int(DEFAULT_IB_TIMEOUT_S * 1.0e9)
        self.nslots = nslots
        self.credit_off = credit_off
        self.request_seq = 0
        self.posted_seq = 0
        self.done_seq = 0
        self.flush_seq = 0
        self.flush_done = 0
        self.flush_t0 = 0
        self.send_done = List[Int]()
        self.credit_sent = 0
        self.tally = List[Int]()
        for _ in range(nslots):
            self.tally.append(0)
        self.credit_recv = List[Int]()
        self.last_progress_ns = 0
        self.stall_seq = 0
        self.n_credit_stalls = 0
        self.consumed_enqueued = 0
        self.wr = Int(alloc_bytes(SZ_SEND_WR))
        self.sge = Int(alloc_bytes(SZ_SGE))
        self.rwr = Int(alloc_bytes(SZ_RECV_WR))
        self.bad = Int(alloc_bytes(16))
        self.wc = Int(alloc_bytes(SZ_WC * 16))
        self.ts = Int(alloc_bytes(16))
        self.works = Int(unsafe_alloc[IbWork](WORK_SLOTS))
        var wp = Pointer[IbWork, MutAnyOrigin](unsafe_from_address=self.works)
        for i in range(WORK_SLOTS):
            wp[unsafe_offset=i] = IbWork()
        self.work_next = 0
        self.mailbox = 0
        self.mailbox_dev = 0
        self.error_word = region
        self.proxy = getenv("MOJOCCL_IB_PROXY", "1") != "0"
        self.thread_id = 0
        self.trace = getenv("MOJOCCL_IB_TRACE", "0") != "0"
        self.t_post_ns = 0
        self.t_wait_ns = 0
        self.t_flush_ns = 0
        self.n_exchanges = 0


@always_inline
def _b(addr: Int) -> P8:
    return P8(unsafe_from_address=addr)


@always_inline
def _st(ib: Int) -> Pointer[IbState, MutAnyOrigin]:
    return Pointer[IbState, MutAnyOrigin](unsafe_from_address=ib)


# `IbState.error` and `IbWork.status` are written by the proxy thread (or the
# `MOJOCCL_IB_PROXY=0` callback thread) inside `ib_drive` and read by
# the calling thread -- `ib_error` from the torch-facing calling thread,
# `ib_enqueue`'s ring-reuse check from the same -- with no other
# synchronization between the two. Every access goes through these two
# helpers rather than a plain field read/write so that relationship is a
# real release/acquire pair, not two threads racing a plain `Int`.
@always_inline
def _load_atomic_i(p: Pointer[Int, MutAnyOrigin]) -> Int:
    return Int(
        Atomic[DType.int64].load[ordering = Ordering.ACQUIRE](
            p.unsafe_bitcast[Int64]()
        )
    )


@always_inline
def _store_atomic_i(p: Pointer[Int, MutAnyOrigin], v: Int):
    Atomic[DType.int64].store[ordering = Ordering.RELEASE](
        p.unsafe_bitcast[Int64](), Int64(v)
    )


@always_inline
def _err_ptr(mut st: IbState) -> Pointer[Int, MutAnyOrigin]:
    return Pointer(to=st.error).unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _status_ptr(mut w: IbWork) -> Pointer[Int, MutAnyOrigin]:
    return Pointer(to=w.status).unsafe_origin_cast[MutAnyOrigin]()


# ===-------------------------------------------------------------------=== #
# The progress engine
# ===-------------------------------------------------------------------=== #
#
# One non-blocking step function, `ib_drive`, shared by all three drivers:
# the progress thread, the `MOJOCCL_IB_PROXY=0` stream callback and the
# GPU-free self-tests. They differ only in who advances `request_seq` (the
# mailbox, the callback, the calling thread) and who reads `done_seq`.
#
# The engine keeps several exchanges in flight. Per step it publishes any
# credit the next exchange carries, posts that exchange if flow control
# allows, drains the completion queue, and retires exchanges in sequence
# order. Retiring is strictly in order even though arrivals are not: RC
# ordering is per queue pair, so peer B's message for e+1 can overtake peer
# C's for e, and `done_seq` is what the GPU waits on.


@always_inline
def _work(st: IbState, seq: Int) -> Pointer[IbWork, MutAnyOrigin]:
    """The ring slot exchange `seq` lives in."""
    return Pointer[IbWork, MutAnyOrigin](
        unsafe_from_address=st.works
        + ((seq - 1) % WORK_SLOTS) * size_of[IbWork]()
    )


@always_inline
def _sender_slot(st: IbState, peer_node: Int) -> Int:
    """Where MY message sits among the receiver's per-sender slots: senders
    are indexed by node, compacted past the receiver's own node."""
    return st.my_node if st.my_node < peer_node else st.my_node - 1


def _release_on_error(mut st: IbState):
    """An exchange failed: release everything outstanding rather than leave
    the GPU's spin kernels (or a host waiter) hanging past the point where
    `ncclCommGetAsyncError` could report it."""
    while st.done_seq < st.request_seq:
        st.done_seq += 1
        ref w = _work(st, st.done_seq)[]
        _store_atomic_i(_status_ptr(w), 2)
    st.flush_seq = 0


def _send_credits(mut st: IbState, upto: Int) -> Bool:
    """Publish "I have consumed through exchange `upto`" to every peer.

    Credits are cumulative, so only the newest is ever on the wire: one
    unsignaled 4-byte RDMA_WRITE_WITH_IMM per peer, the immediate carrying
    the credit bit and the number. Unsignaled because a completion here
    would be indistinguishable from a data send, and the data sends -- which
    ARE counted, `IbWork.sends_cum` -- are what reclaims the send queue.
    A failed credit still raises a completion with a bad status.
    """
    if upto <= st.credit_sent:
        return False
    for i in range(len(st.peers)):
        ref p = st.peers[i]
        build_write_wr(
            _b(st.wr),
            _b(st.sge),
            upto,
            st.flush_host,
            st.flush_lkey,
            CREDIT_PAYLOAD_BYTES,
            p.remote_base
            + st.credit_off
            + _sender_slot(st, p.node) * CREDIT_SLOT_BYTES,
            p.remote_rkey,
            IMM_CREDIT_BIT | (UInt32(upto) & IMM_SEQ_MASK),
            True,
            False,
        )
        if post_send(p.qp, _b(st.wr), _b(st.bad)) != 0:
            _store_atomic_i(_err_ptr(st), 8)
            return False
    st.credit_sent = upto
    return True


@always_inline
def _can_post(st: IbState, seq: Int) -> Bool:
    """Flow control: exchange `seq` lands in slot group `seq % nslots`, whose
    previous occupant was `seq - nslots`, so every peer must have released
    that one first."""
    var need = seq - st.nslots
    if need <= 0:
        return True
    for i in range(len(st.credit_recv)):
        if st.credit_recv[i] < need:
            return False
    return True


def _post_data(mut st: IbState, mut w: IbWork) -> Bool:
    """Post this exchange's payload to every peer, one
    RDMA_WRITE_WITH_IMM each."""
    if w.do_send != 0 and w.send_bytes > 0:
        for i in range(len(st.peers)):
            ref p = st.peers[i]
            build_write_wr(
                _b(st.wr),
                _b(st.sge),
                w.seq,
                w.send_addr,
                st.lkey,
                w.send_bytes,
                p.remote_base
                + w.inbox_base
                + _sender_slot(st, p.node) * w.slot_bytes,
                p.remote_rkey,
                UInt32(w.seq) & IMM_SEQ_MASK,
                True,
                True,
            )
            if post_send(p.qp, _b(st.wr), _b(st.bad)) != 0:
                _store_atomic_i(_err_ptr(st), 1)
                return False
        w.sent = 1
    return True


@always_inline
def _peer_index(st: IbState, qpn: UInt32) -> Int:
    """Index into `st.peers` of the queue pair a completion came from, or -1
    (the flush QP, whose completions are classified by opcode instead)."""
    for k in range(len(st.peers)):
        if st.peers[k].qpn == qpn:
            return k
    return -1


@always_inline
def _sends_done(st: IbState, mut w: IbWork) -> Bool:
    """Every peer's queue pair has completed this exchange's data write.
    Completions on one RC queue pair are in order, so `send_done[i] >= seq`
    covers every earlier send on that pair too."""
    if w.sent == 0:
        return True
    for i in range(len(st.send_done)):
        if st.send_done[i] < w.seq:
            return False
    return True


def _consume_wc(mut st: IbState, c: P8) -> Int:
    """Account for one completion; -1 for a failed one (error recorded).

    Every completion is classified here, never "the one this exchange is
    waiting for": an arrival for a later exchange can land at any time, and
    a loop that dropped it would lose a tally somebody is waiting on and a
    receive WR nobody reposts.
    """
    if Int32(ld32(c, WC_STATUS)) != IBV_WC_SUCCESS:
        _store_atomic_i(
            _err_ptr(st),
            1000 + ld32(c, WC_STATUS) * 1000 + ld32(c, WC_VENDOR_ERR),
        )
        return -1
    var op = Int32(ld32(c, WC_OPCODE))
    var pi = _peer_index(st, ldu32(c, WC_QP_NUM))
    if op == IBV_WC_RECV_RDMA_WITH_IMM:
        var imm = be32(ldu32(c, WC_IMM_DATA))
        if pi >= 0:
            build_recv_wr(_b(st.rwr), 0)
            if post_recv(st.peers[pi].qp, _b(st.rwr), _b(st.bad)) != 0:
                # A receive slot lost here is a later RNR the peer retries
                # forever (IB_RNR_RETRY = 7), i.e. a silent hang; latch it.
                _store_atomic_i(_err_ptr(st), 5)
                return -1
        var seq = Int(imm & IMM_SEQ_MASK)
        if (imm & IMM_CREDIT_BIT) != 0:
            if pi >= 0 and st.credit_recv[pi] < seq:
                st.credit_recv[pi] = seq
        else:
            st.tally[seq % st.nslots] += 1
        return 0
    if op == IBV_WC_RDMA_WRITE:
        # Only data writes are signaled (credits are not), and their wr_id
        # is the exchange number, dense and increasing per queue pair.
        var seq = ld64(c, WC_WR_ID)
        if pi >= 0 and st.send_done[pi] < seq:
            st.send_done[pi] = seq
        return 0
    if op == IBV_WC_RDMA_READ:
        st.flush_done += 1
        return 0
    return 0


def _advance(mut st: IbState) -> Bool:
    """Retire every exchange that is complete, in sequence order.

    An exchange is complete when all `nrecv` peers' messages for it have
    arrived, its own send to EVERY peer has completed (the NIC is done
    reading the buffer the consumer kernel and the next reduce-scatter will
    overwrite) and its GPUDirect flush read has come back.
    """
    var moved = False
    while True:
        if st.flush_seq != 0:
            if st.flush_done <= 0:
                break
            st.flush_done -= 1
            st.t_flush_ns += perf_counter_ns() - st.flush_t0
            st.done_seq = st.flush_seq
            st.flush_seq = 0
            _retire(st, st.done_seq)
            moved = True
            continue
        if st.done_seq >= st.posted_seq:
            break
        var e = st.done_seq + 1
        ref w = _work(st, e)[]
        var idx = e % st.nslots
        if st.tally[idx] < w.nrecv or not _sends_done(st, w):
            break
        st.tally[idx] -= w.nrecv
        if w.nrecv > 0 and w.flush_addr != 0:
            # Seeing the arrivals does NOT mean the payload is visible in GPU
            # memory: the completion lands in host memory and the payload in
            # the GPU's BAR. A read of the destination flushes the writes
            # ahead of it -- NCCL's gpuFlush QP,
            # nccl:src/transport/net_ib/p2p.cc:589-602.
            build_read_wr(
                _b(st.wr),
                _b(st.sge),
                e,
                st.flush_host,
                st.flush_lkey,
                FLUSH_BYTES,
                w.flush_addr,
                st.rkey,
            )
            if post_send(st.flush_qp, _b(st.wr), _b(st.bad)) != 0:
                _store_atomic_i(_err_ptr(st), 4)
                return moved
            st.flush_seq = e
            st.flush_t0 = perf_counter_ns()
            moved = True
            continue
        st.done_seq = e
        _retire(st, e)
        moved = True
    return moved


def _retire(mut st: IbState, seq: Int):
    ref w = _work(st, seq)[]
    st.t_wait_ns += perf_counter_ns() - w.t0
    st.n_exchanges += 1
    _store_atomic_i(_status_ptr(w), 1)


def ib_drive(mut st: IbState) -> Bool:
    """One non-blocking step of the progress engine; True if anything moved.

    Order matters: credits go out BEFORE this step's own flow-control check,
    or two ranks that arrive at the same exchange together would each wait
    for a credit the other is holding back.
    """
    if _load_atomic_i(_err_ptr(st)) != 0:
        _release_on_error(st)
        return False
    var moved = False
    if st.posted_seq < st.request_seq:
        var e = st.posted_seq + 1
        ref w = _work(st, e)[]
        if _send_credits(st, w.credit_upto):
            moved = True
        if _can_post(st, e):
            var t0 = perf_counter_ns()
            if not _post_data(st, w):
                _release_on_error(st)
                return False
            w.t0 = t0
            st.posted_seq = e
            st.t_post_ns += perf_counter_ns() - t0
            moved = True
        elif st.stall_seq != e:
            # Counted once per exchange, not once per spin: a nonzero number
            # in the trace means flow control, not the network, held a chunk
            # back, which is the knob INBOX_SLOTS turns.
            st.stall_seq = e
            st.n_credit_stalls += 1
    var n = poll_cq(st.cq, 16, _b(st.wc))
    if n < 0:
        _store_atomic_i(_err_ptr(st), 2)
        _release_on_error(st)
        return False
    for i in range(Int(n)):
        if _consume_wc(st, P8(unsafe_from_address=st.wc + i * SZ_WC)) < 0:
            _release_on_error(st)
            return False
        moved = True
    if _advance(st):
        moved = True
    if moved:
        st.last_progress_ns = perf_counter_ns()
    elif st.request_seq > st.done_seq:
        # Nothing outstanding can move and nothing has moved for a whole
        # timeout: a peer is gone, or a credit was lost.
        if perf_counter_ns() - st.last_progress_ns > st.timeout_ns:
            _store_atomic_i(_err_ptr(st), 3)
            _release_on_error(st)
    return moved


def _ib_progress(user: OpaquePointer[MutAnyOrigin]) abi("C"):
    """`cuLaunchHostFunc` entry point -- the MOJOCCL_IB_PROXY=0 path.

    Runs on a driver-owned thread with the stream stalled behind it, so it
    must never call the CUDA/HIP driver, and it cannot pipeline: it drives
    the engine until its own exchange is done. Kept as a fallback, and as
    the thing the proxy thread is measured against: on this cluster the
    driver's stop-the-stream / wake-a-thread / restart round trip costs
    about 480 us per exchange (job 234035: a 1 MiB two-node allreduce took
    496 us against 24 us on one node, and the transfer in it is 3 us).
    """
    ref w = Pointer[IbWork, MutAnyOrigin](unsafe_from_address=Int(user))[]
    ref st = _st(w.state)[]
    _drive_until(st, w.seq)


def _drive_until(mut st: IbState, seq: Int):
    """Run the engine until exchange `seq` is retired (or the engine fails).
    The stall deadline inside `ib_drive` is what ends this if a peer never
    answers."""
    if seq > st.request_seq:
        st.request_seq = seq
        st.last_progress_ns = perf_counter_ns()
    while st.done_seq < seq:
        if _load_atomic_i(_err_ptr(st)) != 0:
            _release_on_error(st)
            return
        _ = ib_drive(st)


@always_inline
def _mb(st: IbState, off: Int) -> Pointer[UInt64, MutAnyOrigin]:
    return Pointer[UInt64, MutAnyOrigin](unsafe_from_address=st.mailbox + off)


def _proxy_main(arg: OpaquePointer[MutAnyOrigin]) abi("C"):
    """The progress thread: `ib_drive` in a loop, with the mailbox on both
    ends.

    `MB_REQUEST` is the highest exchange the stream has released (a
    one-thread kernel's release store); `MB_DONE` is the highest one
    retired, which the matching spin kernel waits for. Several exchanges may
    be in flight between the two.

    Hard-spins like NCCL's proxy only WHILE SOMETHING IS OUTSTANDING --
    otherwise it backs off (`sched_yield` once, then a short `nanosleep`)
    instead of burning a full core on a mailbox word that is not going to
    change for a while. A training step's host-side dispatch (hundreds of
    aten launches on the Python main thread) shares this core's SMT sibling,
    and measured end to end (job 234072, 2x8 H100) an unconditional hot spin
    here cost nanoGPT DDP ~20% of its steady-state tok/s against real NCCL.
    `MOJOCCL_IB_PROXY_IDLE_US` tunes the backoff quantum.
    """
    ref st = _st(Int(arg))[]
    var idle_ns = _proxy_idle_ns()
    var published = 0
    st.last_progress_ns = perf_counter_ns()
    while True:
        if Atomic[DType.uint64].load[ordering = Ordering.ACQUIRE](
            _mb(st, MB_STOP)
        ) != 0:
            return
        var req = Int(
            Atomic[DType.uint64].load[ordering = Ordering.ACQUIRE](
                _mb(st, MB_REQUEST)
            )
        )
        if req > st.request_seq:
            st.request_seq = req
            st.last_progress_ns = perf_counter_ns()
        var moved = ib_drive(st)
        if st.done_seq > published:
            # Published even on failure: the spin kernels must be released or
            # the stream hangs past the point where the error can be
            # reported.
            published = st.done_seq
            Atomic[DType.uint64].store[ordering = Ordering.RELEASE](
                _mb(st, MB_DONE), UInt64(published)
            )
        if moved or st.done_seq < st.request_seq:
            continue
        _ = external_call["sched_yield", Int32]()
        if Atomic[DType.uint64].load[ordering = Ordering.ACQUIRE](
            _mb(st, MB_REQUEST)
        ) <= UInt64(st.request_seq):
            _nanosleep_ns(st.ts, idle_ns)


def _proxy_address() -> Int:
    var f: def (
        OpaquePointer[MutAnyOrigin]
    ) thin abi("C") -> None = _proxy_main
    return Pointer(to=f).unsafe_bitcast[Int]()[]


def _set_thread_affinity(tid: Int, cpu: Int) raises:
    """`pthread_setaffinity_np` to a single CPU. `cpu_set_t` is a 128-byte
    (1024-bit) bitmask on this ABI; only the one bit for `cpu` is set."""
    comptime CPU_SET_BYTES = 128
    var byte_idx = cpu // 8
    if cpu < 0 or byte_idx >= CPU_SET_BYTES:
        raise Error(
            "mojoccl: MOJOCCL_IB_PROXY_CPU=" + String(cpu) + " out of range"
        )
    var mask = alloc_bytes(CPU_SET_BYTES)
    mask[unsafe_offset=byte_idx] = UInt8(1) << UInt8(cpu % 8)
    var rc = external_call["pthread_setaffinity_np", Int32](
        Int64(tid), UInt64(CPU_SET_BYTES), mask
    )
    if rc != 0:
        raise Error(
            "mojoccl: pthread_setaffinity_np failed, rc=" + String(rc)
        )


def _start_proxy(ib: Int) raises:
    ref st = _st(ib)[]
    var tid = unsafe_alloc[Int64](1)
    tid[unsafe_offset=0] = 0
    var rc = external_call["pthread_create", Int32](
        tid, Int64(0), _proxy_address(), ib
    )
    if rc != 0:
        raise Error("mojoccl: pthread_create failed, rc=" + String(rc))
    st.thread_id = Int(tid[unsafe_offset=0])
    # Opt-in only (default: no pinning) -- a best-effort placement hint, not
    # load-bearing for correctness, so a bad CPU index or a failed syscall
    # only prints rather than failing communicator init.
    var cpu_s = getenv("MOJOCCL_IB_PROXY_CPU", "")
    if cpu_s.byte_length() > 0:
        try:
            _set_thread_affinity(st.thread_id, Int(cpu_s))
        except e:
            print("mojoccl: MOJOCCL_IB_PROXY_CPU pinning failed:", e)


def _stop_proxy(mut st: IbState):
    if st.thread_id == 0:
        return
    Atomic[DType.uint64].store[ordering = Ordering.RELEASE](
        _mb(st, MB_STOP), 1
    )
    _ = external_call["pthread_join", Int32](st.thread_id, Int64(0))
    st.thread_id = 0


def ib_signal_abort(ib: Int):
    """`ncclCommAbort`'s hook: raise MB_STOP and reclaim the progress thread
    without ncclCommDestroy's unbounded wait.

    Left unsignaled, the thread spins on the mailbox forever (nothing else
    ever sets MB_STOP for it) and burns one CPU core for the rest of the
    process. Unlike `_stop_proxy`, the join here is bounded
    (`IB_ABORT_JOIN_TIMEOUT_S`) with `pthread_tryjoin_np`, polled rather than
    blocking: abort must not hang because a dead peer left this rank's
    thread waiting for a completion that will never come -- `ib_drive` never
    blocks, so the loop notices MB_STOP within one step; this bound is only
    insurance against the case that doesn't anticipate.
    A thread this gives up on is simply left running; it exits on its own
    once it next checks MB_STOP, and the process exiting reclaims it either
    way.
    """
    if ib == 0:
        return
    ref st = _st(ib)[]
    if st.thread_id == 0:
        return
    Atomic[DType.uint64].store[ordering = Ordering.RELEASE](
        _mb(st, MB_STOP), 1
    )
    var tid = st.thread_id
    var deadline = perf_counter_ns() + Int(IB_ABORT_JOIN_TIMEOUT_S * 1.0e9)
    var retval = unsafe_alloc[Int64](1)
    while perf_counter_ns() < deadline:
        var rc = external_call["pthread_tryjoin_np", Int32](tid, retval)
        if rc == 0:
            st.thread_id = 0
            return
        sleep(0.001)


def _callback_address() -> Int:
    var f: def (OpaquePointer[MutAnyOrigin]) thin abi("C") -> None = _ib_progress
    return Pointer(to=f).unsafe_bitcast[Int]()[]


# ===-------------------------------------------------------------------=== #
# Setup
# ===-------------------------------------------------------------------=== #


def _realpath(path: String) -> String:
    """realpath(3): the ABSOLUTE /sys/devices path behind a sysfs symlink.

    `readlink` would return the stored relative target (`../../..0000:18:00.0`),
    and two of those share no comparable prefix -- the whole point here is to
    compare where a GPU and an HCA sit in one PCI tree.
    """
    var buf = alloc_bytes(4096)
    var cpath = alloc_bytes(path.byte_length() + 1)
    var pb = path.as_bytes()
    for i in range(len(pb)):
        cpath[unsafe_offset=i] = pb[i]
    var rc = Int(external_call["realpath", Int64](cpath, buf))
    if rc == 0:
        return String("")
    var s = String("")
    var i = 0
    while i < 4095 and buf[unsafe_offset=i] != 0:
        s += chr(Int(buf[unsafe_offset=i]))
        i += 1
    return s^


def _common_prefix(a: String, b: String) -> Int:
    var ab = a.as_bytes()
    var bb = b.as_bytes()
    var n = min(len(ab), len(bb))
    var i = 0
    while i < n and ab[i] == bb[i]:
        i += 1
    return i


def _choose_port(
    ports: List[IbPort], gpu_bdf: String, local_rank: Int
) raises -> Int:
    """Index of the HCA this rank should use.

    Preference is PCI proximity, measured the cheap way NCCL's topology
    measures it in spirit: both the GPU and the HCA are PCI devices, so the
    longer the shared prefix of their /sys/devices paths, the fewer switch
    hops between them. Ranks that tie fall back to `local_rank % n`, which
    on these nodes (10 IB HCAs, 8 GPUs) already hands every rank its own
    NIC. MOJOCCL_IB_HCA short-circuits all of it upstream, in list_ib_ports.
    """
    if len(ports) == 1 or gpu_bdf.byte_length() == 0:
        return local_rank % len(ports)
    var gpu_path = _realpath("/sys/bus/pci/devices/" + gpu_bdf)
    if gpu_path.byte_length() == 0:
        return local_rank % len(ports)
    var best = -1
    var best_score = -1
    var ties = 0
    for i in range(len(ports)):
        var hca_path = _realpath(
            "/sys/class/infiniband/" + ports[i].name + "/device"
        )
        var score = _common_prefix(gpu_path, hca_path)
        if score > best_score:
            best_score = score
            best = i
            ties = 1
        elif score == best_score:
            ties += 1
    if ties > 1 or best < 0:
        # Several NICs are equally close (the common case: one PCI switch per
        # pair of GPUs). Spread ranks over them deterministically.
        var chosen = List[Int]()
        for i in range(len(ports)):
            var hca_path = _realpath(
                "/sys/class/infiniband/" + ports[i].name + "/device"
            )
            if _common_prefix(gpu_path, hca_path) == best_score:
                chosen.append(i)
        return chosen[local_rank % len(chosen)]
    return best


def ib_setup(
    driver: OwnedDLHandle,
    ordinal: Int,
    local_rank: Int,
    my_node: Int,
    nnodes: Int,
    region: Int,
    region_bytes: Int,
    nslots: Int,
    credit_off: Int,
) raises -> Int:
    """Open an HCA, register the region, create the QPs (still in INIT).

    Returns the address of a heap `IbState`. The QPs cannot reach RTR until
    the peers' `(qpn, lid, gid)` have been gathered, so bring-up is split:
    this, then `ib_local_info` / `ib_connect`.

    `nslots` is how many fixed inbox slot groups the caller carved out of the
    region and therefore how many exchanges may be outstanding; `credit_off`
    is the region offset of the credit landing pad. Both must be identical on
    every rank -- they are part of the wire layout.
    """
    var ibv = Ibv()
    var want = getenv("MOJOCCL_IB_HCA", "")
    var ports = list_ib_ports(ibv, want)
    if len(ports) == 0:
        raise Error(
            "mojoccl: no ACTIVE InfiniBand port found"
            + (" matching MOJOCCL_IB_HCA=" + want if want.byte_length() > 0 else "")
            + "; a multi-node communicator needs one"
        )
    # Only a hint for HCA affinity: a driver without the symbol, or a
    # device that will not report one, falls back to round-robin.
    var gpu_bdf = String("")
    try:
        gpu_bdf = device_pci_bus_id(driver, ordinal)
    except:
        pass
    var pick = _choose_port(ports, gpu_bdf, local_rank)
    ref port = ports[pick]
    # These nodes carry ~10 IB HCAs and every rank opened all of them to
    # read their ports; hold only the one this rank will use.
    for i in range(len(ports)):
        if ports[i].ctx != port.ctx:
            ibv.close_device(ports[i].ctx)

    if nslots < 1 or nslots > PIPE_MAX_SLOTS:
        raise Error(
            "mojoccl: nslots must be in 1.."
            + String(PIPE_MAX_SLOTS)
            + ", got "
            + String(nslots)
        )
    var st = IbState(ibv^, region, my_node, nnodes, nslots, credit_off)
    st.hca = String(port.name)
    st.ctx = port.ctx
    st.port = port.port
    st.timeout_ns = Int(_ib_timeout_s() * 1.0e9)

    # Every step below can raise after an earlier one already allocated a
    # real ibverbs/host resource (pd, mr, cq, QPs, the pinned mailbox) --
    # unwind whatever got that far instead of leaking it.
    try:
        st.pd = st.ibv.alloc_pd(st.ctx)
        if st.pd == 0:
            raise Error("mojoccl: ibv_alloc_pd failed on " + st.hca)
        var ro = getenv("MOJOCCL_IB_RELAXED_ORDERING", "1") != "0"
        var acc = (
            IBV_ACCESS_LOCAL_WRITE
            | IBV_ACCESS_REMOTE_WRITE
            | IBV_ACCESS_REMOTE_READ
        )
        st.mr = (
            st.ibv.reg_mr_relaxed(st.pd, region, region_bytes, acc)
            if ro
            else st.ibv.reg_mr(st.pd, region, region_bytes, acc)
        )
        if st.mr == 0:
            raise Error(
                "mojoccl: ibv_reg_mr of the "
                + String(region_bytes // (1024 * 1024))
                + " MiB device region failed on "
                + st.hca
                + "; is nvidia_peermem (or the ROCm equivalent) loaded?"
            )
        var mrp = P8(unsafe_from_address=st.mr)
        st.lkey = ldu32(mrp, MR_LKEY)
        st.rkey = ldu32(mrp, MR_RKEY)

        # Host landing pad for the flush read.
        st.flush_host = Int(alloc_bytes(4096))
        st.flush_mr = st.ibv.reg_mr(
            st.pd, st.flush_host, 4096, IBV_ACCESS_LOCAL_WRITE
        )
        if st.flush_mr == 0:
            raise Error("mojoccl: ibv_reg_mr of the flush buffer failed")
        st.flush_lkey = ldu32(P8(unsafe_from_address=st.flush_mr), MR_LKEY)

        st.cq = st.ibv.create_cq(st.ctx, CQ_SIZE)
        if st.cq == 0:
            raise Error("mojoccl: ibv_create_cq failed")

        for j in range(nnodes):
            if j == my_node:
                continue
            var qp = create_rc_qp(
                st.ibv, st.pd, st.cq, SEND_WR_DEPTH, RECV_DEPTH + 8
            )
            # Recorded before `qp_to_init` can raise, so the unwind below
            # destroys it.
            st.peers.append(IbPeer(j, qp, qp_number(qp), 0, 0))
            st.credit_recv.append(0)
            st.send_done.append(0)
            qp_to_init(st.ibv, qp, st.port)
        st.flush_qp = create_rc_qp(st.ibv, st.pd, st.cq, SEND_WR_DEPTH, 8)
        qp_to_init(st.ibv, st.flush_qp, st.port)

        if st.proxy:
            st.mailbox = alloc_host(driver, MB_BYTES)
            st.mailbox_dev = host_device_ptr(driver, st.mailbox)
            for i in range(MB_BYTES // 8):
                Pointer[UInt64, MutAnyOrigin](unsafe_from_address=st.mailbox)[
                    unsafe_offset=i
                ] = 0
    except e:
        _teardown_ib_resources(st)
        raise e

    var holder = unsafe_alloc[IbState](1)
    holder.unsafe_write(st^)
    if _st(Int(holder))[].proxy:
        try:
            _start_proxy(Int(holder))
        except e:
            _teardown_ib_resources(_st(Int(holder))[])
            raise e
    return Int(holder)


def _ib_timeout_s() -> Float64:
    var s = getenv("MOJOCCL_IB_TIMEOUT_S", String(DEFAULT_IB_TIMEOUT_S))
    try:
        return Float64(s)
    except:
        return DEFAULT_IB_TIMEOUT_S


def _proxy_idle_ns() -> Int:
    """`MOJOCCL_IB_PROXY_IDLE_US`, read once at thread start (not from inside
    the progress-thread loop -- a `getenv` per idle iteration would defeat
    the point of backing off)."""
    var s = getenv(
        "MOJOCCL_IB_PROXY_IDLE_US", String(DEFAULT_IB_PROXY_IDLE_US)
    )
    var us = DEFAULT_IB_PROXY_IDLE_US
    try:
        var parsed = Int(s)
        if parsed > 0:
            us = parsed
    except:
        pass
    return us * 1000


def _nanosleep_ns(ts_addr: Int, ns: Int):
    """`nanosleep(2)` for `ns` nanoseconds; `struct timespec{tv_sec,tv_nsec}`,
    16 bytes on this ABI, in the caller-owned scratch at `ts_addr` (an
    allocation per call here leaked 16 bytes per idle iteration, ~3 GB/h at
    the 20 us quantum). Best-effort: an interrupted sleep just returns early,
    which only means the next mailbox check happens a bit sooner."""
    var ts = Pointer[Int64, MutAnyOrigin](unsafe_from_address=ts_addr)
    ts[unsafe_offset=0] = 0
    ts[unsafe_offset=1] = Int64(ns)
    _ = external_call["nanosleep", Int32](ts, Int64(0))


def ib_local_info(ib: Int, out_blob: P8, port_lid: Int, port_mtu: Int):
    """Fill this rank's IB half of the bootstrap blob.

    Layout (little-endian, matches `ib_connect`'s reader):
      +0   u64 region base VA
      +8   u32 rkey
      +12  u32 lid
      +16  u32 active_mtu (ibv_mtu enum)
      +20  u32 number of QPs that follow
      +24  u32 qpn[MAX_NODES]     -- indexed by the PEER's node
      +24+4*MAX_NODES  u8 gid[16]
    """
    ref st = _st(ib)[]
    out_blob.unsafe_bitcast[UInt64]()[unsafe_offset=0] = UInt64(st.region)
    out_blob.unsafe_bitcast[UInt32]()[unsafe_offset=2] = st.rkey
    out_blob.unsafe_bitcast[UInt32]()[unsafe_offset=3] = UInt32(port_lid)
    out_blob.unsafe_bitcast[UInt32]()[unsafe_offset=4] = UInt32(port_mtu)
    out_blob.unsafe_bitcast[UInt32]()[unsafe_offset=5] = UInt32(len(st.peers))
    for i in range(len(st.peers)):
        out_blob.unsafe_bitcast[UInt32]()[
            unsafe_offset = 6 + st.peers[i].node
        ] = st.peers[i].qpn


comptime IB_BLOB_BYTES = 24 + 4 * MAX_NODES + 16


def ib_port_lid(ib: Int) raises -> Int:
    """The port's LID, straight from `ibv_query_port`.

    A nonzero return code (not the same thing as the `try/except` this used
    to have -- `query_port`'s own C call never raises, it returns an errno)
    used to be silently discarded, reading LID 0 out of `pa`'s zeroed
    scratch. For the self-connected flush QP that 0 is not a sentinel
    anyone downstream checks; it just quietly modifies the flush QP with the
    wrong address. Raise instead.
    """
    ref st = _st(ib)[]
    var pa = alloc_bytes(56)
    var rc = st.ibv.query_port(st.ctx, st.port, pa)
    if rc != 0:
        raise Error("mojoccl: ibv_query_port failed, rc=" + String(rc))
    return Int(pa.unsafe_bitcast[UInt16]()[unsafe_offset=17])


def ib_port_mtu(ib: Int) raises -> Int:
    """The port's active MTU, straight from `ibv_query_port` (see
    `ib_port_lid` for why a failed query now raises instead of reading 0 out
    of zeroed scratch)."""
    ref st = _st(ib)[]
    var pa = alloc_bytes(56)
    var rc = st.ibv.query_port(st.ctx, st.port, pa)
    if rc != 0:
        raise Error("mojoccl: ibv_query_port failed, rc=" + String(rc))
    return ld32(pa, 8)


def ib_connect(
    ib: Int, blobs: P8, blob_stride: Int, peer_rank_of_node: List[Int], my_mtu: Int
) raises:
    """Move every QP to RTS from the gathered table, then pre-post recvs.

    `peer_rank_of_node[j]` is the global rank on node j holding this rank's
    local_rank -- the one this rank's QP j is paired with. `blobs` is the
    whole round-2 table; `blob_stride` its per-rank size.
    """
    ref st = _st(ib)[]
    for i in range(len(st.peers)):
        var j = st.peers[i].node
        var b = P8(
            unsafe_from_address=Int(blobs) + peer_rank_of_node[j] * blob_stride
        )
        var base = Int(b.unsafe_bitcast[UInt64]()[unsafe_offset=0])
        var rkey = b.unsafe_bitcast[UInt32]()[unsafe_offset=2]
        var lid = Int(b.unsafe_bitcast[UInt32]()[unsafe_offset=3])
        var mtu = Int(b.unsafe_bitcast[UInt32]()[unsafe_offset=4])
        # The peer's QP for MY node, not for its own.
        var dest_qpn = b.unsafe_bitcast[UInt32]()[
            unsafe_offset = 6 + st.my_node
        ]
        var gid = P8(unsafe_from_address=Int(b) + 24 + 4 * MAX_NODES)
        if lid == 0:
            raise Error(
                "mojoccl: peer on node "
                + String(j)
                + " reported LID 0 -- its HCA port is not on an InfiniBand"
                " fabric this library can address"
            )
        st.peers[i].remote_base = base
        st.peers[i].remote_rkey = rkey
        qp_to_rtr(
            st.ibv,
            st.peers[i].qp,
            dest_qpn,
            lid,
            min(my_mtu, mtu),
            st.port,
            gid,
            0,
            False,
        )
        qp_to_rts(st.ibv, st.peers[i].qp)
        for _ in range(RECV_DEPTH):
            build_recv_wr(_b(st.rwr), 0)
            if post_recv(st.peers[i].qp, _b(st.rwr), _b(st.bad)) != 0:
                raise Error("mojoccl: ibv_post_recv failed while pre-posting")

    # The flush QP talks to itself.
    var self_lid = ib_port_lid(ib)
    var gid0 = alloc_bytes(16)
    qp_to_rtr(
        st.ibv,
        st.flush_qp,
        qp_number(st.flush_qp),
        self_lid,
        my_mtu,
        st.port,
        gid0,
        0,
        False,
    )
    qp_to_rts(st.ibv, st.flush_qp)


# ===-------------------------------------------------------------------=== #
# Per-collective use
# ===-------------------------------------------------------------------=== #


def ib_npeers(ib: Int) -> Int:
    return len(_st(ib)[].peers)


def ib_error(ib: Int) -> Int:
    ref st = _st(ib)[]
    return _load_atomic_i(_err_ptr(st))


def ib_next_seq(ib: Int) -> Int:
    """Consume one exchange counter. Every rank of the communicator calls
    this the same number of times in the same order, so the slot group an
    exchange lands in, the credit window and the immediate that tags an
    arrival all agree across nodes."""
    ref st = _st(ib)[]
    st.exchanges += 1
    return st.exchanges


def ib_note_consumed(ib: Int, seq: Int):
    """Record that the consumer kernel for exchange `seq` has just been
    ENQUEUED on the stream.

    The number is carried into the next exchange's work item as
    `credit_upto` and published to the peers when the engine picks that
    exchange up -- at which point the request kernel has run and therefore
    every kernel enqueued before it, this consumer included, has completed.
    Callers enqueue the consumer first and call this second.

    That argument is stream order, so every exchange of one communicator has
    to be enqueued on ONE stream in issue order. The engine's dense sequence
    counter already required that (`_work` derives a ring slot from the
    number); `credit_upto` is the second thing that does.
    """
    ref st = _st(ib)[]
    if seq > st.consumed_enqueued:
        st.consumed_enqueued = seq


def _fill_work(
    mut st: IbState,
    ib: Int,
    send_addr: Int,
    send_bytes: Int,
    inbox_base: Int,
    slot_bytes: Int,
    do_send: Bool,
    nrecv: Int,
    flush_addr: Int,
    seq: Int,
    credit_upto: Int,
) raises:
    ref w = _work(st, seq)[]
    if _load_atomic_i(_status_ptr(w)) == 0:
        raise Error(
            "mojoccl: the inter-node work ring wrapped with an exchange still"
            " in flight; MOJOCCL_IB_TRACE=1 to see how far behind the network"
            " is"
        )
    w.state = ib
    w.send_addr = send_addr
    w.send_bytes = send_bytes
    w.inbox_base = inbox_base
    w.slot_bytes = slot_bytes
    w.do_send = 1 if do_send else 0
    w.nrecv = nrecv
    w.flush_addr = flush_addr
    w.seq = seq
    w.credit_upto = credit_upto
    w.sent = 0
    w.t0 = 0
    _store_atomic_i(_status_ptr(w), 0)


def ib_enqueue_request(
    ib: Int,
    driver: OwnedDLHandle,
    ctx: DeviceContext,
    stream: DeviceStream,
    raw_stream: Int,
    send_addr: Int,
    send_bytes: Int,
    inbox_base: Int,
    slot_bytes: Int,
    do_send: Bool,
    nrecv: Int,
    flush_addr: Int,
    seq: Int,
) raises:
    """Release exchange `seq` to the network, at this point in stream order.

    Enqueued right after the kernel that produced its payload. With the
    proxy thread (default) it is a one-thread kernel storing `seq` into the
    pinned mailbox, and the stream runs on: several exchanges may be in
    flight, and `ib_enqueue_wait` is what eventually stops the stream.
    Without the proxy (`MOJOCCL_IB_PROXY=0`) it is a `cuLaunchHostFunc` that
    runs the whole exchange inline -- correct with the same schedule, but
    with no overlap and several hundred microseconds of driver latency per
    exchange.
    """
    ref st = _st(ib)[]
    _fill_work(
        st,
        ib,
        send_addr,
        send_bytes,
        inbox_base,
        slot_bytes,
        do_send,
        nrecv,
        flush_addr,
        seq,
        st.consumed_enqueued,
    )
    if st.proxy:
        proxy_request(ctx, stream, st.mailbox_dev + MB_REQUEST, seq)
        return
    launch_host_func(
        driver,
        raw_stream,
        _callback_address(),
        st.works + ((seq - 1) % WORK_SLOTS) * size_of[IbWork](),
    )


def ib_enqueue_wait(
    ib: Int, ctx: DeviceContext, stream: DeviceStream, seq: Int
) raises:
    """Hold the stream until exchange `seq` has been retired, so the kernel
    enqueued next may read the inbox. A no-op on the `MOJOCCL_IB_PROXY=0`
    path, where `ib_enqueue_request`'s callback already waited."""
    ref st = _st(ib)[]
    if not st.proxy:
        return
    proxy_wait(
        ctx,
        stream,
        st.mailbox_dev + MB_DONE,
        st.error_word,
        seq,
        st.timeout_ns,
    )


def ib_submit_now(
    ib: Int,
    send_addr: Int,
    send_bytes: Int,
    inbox_base: Int,
    slot_bytes: Int,
    do_send: Bool,
    nrecv: Int,
    flush_addr: Int,
    seq: Int,
    credit_upto: Int,
) raises:
    """Hand one exchange to the engine from the calling thread, without
    waiting for it.

    The GPU-free self-tests use this (with `ib_wait_now`) to keep several
    exchanges in flight and exercise the credit protocol past the slot
    count, which is the case a leaked credit turns into a hang. Never
    correct inside a collective: there the exchange's position in stream
    order is the whole ordering argument.
    """
    ref st = _st(ib)[]
    _fill_work(
        st,
        ib,
        send_addr,
        send_bytes,
        inbox_base,
        slot_bytes,
        do_send,
        nrecv,
        flush_addr,
        seq,
        credit_upto,
    )
    if seq > st.request_seq:
        st.request_seq = seq
        st.last_progress_ns = perf_counter_ns()


def ib_wait_now(ib: Int, seq: Int) raises:
    """Drive the engine on the calling thread until exchange `seq` is done."""
    ref st = _st(ib)[]
    _drive_until(st, seq)
    if _load_atomic_i(_err_ptr(st)) != 0:
        raise Error(
            "mojoccl: inline exchange failed, ib error "
            + String(_load_atomic_i(_err_ptr(st)))
        )


def ib_exchange_now(
    ib: Int,
    send_addr: Int,
    send_bytes: Int,
    inbox_base: Int,
    slot_bytes: Int,
    do_send: Bool,
    nrecv: Int,
    flush_addr: Int,
    seq: Int,
    credit_upto: Int,
) raises:
    """One exchange, submitted and waited for on the calling thread.

    The bring-up self-test uses it: the transport can then be exercised on
    a host with InfiniBand but no GPU (registered host memory, no stream to
    hang kernels on), which is where the bootstrap/QP/immediate wiring is
    cheapest to debug -- run it with `MOJOCCL_IB_PROXY=0`, since the proxy
    mailbox needs a driver that can pin host memory.
    """
    ib_submit_now(
        ib,
        send_addr,
        send_bytes,
        inbox_base,
        slot_bytes,
        do_send,
        nrecv,
        flush_addr,
        seq,
        credit_upto,
    )
    ib_wait_now(ib, seq)


def ib_report(ib: Int):
    ref st = _st(ib)[]
    if not st.trace or st.n_exchanges == 0:
        return
    print(
        "mojoccl ib:",
        st.hca,
        "port",
        st.port,
        "peers",
        len(st.peers),
        "slots",
        st.nslots,
        "exchanges",
        st.n_exchanges,
        "credit stalls",
        st.n_credit_stalls,
        "| mean us post",
        Float64(st.t_post_ns) / Float64(st.n_exchanges) / 1000.0,
        "in flight",
        Float64(st.t_wait_ns) / Float64(st.n_exchanges) / 1000.0,
        "flush",
        Float64(st.t_flush_ns) / Float64(st.n_exchanges) / 1000.0,
    )


def _teardown_ib_resources(mut st: IbState):
    """Release every ibverbs/host resource `ib_setup` may have created.

    Shared by `ib_teardown` (a live communicator) and `ib_setup`'s own
    failure path (a later step raised after an earlier one already
    succeeded) -- both leave `st` in the same "some fields non-zero, some
    still their zero default" shape, and every field here is zero-guarded
    for exactly that reason.
    """
    if st.mailbox != 0:
        # Pinned, device-mapped host memory: a scarce OS resource, unlike the
        # few hundred bytes of plain heap this struct also holds. Safe here
        # and only here -- the progress thread is joined (or, from
        # `ib_setup`'s failure path, never started) and, for a live
        # communicator, the caller synchronized the stream the spin kernels
        # were on. `open_driver` re-opens an already-loaded library, so it
        # costs a refcount.
        try:
            free_host(open_driver(), st.mailbox)
        except:
            pass
        st.mailbox = 0
        st.mailbox_dev = 0
    try:
        for i in range(len(st.peers)):
            st.ibv.destroy_qp(st.peers[i].qp)
        if st.flush_qp != 0:
            st.ibv.destroy_qp(st.flush_qp)
        if st.cq != 0:
            st.ibv.destroy_cq(st.cq)
        if st.flush_mr != 0:
            st.ibv.dereg_mr(st.flush_mr)
        if st.mr != 0:
            st.ibv.dereg_mr(st.mr)
        if st.pd != 0:
            st.ibv.dealloc_pd(st.pd)
        if st.ctx != 0:
            st.ibv.close_device(st.ctx)
    except:
        pass


def ib_teardown(ib: Int):
    if ib == 0:
        return
    ref st = _st(ib)[]
    _stop_proxy(st)
    ib_report(ib)
    _teardown_ib_resources(st)
