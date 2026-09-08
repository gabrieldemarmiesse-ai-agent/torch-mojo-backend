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
# Flow control. The inbox is DOUBLE BUFFERED by the parity of a
# per-communicator exchange counter, and that is not an optimization, it is
# the proof of safety: peer B writes half `p` at exchanges e and e+2, and B
# cannot reach e+2 before receiving my e+1 data, which I send only after my
# own stream ran the add kernel of exchange e. Single buffering would leave
# B's e+1 write racing my e add kernel with nothing but timing in between.
# The argument needs two things. Every exchange must be all-to-all, which is
# why a rank with nothing to contribute still sends CREDIT_BYTES
# (mojoccl.mojo); and the two halves must be DISJOINT ADDRESSES for every
# exchange alike, which is why `_inbox_base` carves them out of the region
# once instead of sizing them from the message in flight (mojoccl.mojo has
# the case that broke). The immediate carries the exchange counter, so an
# arrival belonging to e+1 is counted into the other parity's tally instead
# of satisfying e.
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
    Ibv,
    alloc_bytes,
    be32,
    build_read_wr,
    build_recv_wr,
    build_write_wr,
    create_rc_qp,
    ld32,
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

# Recv WRs kept posted per peer QP. Each RDMA_WRITE_WITH_IMM consumes one;
# the callback reposts every one it consumes, so the depth only has to cover
# the burst a peer can produce while this rank is elsewhere -- two exchanges
# by the double-buffer argument above, times a wide margin.
comptime RECV_DEPTH = 64
comptime CQ_SIZE = 1024
comptime SEND_WR_DEPTH = 64
comptime WORK_SLOTS = 512
comptime MAX_NODES = 16
comptime DEFAULT_IB_TIMEOUT_S: Float64 = 60.0
# ncclCommAbort's bound on waiting for the progress thread: long enough for
# it to notice MB_STOP (immediate when idle, at most one poll_cq when inside
# `_run_exchange` -- see `_stop_requested`), short enough that abort's
# documented "don't wait" contract still holds even if the thread is wedged
# somewhere this bound does not anticipate.
comptime IB_ABORT_JOIN_TIMEOUT_S: Float64 = 2.0
# Bytes of the inbox read back by the flush; any read of the destination
# device flushes the writes ahead of it, the size is irrelevant.
comptime FLUSH_BYTES = 4

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
    """One inter-node exchange, handed to the host callback by address.

    Lives in a ring inside `IbState` so the host can enqueue ahead; a slot
    is reused only once its callback has set `status`.
    """

    var state: Int
    var send_addr: Int
    var send_bytes: Int
    var inbox_base: Int  # absolute VA offset from a region base, parity folded in
    var slot_bytes: Int
    var do_send: Int
    var nrecv: Int
    var flush_addr: Int
    var seq: Int
    var status: Int  # 0 running, 1 done, 2 failed

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
    var arrivals0: Int
    var arrivals1: Int
    var error: Int
    var timeout_ns: Int
    # Scratch buffers and the work ring are held as raw addresses: a
    # `Pointer[..., MutAnyOrigin]` cannot be a struct field, and these
    # outlive every borrow anyway (allocated once, freed never -- a few
    # hundred bytes per communicator).
    var wr: Int
    var sge: Int
    var rwr: Int
    var bad: Int
    var wc: Int
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

    def __init__(out self, var ibv: Ibv, region: Int, my_node: Int, nnodes: Int):
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
        self.arrivals0 = 0
        self.arrivals1 = 0
        self.error = 0
        self.timeout_ns = Int(DEFAULT_IB_TIMEOUT_S * 1.0e9)
        self.wr = Int(alloc_bytes(SZ_SEND_WR))
        self.sge = Int(alloc_bytes(SZ_SGE))
        self.rwr = Int(alloc_bytes(SZ_RECV_WR))
        self.bad = Int(alloc_bytes(16))
        self.wc = Int(alloc_bytes(SZ_WC * 16))
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
# `MOJOCCL_IB_PROXY=0` callback thread) inside `_run_exchange` and read by
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
# The host callback
# ===-------------------------------------------------------------------=== #


def _consume_wc(mut st: IbState, c: P8, npeers: Int) -> Int:
    """Classify one completion; returns 1 for a send, 2 for the flush read,
    0 for anything else, -1 for a failed completion (error recorded).

    Shared by both poll loops on purpose. An arrival for the NEXT exchange
    can land while this one is flushing, and a loop that only looked for its
    own opcode would drop it -- losing a tally the next callback is waiting
    on, and a recv WR nobody reposts.
    """
    if Int32(ld32(c, WC_STATUS)) != IBV_WC_SUCCESS:
        _store_atomic_i(
            _err_ptr(st),
            1000 + ld32(c, WC_STATUS) * 1000 + ld32(c, WC_VENDOR_ERR),
        )
        return -1
    var op = Int32(ld32(c, WC_OPCODE))
    if op == IBV_WC_RECV_RDMA_WITH_IMM:
        if Int(be32(ldu32(c, WC_IMM_DATA))) & 1 == 0:
            st.arrivals0 += 1
        else:
            st.arrivals1 += 1
        var qpn = ldu32(c, WC_QP_NUM)
        for k in range(npeers):
            if st.peers[k].qpn == qpn:
                build_recv_wr(_b(st.rwr), 0)
                _ = post_recv(st.peers[k].qp, _b(st.rwr), _b(st.bad))
                break
        return 0
    if op == IBV_WC_RDMA_WRITE:
        return 1
    if op == IBV_WC_RDMA_READ:
        return 2
    return 0


def _ib_progress(user: OpaquePointer[MutAnyOrigin]) abi("C"):
    """`cuLaunchHostFunc` entry point -- the MOJOCCL_IB_PROXY=0 path.

    Runs on a driver-owned thread with the stream stalled behind it, so it
    must never call the CUDA/HIP driver. Kept as a fallback, and as the
    thing the proxy thread is measured against: on this cluster the
    driver's stop-the-stream / wake-a-thread / restart round trip costs
    about 480 us per exchange (job 234035: a 1 MiB two-node allreduce took
    496 us against 24 us on one node, and the transfer in it is 3 us),
    which at the 27 MiB DDP bucket is several times the transfer it waits
    for.
    """
    ref w = Pointer[IbWork, MutAnyOrigin](unsafe_from_address=Int(user))[]
    _run_exchange(_st(w.state)[], w)


def _run_exchange(mut st: IbState, mut w: IbWork):
    """Post this exchange's RDMA writes, wait for the peers', flush."""
    if _load_atomic_i(_err_ptr(st)) != 0:
        _store_atomic_i(_status_ptr(w), 2)
        return
    var npeers = len(st.peers)
    var parity = w.seq & 1
    var t0 = perf_counter_ns()
    var deadline = t0 + st.timeout_ns
    var nsend = 0

    if w.do_send != 0 and w.send_bytes > 0:
        for i in range(npeers):
            ref p = st.peers[i]
            # Where MY shard sits in the peer's inbox: senders are indexed by
            # node, compacted past the receiver's own node.
            var slot = st.my_node if st.my_node < p.node else st.my_node - 1
            build_write_wr(
                _b(st.wr),
                _b(st.sge),
                w.seq,
                w.send_addr,
                st.lkey,
                w.send_bytes,
                p.remote_base + w.inbox_base + slot * w.slot_bytes,
                p.remote_rkey,
                UInt32(w.seq & 0x7FFFFFFF),
                True,
                True,
            )
            if post_send(p.qp, _b(st.wr), _b(st.bad)) != 0:
                _store_atomic_i(_err_ptr(st), 1)
                _store_atomic_i(_status_ptr(w), 2)
                return
            nsend += 1
    var t1 = perf_counter_ns()

    var sends_done = 0
    while sends_done < nsend or _arrivals(st, parity) < w.nrecv:
        if _stop_requested(st):
            _store_atomic_i(_err_ptr(st), 6)
            _store_atomic_i(_status_ptr(w), 2)
            return
        var n = poll_cq(st.cq, 16, _b(st.wc))
        if n < 0:
            _store_atomic_i(_err_ptr(st), 2)
            _store_atomic_i(_status_ptr(w), 2)
            return
        for i in range(Int(n)):
            var kind = _consume_wc(
                st, P8(unsafe_from_address=st.wc + i * SZ_WC), npeers
            )
            if kind < 0:
                _store_atomic_i(_status_ptr(w), 2)
                return
            if kind == 1:
                sends_done += 1
        if perf_counter_ns() > deadline:
            _store_atomic_i(_err_ptr(st), 3)
            _store_atomic_i(_status_ptr(w), 2)
            return
    if parity == 0:
        st.arrivals0 -= w.nrecv
    else:
        st.arrivals1 -= w.nrecv
    var t2 = perf_counter_ns()

    if w.nrecv > 0 and w.flush_addr != 0:
        build_read_wr(
            _b(st.wr),
            _b(st.sge),
            0,
            st.flush_host,
            st.flush_lkey,
            FLUSH_BYTES,
            w.flush_addr,
            st.rkey,
        )
        if post_send(st.flush_qp, _b(st.wr), _b(st.bad)) != 0:
            _store_atomic_i(_err_ptr(st), 4)
            _store_atomic_i(_status_ptr(w), 2)
            return
        var flushed = False
        while not flushed:
            if _stop_requested(st):
                _store_atomic_i(_err_ptr(st), 6)
                _store_atomic_i(_status_ptr(w), 2)
                return
            var n = poll_cq(st.cq, 16, _b(st.wc))
            if n < 0:
                _store_atomic_i(_err_ptr(st), 5)
                _store_atomic_i(_status_ptr(w), 2)
                return
            for i in range(Int(n)):
                var kind = _consume_wc(
                    st, P8(unsafe_from_address=st.wc + i * SZ_WC), npeers
                )
                if kind < 0:
                    _store_atomic_i(_status_ptr(w), 2)
                    return
                if kind == 2:
                    flushed = True
            if perf_counter_ns() > deadline:
                _store_atomic_i(_err_ptr(st), 7)
                _store_atomic_i(_status_ptr(w), 2)
                return
    var t3 = perf_counter_ns()

    st.t_post_ns += t1 - t0
    st.t_wait_ns += t2 - t1
    st.t_flush_ns += t3 - t2
    st.n_exchanges += 1
    _store_atomic_i(_status_ptr(w), 1)


@always_inline
def _arrivals(st: IbState, parity: Int) -> Int:
    return st.arrivals0 if parity == 0 else st.arrivals1


@always_inline
def _mb(st: IbState, off: Int) -> Pointer[UInt64, MutAnyOrigin]:
    return Pointer[UInt64, MutAnyOrigin](unsafe_from_address=st.mailbox + off)


@always_inline
def _stop_requested(st: IbState) -> Bool:
    """True once `ncclCommAbort` (or teardown) has raised MB_STOP.

    Checked inside `_run_exchange`'s two poll loops so a stop request ends a
    stuck exchange (a dead peer, nothing left to poll) promptly instead of
    making `_stop_proxy`'s `pthread_join` wait out the rest of
    `MOJOCCL_IB_TIMEOUT_S`. Only meaningful under the proxy -- the mailbox is
    allocated only when `st.proxy`, so the `MOJOCCL_IB_PROXY=0` callback path
    and the self-test's `ib_exchange_now` (no mailbox, no thread) never see
    it set.
    """
    return (
        st.mailbox != 0
        and Atomic[DType.uint64].load[ordering = Ordering.ACQUIRE](
            _mb(st, MB_STOP)
        )
        != 0
    )


def _proxy_main(arg: OpaquePointer[MutAnyOrigin]) abi("C"):
    """The progress thread: one exchange at a time, in stream order.

    Spins on the mailbox word a one-thread kernel releases after the
    reduce-scatter, runs the exchange, releases the done word the matching
    spin kernel is waiting on. Exchanges are strictly ordered on the
    communicator's stream and the counter is dense, so slot `(seq-1) mod
    WORK_SLOTS` is this exchange's work item and no queue is needed.

    A pure spin, like NCCL's proxy: the whole point is that neither side
    ever sleeps. One core per rank, and `MOJOCCL_IB_PROXY=0` gives it back
    at the cost of the host-callback latency.
    """
    ref st = _st(Int(arg))[]
    var next_seq = 1
    while True:
        if Atomic[DType.uint64].load[ordering = Ordering.ACQUIRE](
            _mb(st, MB_STOP)
        ) != 0:
            return
        if Atomic[DType.uint64].load[ordering = Ordering.ACQUIRE](
            _mb(st, MB_REQUEST)
        ) < UInt64(next_seq):
            continue
        ref w = Pointer[IbWork, MutAnyOrigin](unsafe_from_address=st.works)[
            unsafe_offset = (next_seq - 1) % WORK_SLOTS
        ]
        _run_exchange(st, w)
        # Published even on failure: the spin kernel must be released or the
        # stream hangs past the point where the error can be reported.
        Atomic[DType.uint64].store[ordering = Ordering.RELEASE](
            _mb(st, MB_DONE), UInt64(next_seq)
        )
        next_seq += 1


def _proxy_address() -> Int:
    var f: def (
        OpaquePointer[MutAnyOrigin]
    ) thin abi("C") -> None = _proxy_main
    return Pointer(to=f).unsafe_bitcast[Int]()[]


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
    thread waiting inside `_run_exchange` for a completion that will never
    come -- `_stop_requested` is what actually gets it out of there quickly;
    this bound is only insurance against the case that doesn't anticipate.
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
    while perf_counter_ns() < deadline:
        var retval = unsafe_alloc[Int64](1)
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
) raises -> Int:
    """Open an HCA, register the region, create the QPs (still in INIT).

    Returns the address of a heap `IbState`. The QPs cannot reach RTR until
    the peers' `(qpn, lid, gid)` have been gathered, so bring-up is split:
    this, then `ib_local_info` / `ib_connect`.
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

    var st = IbState(ibv^, region, my_node, nnodes)
    st.hca = String(port.name)
    st.ctx = port.ctx
    st.port = port.port
    st.timeout_ns = Int(_ib_timeout_s() * 1.0e9)

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
        qp_to_init(st.ibv, qp, st.port)
        st.peers.append(IbPeer(j, qp, qp_number(qp), 0, 0))
    st.flush_qp = create_rc_qp(st.ibv, st.pd, st.cq, SEND_WR_DEPTH, 8)
    qp_to_init(st.ibv, st.flush_qp, st.port)

    if st.proxy:
        st.mailbox = alloc_host(driver, MB_BYTES)
        st.mailbox_dev = host_device_ptr(driver, st.mailbox)
        for i in range(MB_BYTES // 8):
            Pointer[UInt64, MutAnyOrigin](unsafe_from_address=st.mailbox)[
                unsafe_offset=i
            ] = 0

    var holder = unsafe_alloc[IbState](1)
    holder.unsafe_write(st^)
    if _st(Int(holder))[].proxy:
        _start_proxy(Int(holder))
    return Int(holder)


def _ib_timeout_s() -> Float64:
    var s = getenv("MOJOCCL_IB_TIMEOUT_S", String(DEFAULT_IB_TIMEOUT_S))
    try:
        return Float64(s)
    except:
        return DEFAULT_IB_TIMEOUT_S


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


def ib_port_lid(ib: Int) -> Int:
    ref st = _st(ib)[]
    var pa = alloc_bytes(56)
    try:
        _ = st.ibv.query_port(st.ctx, st.port, pa)
    except:
        return 0
    return Int(pa.unsafe_bitcast[UInt16]()[unsafe_offset=17])


def ib_port_mtu(ib: Int) -> Int:
    ref st = _st(ib)[]
    var pa = alloc_bytes(56)
    try:
        _ = st.ibv.query_port(st.ctx, st.port, pa)
    except:
        return 0
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
    this the same number of times in the same order, so the parity that
    selects the inbox half and the immediate that tags an arrival agree
    across nodes."""
    ref st = _st(ib)[]
    st.exchanges += 1
    return st.exchanges


def ib_enqueue(
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
    """Put one exchange on `stream`, between the kernel that produced its
    payload and the kernel that consumes what arrives.

    Two shapes, same ordering guarantee. With the proxy thread (default): a
    one-thread kernel releases `seq` into the pinned mailbox and a second
    one spins until the thread reports it done. Without it
    (`MOJOCCL_IB_PROXY=0`): a `cuLaunchHostFunc` that does the exchange
    inline, which is simpler and several hundred microseconds slower per
    exchange.
    """
    ref st = _st(ib)[]
    var slot = (seq - 1) % WORK_SLOTS
    ref w = Pointer[IbWork, MutAnyOrigin](unsafe_from_address=st.works)[
        unsafe_offset=slot
    ]
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
    _store_atomic_i(_status_ptr(w), 0)
    if st.proxy:
        proxy_request(ctx, stream, st.mailbox_dev + MB_REQUEST, seq)
        proxy_wait(
            ctx,
            stream,
            st.mailbox_dev + MB_DONE,
            st.error_word,
            seq,
            st.timeout_ns,
        )
        return
    launch_host_func(
        driver,
        raw_stream,
        _callback_address(),
        st.works + slot * size_of[IbWork](),
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
) raises:
    """Run one exchange inline on the calling thread.

    The bring-up self-test uses it: the transport can then be exercised on
    a host with InfiniBand but no GPU (registered host memory, no stream to
    hang kernels on), which is where the bootstrap/QP/immediate wiring is
    cheapest to debug -- run it with `MOJOCCL_IB_PROXY=0`, since the proxy
    mailbox needs a driver that can pin host memory. Never correct inside a
    collective: there the exchange's position in stream order is the whole
    ordering argument.
    """
    ref st = _st(ib)[]
    var w = IbWork()
    w.state = ib
    w.send_addr = send_addr
    w.send_bytes = send_bytes
    w.inbox_base = inbox_base
    w.slot_bytes = slot_bytes
    w.do_send = 1 if do_send else 0
    w.nrecv = nrecv
    w.flush_addr = flush_addr
    w.seq = seq
    _store_atomic_i(_status_ptr(w), 0)
    _run_exchange(st, w)
    if _load_atomic_i(_status_ptr(w)) != 1:
        raise Error(
            "mojoccl: inline exchange failed, ib error "
            + String(_load_atomic_i(_err_ptr(st)))
        )


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
        "exchanges",
        st.n_exchanges,
        "| mean us post",
        Float64(st.t_post_ns) / Float64(st.n_exchanges) / 1000.0,
        "wait",
        Float64(st.t_wait_ns) / Float64(st.n_exchanges) / 1000.0,
        "flush",
        Float64(st.t_flush_ns) / Float64(st.n_exchanges) / 1000.0,
    )


def ib_teardown(ib: Int):
    if ib == 0:
        return
    ref st = _st(ib)[]
    _stop_proxy(st)
    ib_report(ib)
    if st.mailbox != 0:
        # Pinned, device-mapped host memory: a scarce OS resource, unlike the
        # few hundred bytes of plain heap this struct also holds. Safe here
        # and only here -- the progress thread is joined and the caller
        # synchronized the stream the spin kernels were on. `open_driver`
        # re-opens an already-loaded library, so it costs a refcount.
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
