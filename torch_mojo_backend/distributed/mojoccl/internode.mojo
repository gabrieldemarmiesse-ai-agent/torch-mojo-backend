# The inter-node hop: GPUDirect RDMA over libibverbs, no vendor collective
# library. One RC queue pair per remote node, to the rank holding the SAME
# local_rank there -- so the 8 ranks of a node drive 8 independent NICs and
# each rank only ever exchanges its own 1/local_world shard.
#
# Ordering without a proxy thread. The GPU cannot post verbs and the NIC
# cannot wait on a kernel, so the two meet in a host function enqueued on
# the caller's stream (`cuLaunchHostFunc`/`hipLaunchHostFunc`):
#
#     [reduce_scatter_stage]  my shard is final in my stage_out
#     [host callback]         post one RDMA_WRITE_WITH_IMM per peer, then
#                             poll the CQ until every peer's shard has
#                             landed in my inbox, then flush (below)
#     [inbox_add]             shard += the peers' shards
#     [allgather_finish]      spread the global sum
#
# One fused callback rather than the post/poll pair: they would be adjacent
# on the stream with nothing in between, so splitting only pays a second
# dispatch latency. A proxy thread with GPU-visible pinned flags (NCCL's
# design) is the fallback if the dispatch turns out to dominate; the split
# above is where it would slot in.
#
# The GPUDirect flush. Seeing the RDMA_WRITE_WITH_IMM completion does NOT
# mean the payload is visible in GPU memory: the completion lands in host
# memory and the payload in the GPU's BAR, two different PCIe destinations
# with no ordering between them. A read from the GPU BAR flushes the posted
# writes ahead of it, so the callback finishes with a 4-byte RDMA_READ of
# the inbox over a self-connected QP -- exactly NCCL's `gpuFlush` QP
# (nccl:src/transport/net_ib/p2p.cc:589-602). Measured 1.9 us.
#
# Inbox aliasing and flow control. The inbox lives in stage_in, above
# whatever the intra-node collective staged there, and is DOUBLE BUFFERED by
# the parity of a per-communicator exchange counter. The double buffer is
# not an optimization, it is the proof of safety: peer B writes half `p` at
# exchanges e and e+2, and B cannot reach e+2 before receiving my e+1 data,
# which I send only after my own stream ran the add kernel of exchange e.
# Single buffering would leave B's e+1 write racing my e add kernel with
# nothing but timing in between. The immediate carries the exchange counter
# so an arrival that belongs to e+1 is counted into the other parity's
# tally instead of satisfying e.

from std.ffi import OwnedDLHandle, external_call
from std.memory.alloc import unsafe_alloc
from std.sys import size_of
from std.os import getenv
from std.time import perf_counter_ns

from driver import device_pci_bus_id, launch_host_func
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
# Bytes of the inbox read back by the flush; any read of the destination
# device flushes the writes ahead of it, the size is irrelevant.
comptime FLUSH_BYTES = 4


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


# ===-------------------------------------------------------------------=== #
# The host callback
# ===-------------------------------------------------------------------=== #


def _ib_progress(user: OpaquePointer[MutAnyOrigin]) abi("C"):
    """Post this exchange's RDMA writes, wait for the peers', flush.

    Runs on a driver-owned thread with the stream stalled behind it, so it
    must be quick and must never call the CUDA/HIP driver. Only libibverbs
    and this struct's own host memory are touched.
    """
    ref w = Pointer[IbWork, MutAnyOrigin](unsafe_from_address=Int(user))[]
    ref st = _st(w.state)[]
    if st.error != 0:
        w.status = 2
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
                st.error = 1
                w.status = 2
                return
            nsend += 1
    var t1 = perf_counter_ns()

    var sends_done = 0
    while sends_done < nsend or _arrivals(st, parity) < w.nrecv:
        var n = poll_cq(st.cq, 16, _b(st.wc))
        if n < 0:
            st.error = 2
            w.status = 2
            return
        for i in range(Int(n)):
            var c = P8(unsafe_from_address=st.wc + i * SZ_WC)
            if Int32(ld32(c, WC_STATUS)) != IBV_WC_SUCCESS:
                st.error = 1000 + ld32(c, WC_STATUS) * 1000 + ld32(
                    c, WC_VENDOR_ERR
                )
                w.status = 2
                return
            var op = Int32(ld32(c, WC_OPCODE))
            if op == IBV_WC_RECV_RDMA_WITH_IMM:
                var seq = Int(be32(ldu32(c, WC_IMM_DATA)))
                if seq & 1 == 0:
                    st.arrivals0 += 1
                else:
                    st.arrivals1 += 1
                var qpn = ldu32(c, WC_QP_NUM)
                for k in range(npeers):
                    if st.peers[k].qpn == qpn:
                        build_recv_wr(_b(st.rwr), 0)
                        _ = post_recv(st.peers[k].qp, _b(st.rwr), _b(st.bad))
                        break
            elif op == IBV_WC_RDMA_WRITE:
                sends_done += 1
        if perf_counter_ns() > deadline:
            st.error = 3
            w.status = 2
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
            st.error = 4
            w.status = 2
            return
        var flushed = False
        while not flushed:
            var n = poll_cq(st.cq, 16, _b(st.wc))
            if n < 0:
                st.error = 5
                w.status = 2
                return
            for i in range(Int(n)):
                var c = P8(unsafe_from_address=st.wc + i * SZ_WC)
                if Int32(ld32(c, WC_STATUS)) != IBV_WC_SUCCESS:
                    st.error = 6
                    w.status = 2
                    return
                if Int32(ld32(c, WC_OPCODE)) == IBV_WC_RDMA_READ:
                    flushed = True
            if perf_counter_ns() > deadline:
                st.error = 7
                w.status = 2
                return
    var t3 = perf_counter_ns()

    st.t_post_ns += t1 - t0
    st.t_wait_ns += t2 - t1
    st.t_flush_ns += t3 - t2
    st.n_exchanges += 1
    w.status = 1


@always_inline
def _arrivals(st: IbState, parity: Int) -> Int:
    return st.arrivals0 if parity == 0 else st.arrivals1


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
    st.mr = st.ibv.reg_mr(
        st.pd,
        region,
        region_bytes,
        IBV_ACCESS_LOCAL_WRITE
        | IBV_ACCESS_REMOTE_WRITE
        | IBV_ACCESS_REMOTE_READ,
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

    var holder = unsafe_alloc[IbState](1)
    holder.unsafe_write(st^)
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
    return _st(ib)[].error


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
    """Enqueue one exchange's host callback on `raw_stream`."""
    ref st = _st(ib)[]
    var slot = st.work_next % WORK_SLOTS
    st.work_next += 1
    ref w = Pointer[IbWork, MutAnyOrigin](unsafe_from_address=st.works)[
        unsafe_offset=slot
    ]
    if w.status == 0:
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
    w.status = 0
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
    """Run one exchange inline on the calling thread instead of from a
    stream callback.

    The bring-up self-test uses it: the transport can be exercised on a host
    with IB but no GPU (registered host memory, no stream to hang a callback
    on), which is where the bootstrap/QP/immediate wiring is cheapest to
    debug. Never correct in a collective -- there the callback's position in
    stream order is the whole ordering argument.
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
    w.status = 0
    var wp = unsafe_alloc[IbWork](1)
    wp.unsafe_write(w^)
    _ib_progress(OpaquePointer[MutAnyOrigin](unsafe_from_address=Int(wp)))
    if wp[unsafe_offset=0].status != 1:
        raise Error(
            "mojoccl: inline exchange failed, ib error " + String(st.error)
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
    ib_report(ib)
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
