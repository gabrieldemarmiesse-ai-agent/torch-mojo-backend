# libibverbs bindings for mojoccl's inter-node hop -- a narrow RC/RDMA-WRITE
# client, no vendor collective library anywhere.
#
# Two calling conventions, because libibverbs has two:
#
#  * control path (open/query/alloc/reg/create/modify/destroy) -- real
#    exported symbols, reached with OwnedDLHandle.get_function. Runs a
#    couple of dozen times per communicator, at init.
#  * data path (post_send / post_recv / poll_cq) -- NOT exported: they are
#    `static inline` in <infiniband/verbs.h> and dispatch through
#    `qp->context->ops.post_send` etc. NCCL does not include the inline
#    either, it hand-writes the same dereference
#    (nccl:src/include/ibvwrap.h:61,78,88), and so does this file: load the
#    8-byte function pointer at a fixed offset from the ibv_context and call
#    it. Zero dlsym on the hot path.
#
# Every struct offset below was dumped with gcc offsetof against
# /usr/include/infiniband/verbs.h on this machine (rdma-core 1.14.54.0 /
# MLNX OFED 24.10) and the ops-table offsets were additionally read back
# from a live mlx5 context; enum values likewise. Nothing here is from
# memory. x86-64 SysV: the structs are built in raw byte buffers because
# std.ffi still has no C-struct ABI (MOCO-3692).

from std.ffi import OwnedDLHandle
from std.os import getenv
from std.sys import size_of

from netutil import (
    MAX_NODES,
    NC_FLUSH,
    NC_OTHER,
    NC_RECV,
    NC_SEND,
    NetCompletion,
    P8,
    alloc_bytes,
    as_fn,
    ld8,
    ld16,
    ld32,
    ld64,
    ldu32,
    pci_pick,
    st8,
    st16,
    st32,
    st64,
    stu32,
    stu64,
)

# ---- enum values ---------------------------------------------------------
comptime IBV_QPS_INIT: Int32 = 1
comptime IBV_QPS_RTR: Int32 = 2
comptime IBV_QPS_RTS: Int32 = 3
comptime IBV_QPT_RC: Int32 = 2

comptime IBV_ACCESS_LOCAL_WRITE: Int32 = 1
comptime IBV_ACCESS_REMOTE_WRITE: Int32 = 2
comptime IBV_ACCESS_REMOTE_READ: Int32 = 4
# 1<<20, and above the range the versioned `ibv_reg_mr@IBVERBS_1.1` will
# accept -- reaching it needs `ibv_reg_mr_iova2@IBVERBS_1.8`, which is the
# only reason NCCL calls that entry point at all
# (nccl:src/transport/net_ib/reg.cc:41-46).
comptime IBV_ACCESS_RELAXED_ORDERING: Int32 = 0x100000

# The three composite attr masks, spelled out at nccl:src/transport/net_ib/
# connect.cc:377 (INIT), :435+441 (RTR), :492+500 (RTS).
comptime QP_MASK_INIT: Int32 = 0x39  # STATE|ACCESS_FLAGS|PKEY_INDEX|PORT
comptime QP_MASK_RTR: Int32 = 0x129181
comptime QP_MASK_RTS: Int32 = 0x12E01

comptime IBV_WR_RDMA_WRITE: Int32 = 0
comptime IBV_WR_RDMA_WRITE_WITH_IMM: Int32 = 1
comptime IBV_WR_RDMA_READ: Int32 = 4
comptime IBV_SEND_SIGNALED: Int32 = 2

comptime IBV_WC_SUCCESS: Int32 = 0
comptime IBV_WC_RDMA_WRITE: Int32 = 1
comptime IBV_WC_RDMA_READ: Int32 = 2
comptime IBV_WC_RECV_RDMA_WITH_IMM: Int32 = 129

comptime IBV_PORT_ACTIVE: Int32 = 4
comptime IBV_LINK_LAYER_INFINIBAND: Int = 1

# ---- struct sizes and field offsets --------------------------------------
comptime SZ_PORT_ATTR = 56
comptime SZ_QP_INIT_ATTR = 64
comptime SZ_QP_ATTR = 144
comptime SZ_SGE = 16
comptime SZ_SEND_WR = 128
comptime SZ_RECV_WR = 32
comptime SZ_WC = 48

# struct ibv_context: the ops table is at +8; these are absolute offsets of
# the four live data-path slots (verified against a live mlx5 context).
comptime CTX_POLL_CQ = 96
comptime CTX_POST_SEND = 208
comptime CTX_POST_RECV = 216

comptime DEV_NAME = 24  # struct ibv_device.name[64]
comptime QP_CONTEXT = 0  # struct ibv_qp.context  (also cq.context, mr.context)
comptime QP_QP_NUM = 52
comptime MR_LKEY = 36
comptime MR_RKEY = 40

comptime PA_STATE = 0  # struct ibv_port_attr
comptime PA_ACTIVE_MTU = 8
comptime PA_GID_TBL_LEN = 12
comptime PA_LID = 34
comptime PA_LINK_LAYER = 46

comptime QIA_SEND_CQ = 8  # struct ibv_qp_init_attr
comptime QIA_RECV_CQ = 16
comptime QIA_MAX_SEND_WR = 32
comptime QIA_MAX_RECV_WR = 36
comptime QIA_MAX_SEND_SGE = 40
comptime QIA_MAX_RECV_SGE = 44
comptime QIA_QP_TYPE = 52

comptime QA_QP_STATE = 0  # struct ibv_qp_attr
comptime QA_PATH_MTU = 8
comptime QA_RQ_PSN = 20
comptime QA_SQ_PSN = 24
comptime QA_DEST_QP_NUM = 28
comptime QA_ACCESS_FLAGS = 32
comptime QA_AH_DGID = 56  # ah_attr.grh.dgid
comptime QA_AH_SGID_INDEX = 76
comptime QA_AH_HOP_LIMIT = 77
comptime QA_AH_DLID = 80
comptime QA_AH_SL = 82
comptime QA_AH_IS_GLOBAL = 85
comptime QA_AH_PORT_NUM = 86
comptime QA_PKEY_INDEX = 120
comptime QA_MAX_RD_ATOMIC = 126
comptime QA_MAX_DEST_RD_ATOMIC = 127
comptime QA_MIN_RNR_TIMER = 128
comptime QA_PORT_NUM = 129
comptime QA_TIMEOUT = 130
comptime QA_RETRY_CNT = 131
comptime QA_RNR_RETRY = 132

comptime SGE_ADDR = 0
comptime SGE_LENGTH = 8
comptime SGE_LKEY = 12

comptime WR_ID = 0  # struct ibv_send_wr (and ibv_recv_wr for the first 4)
comptime WR_NEXT = 8
comptime WR_SG_LIST = 16
comptime WR_NUM_SGE = 24
comptime WR_OPCODE = 28
comptime WR_SEND_FLAGS = 32
comptime WR_IMM_DATA = 36
comptime WR_RDMA_REMOTE_ADDR = 40
comptime WR_RDMA_RKEY = 48

comptime WC_WR_ID = 0  # struct ibv_wc
comptime WC_STATUS = 8
comptime WC_OPCODE = 12
comptime WC_VENDOR_ERR = 16
comptime WC_BYTE_LEN = 20
comptime WC_IMM_DATA = 24
comptime WC_QP_NUM = 28

# ---- QP attribute values, all from NCCL ----------------------------------
# nccl:src/transport/net_ib/connect.cc:440,:502 -- both PSNs are hardcoded
# 0 and no PSN is exchanged, so neither is this library's wire format.
comptime IB_PSN: Int32 = 0
comptime IB_MIN_RNR_TIMER: UInt8 = 12  # connect.cc:443
comptime IB_TIMEOUT: UInt8 = 20  # connect.cc:496, NCCL_IB_TIMEOUT default
comptime IB_RETRY_CNT: UInt8 = 7  # connect.cc:497, NCCL_IB_RETRY_CNT default
comptime IB_RNR_RETRY: UInt8 = 7  # connect.cc:498, hardcoded (= infinite)
comptime IB_HOP_LIMIT: UInt8 = 255  # connect.cc:452,:476


# ===-------------------------------------------------------------------=== #
# The library handle (control path)
# ===-------------------------------------------------------------------=== #


struct Ibv(Movable):
    """The dlopened libibverbs.so.1.

    Control-path calls go through `get_function`; the data path never
    touches this struct.
    """

    var lib: OwnedDLHandle

    def __init__(out self) raises:
        self.lib = OwnedDLHandle("libibverbs.so.1")

    def get_device_list(self, out_n: P8) raises -> Int:
        return Int(self.lib.get_function[Int64]("ibv_get_device_list")(out_n))

    def free_device_list(self, list_addr: Int) raises:
        _ = self.lib.get_function[NoneType]("ibv_free_device_list")(list_addr)

    def open_device(self, dev: Int) raises -> Int:
        return Int(self.lib.get_function[Int64]("ibv_open_device")(dev))

    def close_device(self, ctx: Int) raises:
        _ = self.lib.get_function[Int32]("ibv_close_device")(ctx)

    def query_port(self, ctx: Int, port: Int, out_attr: P8) raises -> Int32:
        """The exported `ibv_query_port@IBVERBS_1.1` fills only the first 48
        of the 56 bytes (the `_compat_ibv_port_attr` layout; the extended
        entry point is `static inline`). Everything read here -- state@0,
        active_mtu@8, lid@34, link_layer@46 -- is inside those 48, and
        `alloc_bytes` zeroed the rest."""
        return self.lib.get_function[Int32]("ibv_query_port")(
            ctx, UInt8(port), out_attr
        )

    def query_gid(
        self, ctx: Int, port: Int, index: Int, out_gid: P8
    ) raises -> Int32:
        return self.lib.get_function[Int32]("ibv_query_gid")(
            ctx, UInt8(port), Int32(index), out_gid
        )

    def alloc_pd(self, ctx: Int) raises -> Int:
        return Int(self.lib.get_function[Int64]("ibv_alloc_pd")(ctx))

    def dealloc_pd(self, pd: Int) raises:
        _ = self.lib.get_function[Int32]("ibv_dealloc_pd")(pd)

    def reg_mr(
        self, pd: Int, addr: Int, length: Int, access: Int32
    ) raises -> Int:
        """Plain `ibv_reg_mr@IBVERBS_1.1`."""
        return Int(
            self.lib.get_function[Int64]("ibv_reg_mr")(
                pd, addr, UInt64(length), access
            )
        )

    def reg_mr_relaxed(
        self, pd: Int, addr: Int, length: Int, access: Int32
    ) raises -> Int:
        """`ibv_reg_mr_iova2` with IBV_ACCESS_RELAXED_ORDERING, falling back
        to `reg_mr`.

        PCIe relaxed ordering is what lets the NIC's writes into GPU memory
        retire out of order; without it a GPUDirect RDMA transfer runs at a
        fraction of link rate on this class of machine. NCCL turns it on by
        default (`NCCL_IB_PCI_RELAXED_ORDERING=2`,
        nccl:src/transport/net_ib/init.cc:11,141-150) and reaches it through
        `ibv_reg_mr_iova2` for the same ABI reason: the older entry point
        silently drops access bits above 0xFFFFF.

        The iova passed is the address itself -- the identity mapping NCCL
        also uses -- so remote addresses stay plain virtual addresses.
        Ordering of the DATA against its COMPLETION is not what RO relaxes
        and not what this library relies on: the flush read in
        internode.mojo is what makes the payload visible, and it is posted
        after the completion either way.
        """
        try:
            var mr = Int(
                self.lib.get_function[Int64]("ibv_reg_mr_iova2")(
                    pd,
                    addr,
                    UInt64(length),
                    UInt64(addr),
                    UInt32(access | IBV_ACCESS_RELAXED_ORDERING),
                )
            )
            if mr != 0:
                return mr
        except:
            # IBVERBS_1.8 absent (an old rdma-core): relaxed ordering is simply
            # unavailable, so register without it.
            return self.reg_mr(pd, addr, length, access)
        return self.reg_mr(pd, addr, length, access)

    def dereg_mr(self, mr: Int) raises:
        _ = self.lib.get_function[Int32]("ibv_dereg_mr")(mr)

    def create_cq(self, ctx: Int, cqe: Int) raises -> Int:
        return Int(
            self.lib.get_function[Int64]("ibv_create_cq")(
                ctx, Int32(cqe), Int64(0), Int64(0), Int32(0)
            )
        )

    def destroy_cq(self, cq: Int) raises:
        _ = self.lib.get_function[Int32]("ibv_destroy_cq")(cq)

    def create_qp(self, pd: Int, init_attr: P8) raises -> Int:
        return Int(self.lib.get_function[Int64]("ibv_create_qp")(pd, init_attr))

    def destroy_qp(self, qp: Int) raises:
        _ = self.lib.get_function[Int32]("ibv_destroy_qp")(qp)

    def modify_qp(self, qp: Int, attr: P8, mask: Int32) raises -> Int32:
        return self.lib.get_function[Int32]("ibv_modify_qp")(qp, attr, mask)


# ===-------------------------------------------------------------------=== #
# Data path -- ctx->ops.<fn>, no dlsym
# ===-------------------------------------------------------------------=== #


@always_inline
def post_send(qp: Int, wr: P8, bad_wr: P8) -> Int32:
    """`qp->context->ops.post_send(qp, wr, &bad_wr)`; 0 or an errno."""
    var f = as_fn[def(Int, P8, P8) thin abi("C") -> Int32](
        ld64(
            P8(
                unsafe_from_address=ld64(P8(unsafe_from_address=qp), QP_CONTEXT)
            ),
            CTX_POST_SEND,
        )
    )
    return f(qp, wr, bad_wr)


@always_inline
def post_recv(qp: Int, wr: P8, bad_wr: P8) -> Int32:
    var f = as_fn[def(Int, P8, P8) thin abi("C") -> Int32](
        ld64(
            P8(
                unsafe_from_address=ld64(P8(unsafe_from_address=qp), QP_CONTEXT)
            ),
            CTX_POST_RECV,
        )
    )
    return f(qp, wr, bad_wr)


@always_inline
def poll_cq(cq: Int, num_entries: Int, wc: P8) -> Int32:
    """Number of completions written into `wc`, or negative on error."""
    var f = as_fn[def(Int, Int32, P8) thin abi("C") -> Int32](
        ld64(
            P8(
                unsafe_from_address=ld64(P8(unsafe_from_address=cq), QP_CONTEXT)
            ),
            CTX_POLL_CQ,
        )
    )
    return f(cq, Int32(num_entries), wc)


# ===-------------------------------------------------------------------=== #
# Work-request builders
# ===-------------------------------------------------------------------=== #


def build_write_wr(
    wr: P8,
    sge: P8,
    wr_id: Int,
    local_addr: Int,
    lkey: UInt32,
    nbytes: Int,
    remote_addr: Int,
    rkey: UInt32,
    immediate: UInt32,
    with_imm: Bool,
    signaled: Bool,
):
    """One unchained RDMA_WRITE[_WITH_IMM]. `wr`/`sge` are caller-owned
    scratch reused across calls, so every field is written every time
    rather than relying on what was there before."""
    st64(sge, SGE_ADDR, local_addr)
    stu32(sge, SGE_LENGTH, UInt32(nbytes))
    stu32(sge, SGE_LKEY, lkey)
    for i in range(SZ_SEND_WR):
        wr[unsafe_offset=i] = 0
    st64(wr, WR_ID, wr_id)
    st64(wr, WR_NEXT, 0)
    st64(wr, WR_SG_LIST, Int(sge))
    st32(wr, WR_NUM_SGE, 1)
    st32(
        wr,
        WR_OPCODE,
        IBV_WR_RDMA_WRITE_WITH_IMM if with_imm else IBV_WR_RDMA_WRITE,
    )
    st32(wr, WR_SEND_FLAGS, IBV_SEND_SIGNALED if signaled else 0)
    # imm_data is __be32 on both sides (nccl:...:p2p.cc:157 htobe32, :653
    # be32toh). Byte-swapped here so the receiver's plain load reads it back.
    stu32(wr, WR_IMM_DATA, _bswap32(immediate))
    st64(wr, WR_RDMA_REMOTE_ADDR, remote_addr)
    stu32(wr, WR_RDMA_RKEY, rkey)


def build_read_wr(
    wr: P8,
    sge: P8,
    wr_id: Int,
    local_addr: Int,
    lkey: UInt32,
    nbytes: Int,
    remote_addr: Int,
    rkey: UInt32,
):
    """A signaled RDMA_READ -- the GPUDirect flush (see internode.mojo)."""
    st64(sge, SGE_ADDR, local_addr)
    stu32(sge, SGE_LENGTH, UInt32(nbytes))
    stu32(sge, SGE_LKEY, lkey)
    for i in range(SZ_SEND_WR):
        wr[unsafe_offset=i] = 0
    st64(wr, WR_ID, wr_id)
    st64(wr, WR_SG_LIST, Int(sge))
    st32(wr, WR_NUM_SGE, 1)
    st32(wr, WR_OPCODE, IBV_WR_RDMA_READ)
    st32(wr, WR_SEND_FLAGS, IBV_SEND_SIGNALED)
    st64(wr, WR_RDMA_REMOTE_ADDR, remote_addr)
    stu32(wr, WR_RDMA_RKEY, rkey)


def build_recv_wr(wr: P8, wr_id: Int):
    """An EMPTY receive WR (`sg_list = NULL, num_sge = 0`), which is what
    NCCL posts too (nccl:src/transport/net_ib/common.cc:88): its only job is
    to be consumed by an incoming RDMA_WRITE_WITH_IMM so a completion with
    the immediate appears on the CQ. The payload went straight to the
    registered region."""
    for i in range(SZ_RECV_WR):
        wr[unsafe_offset=i] = 0
    st64(wr, WR_ID, wr_id)


@always_inline
def _bswap32(v: UInt32) -> UInt32:
    return (
        ((v & 0xFF) << 24)
        | ((v & 0xFF00) << 8)
        | ((v >> 8) & 0xFF00)
        | ((v >> 24) & 0xFF)
    )


@always_inline
def be32(v: UInt32) -> UInt32:
    """Host <-> big-endian for the 32-bit immediate (involutive)."""
    return _bswap32(v)


# ===-------------------------------------------------------------------=== #
# QP bring-up
# ===-------------------------------------------------------------------=== #


def create_rc_qp(
    ibv: Ibv, pd: Int, cq: Int, max_send_wr: Int, max_recv_wr: Int
) raises -> Int:
    var ia = alloc_bytes(SZ_QP_INIT_ATTR)
    st64(ia, QIA_SEND_CQ, cq)
    st64(ia, QIA_RECV_CQ, cq)
    st32(ia, QIA_MAX_SEND_WR, Int32(max_send_wr))
    st32(ia, QIA_MAX_RECV_WR, Int32(max_recv_wr))
    st32(ia, QIA_MAX_SEND_SGE, 1)
    st32(ia, QIA_MAX_RECV_SGE, 1)
    st32(ia, QIA_QP_TYPE, IBV_QPT_RC)
    var qp = ibv.create_qp(pd, ia)
    if qp == 0:
        raise Error("mojoccl: ibv_create_qp failed")
    return qp


def qp_number(qp: Int) -> UInt32:
    return ldu32(P8(unsafe_from_address=qp), QP_QP_NUM)


def qp_to_init(ibv: Ibv, qp: Int, port: Int) raises:
    var a = alloc_bytes(SZ_QP_ATTR)
    st32(a, QA_QP_STATE, IBV_QPS_INIT)
    st16(a, QA_PKEY_INDEX, 0)
    st8(a, QA_PORT_NUM, UInt8(port))
    # Both directions on one QP pair: this rank writes into the peer's
    # region and the peer writes into this one, so REMOTE_WRITE is needed
    # on both ends (NCCL splits it because its QPs are one-directional).
    st32(
        a,
        QA_ACCESS_FLAGS,
        IBV_ACCESS_LOCAL_WRITE
        | IBV_ACCESS_REMOTE_WRITE
        | IBV_ACCESS_REMOTE_READ,
    )
    var rc = ibv.modify_qp(qp, a, QP_MASK_INIT)
    if rc != 0:
        raise Error("mojoccl: ibv_modify_qp(INIT) failed, rc=" + String(rc))


def qp_to_rtr(
    ibv: Ibv,
    qp: Int,
    dest_qpn: UInt32,
    dlid: Int,
    mtu: Int,
    port: Int,
    remote_gid: P8,
    local_gid_index: Int,
    global_route: Bool,
) raises:
    var a = alloc_bytes(SZ_QP_ATTR)
    st32(a, QA_QP_STATE, IBV_QPS_RTR)
    st32(a, QA_PATH_MTU, Int32(mtu))
    stu32(a, QA_DEST_QP_NUM, dest_qpn)
    st32(a, QA_RQ_PSN, IB_PSN)
    st8(a, QA_MAX_DEST_RD_ATOMIC, 1)
    st8(a, QA_MIN_RNR_TIMER, IB_MIN_RNR_TIMER)
    st16(a, QA_AH_DLID, UInt16(dlid))
    st8(a, QA_AH_SL, 0)
    st8(a, QA_AH_PORT_NUM, UInt8(port))
    if global_route:
        # Only when the two ports are on different IB subnets -- NCCL's rule
        # (connect.cc:456-458); a single-subnet fabric never takes this.
        st8(a, QA_AH_IS_GLOBAL, 1)
        for i in range(16):
            a[unsafe_offset=QA_AH_DGID + i] = remote_gid[unsafe_offset=i]
        st8(a, QA_AH_SGID_INDEX, UInt8(local_gid_index))
        st8(a, QA_AH_HOP_LIMIT, IB_HOP_LIMIT)
    var rc = ibv.modify_qp(qp, a, QP_MASK_RTR)
    if rc != 0:
        raise Error("mojoccl: ibv_modify_qp(RTR) failed, rc=" + String(rc))


def qp_to_rts(ibv: Ibv, qp: Int) raises:
    var a = alloc_bytes(SZ_QP_ATTR)
    st32(a, QA_QP_STATE, IBV_QPS_RTS)
    st32(a, QA_SQ_PSN, IB_PSN)
    st8(a, QA_TIMEOUT, IB_TIMEOUT)
    st8(a, QA_RETRY_CNT, IB_RETRY_CNT)
    st8(a, QA_RNR_RETRY, IB_RNR_RETRY)
    st8(a, QA_MAX_RD_ATOMIC, 1)
    var rc = ibv.modify_qp(qp, a, QP_MASK_RTS)
    if rc != 0:
        raise Error("mojoccl: ibv_modify_qp(RTS) failed, rc=" + String(rc))


# ===-------------------------------------------------------------------=== #
# Device selection
# ===-------------------------------------------------------------------=== #


struct IbPort(Copyable, Movable):
    """One usable (device, port): ACTIVE and InfiniBand link layer."""

    var name: String
    var ctx: Int
    var port: Int
    var lid: Int
    var mtu: Int
    var gid_index: Int
    var subnet_prefix: UInt64

    def __init__(
        out self,
        var name: String,
        ctx: Int,
        port: Int,
        lid: Int,
        mtu: Int,
        gid_index: Int,
        subnet_prefix: UInt64,
    ):
        self.name = name^
        self.ctx = ctx
        self.port = port
        self.lid = lid
        self.mtu = mtu
        self.gid_index = gid_index
        self.subnet_prefix = subnet_prefix


def _device_name(dev: Int) -> String:
    var dp = P8(unsafe_from_address=dev)
    var s = String("")
    var k = 0
    while k < 64 and dp[unsafe_offset=DEV_NAME + k] != 0:
        s += chr(Int(dp[unsafe_offset=DEV_NAME + k]))
        k += 1
    return s^


def list_ib_ports(ibv: Ibv, want: String) raises -> List[IbPort]:
    """Every ACTIVE InfiniBand port, in device order.

    RoCE (link_layer 2) and DOWN ports are skipped -- these nodes carry 10
    IB HCAs plus 2 RoCE ports and only the former are wanted. `want`, if
    non-empty, keeps only that device name (MOJOCCL_IB_HCA).
    Contexts of devices with no accepted port are closed again.
    """
    var out = List[IbPort]()
    var nbuf = alloc_bytes(8)
    var lst = ibv.get_device_list(nbuf)
    if lst == 0:
        return out^
    var n = ld32(nbuf, 0)
    var lp = P8(unsafe_from_address=lst)
    for i in range(n):
        var dev = ld64(lp, i * 8)
        var name = _device_name(dev)
        if want.byte_length() > 0 and name != want:
            continue
        var ctx = ibv.open_device(dev)
        if ctx == 0:
            continue
        var kept = False
        var pa = alloc_bytes(SZ_PORT_ATTR)
        # phys_port_cnt lives in ibv_device_attr; every HCA here is
        # single-port and NCCL itself iterates 1..phys_port_cnt, so probe
        # ports 1 and 2 and let query_port reject what is not there.
        for port in range(1, 3):
            for j in range(SZ_PORT_ATTR):
                pa[unsafe_offset=j] = 0
            if ibv.query_port(ctx, port, pa) != 0:
                continue
            if Int32(ld32(pa, PA_STATE)) != IBV_PORT_ACTIVE:
                continue
            if ld8(pa, PA_LINK_LAYER) != IBV_LINK_LAYER_INFINIBAND:
                continue
            var gid = alloc_bytes(16)
            var prefix: UInt64 = 0
            if ibv.query_gid(ctx, port, 0, gid) == 0:
                prefix = gid.unsafe_bitcast[UInt64]()[unsafe_offset=0]
            out.append(
                IbPort(
                    String(name),
                    ctx,
                    port,
                    ld16(pa, PA_LID),
                    ld32(pa, PA_ACTIVE_MTU),
                    0,
                    prefix,
                )
            )
            kept = True
        if not kept:
            ibv.close_device(ctx)
    ibv.free_device_list(lst)
    return out^


# ===-------------------------------------------------------------------=== #
# The transport, in the shape `internode.mojo`'s engine drives
# ===-------------------------------------------------------------------=== #
#
# `libfabric.mojo` offers the same six operations over a completely
# different API (post payload, post immediate, post flush, poll, local info,
# add peer); the engine calls one or the other and never learns which
# library is underneath. Everything below is the verbs half, moved here
# unchanged from the engine when the second transport arrived -- it is the
# code the two-node H100 + InfiniBand measurements were taken with.

# Recv WRs kept posted per peer QP. Each RDMA_WRITE_WITH_IMM consumes one --
# data and credits alike; the engine reposts every one it consumes, so the
# depth only has to cover the burst a peer can produce while this rank is
# elsewhere: `nslots` data messages plus `nslots` credits, times a wide
# margin.
comptime RECV_DEPTH = 64
comptime CQ_SIZE = 1024
comptime SEND_WR_DEPTH = 64


struct VerbsNet(Movable):
    """Everything the libibverbs transport owns, per communicator."""

    var ibv: Ibv
    var hca: String
    var ctx: Int
    var port: Int
    var pd: Int
    var mr: Int
    var lkey: UInt32
    var rkey: UInt32
    var cq: Int
    var qps: List[Int]  # one RC queue pair per peer, indexed as IbState.peers
    var qpns: List[UInt32]
    var flush_qp: Int
    var flush_mr: Int
    var flush_host: Int
    var flush_lkey: UInt32
    # Scratch the work-request builders write into, reused across posts.
    var wr: Int
    var sge: Int
    var rwr: Int
    var bad: Int
    var wc: Int

    def __init__(out self, var ibv: Ibv):
        self.ibv = ibv^
        self.hca = String("")
        self.ctx = 0
        self.port = 0
        self.pd = 0
        self.mr = 0
        self.lkey = 0
        self.rkey = 0
        self.cq = 0
        self.qps = List[Int]()
        self.qpns = List[UInt32]()
        self.flush_qp = 0
        self.flush_mr = 0
        self.flush_host = 0
        self.flush_lkey = 0
        self.wr = Int(alloc_bytes(SZ_SEND_WR))
        self.sge = Int(alloc_bytes(SZ_SGE))
        self.rwr = Int(alloc_bytes(SZ_RECV_WR))
        self.bad = Int(alloc_bytes(16))
        self.wc = Int(alloc_bytes(SZ_WC * 16))


@always_inline
def _b(addr: Int) -> P8:
    return P8(unsafe_from_address=addr)


def verbs_available() -> Bool:
    """True if libibverbs opens and lists at least one usable port. Used by
    the backend auto-selection in `internode.mojo`; every context it opens is
    closed again before it returns."""
    try:
        var ibv = Ibv()
        var ports = list_ib_ports(ibv, getenv("MOJOCCL_IB_HCA", ""))
        var n = len(ports)
        for i in range(n):
            ibv.close_device(ports[i].ctx)
        return n > 0
    except:
        return False


def vrb_setup(
    gpu_bdf: String,
    local_rank: Int,
    nnodes: Int,
    region: Int,
    region_bytes: Int,
) raises -> VerbsNet:
    """Open an HCA, register the region, create every queue pair (in INIT).

    The QPs cannot reach RTR until the peers' `(qpn, lid, gid)` have been
    gathered, so `vrb_connect_peer` finishes the job.
    """
    var ibv = Ibv()
    var want = getenv("MOJOCCL_IB_HCA", "")
    var ports = list_ib_ports(ibv, want)
    if len(ports) == 0:
        raise Error(
            "mojoccl: no ACTIVE InfiniBand port found"
            + (
                " matching MOJOCCL_IB_HCA=" + want if want.byte_length()
                > 0 else ""
            )
            + "; a multi-node communicator needs one"
        )
    var paths = List[String]()
    for i in range(len(ports)):
        paths.append("/sys/class/infiniband/" + ports[i].name + "/device")
    var pick = pci_pick(paths, gpu_bdf, local_rank)
    ref port = ports[pick]
    # These nodes carry ~10 IB HCAs and every rank opened all of them to
    # read their ports; hold only the one this rank will use.
    for i in range(len(ports)):
        if ports[i].ctx != port.ctx:
            ibv.close_device(ports[i].ctx)

    var v = VerbsNet(ibv^)
    v.hca = String(port.name)
    v.ctx = port.ctx
    v.port = port.port
    try:
        v.pd = v.ibv.alloc_pd(v.ctx)
        if v.pd == 0:
            raise Error("mojoccl: ibv_alloc_pd failed on " + v.hca)
        var ro = getenv("MOJOCCL_IB_RELAXED_ORDERING", "1") != "0"
        var acc = (
            IBV_ACCESS_LOCAL_WRITE
            | IBV_ACCESS_REMOTE_WRITE
            | IBV_ACCESS_REMOTE_READ
        )
        v.mr = v.ibv.reg_mr_relaxed(
            v.pd, region, region_bytes, acc
        ) if ro else v.ibv.reg_mr(v.pd, region, region_bytes, acc)
        if v.mr == 0:
            raise Error(
                "mojoccl: ibv_reg_mr of the "
                + String(region_bytes // (1024 * 1024))
                + " MiB device region failed on "
                + v.hca
                + "; is nvidia_peermem (or the ROCm equivalent) loaded?"
            )
        var mrp = _b(v.mr)
        v.lkey = ldu32(mrp, MR_LKEY)
        v.rkey = ldu32(mrp, MR_RKEY)

        # Host landing pad for the flush read, and the source of the 4-byte
        # credit writes.
        v.flush_host = Int(alloc_bytes(4096))
        v.flush_mr = v.ibv.reg_mr(
            v.pd, v.flush_host, 4096, IBV_ACCESS_LOCAL_WRITE
        )
        if v.flush_mr == 0:
            raise Error("mojoccl: ibv_reg_mr of the flush buffer failed")
        v.flush_lkey = ldu32(_b(v.flush_mr), MR_LKEY)

        v.cq = v.ibv.create_cq(v.ctx, CQ_SIZE)
        if v.cq == 0:
            raise Error("mojoccl: ibv_create_cq failed")

        for _ in range(nnodes - 1):
            var qp = create_rc_qp(
                v.ibv, v.pd, v.cq, SEND_WR_DEPTH, RECV_DEPTH + 8
            )
            # Recorded before `qp_to_init` can raise, so the unwind below
            # destroys it.
            v.qps.append(qp)
            v.qpns.append(qp_number(qp))
            qp_to_init(v.ibv, qp, v.port)
        v.flush_qp = create_rc_qp(v.ibv, v.pd, v.cq, SEND_WR_DEPTH, 8)
        qp_to_init(v.ibv, v.flush_qp, v.port)
    except e:
        vrb_teardown(v)
        raise e
    return v^


def vrb_port_lid(v: VerbsNet) raises -> Int:
    """The port's LID, straight from `ibv_query_port`.

    A nonzero return code (not the same thing as a `try/except` --
    `query_port`'s own C call never raises, it returns an errno) used to be
    silently discarded, reading LID 0 out of `pa`'s zeroed scratch. For the
    self-connected flush QP that 0 is not a sentinel anyone downstream
    checks; it just quietly modifies the flush QP with the wrong address.
    Raise instead.
    """
    var pa = alloc_bytes(SZ_PORT_ATTR)
    var rc = v.ibv.query_port(v.ctx, v.port, pa)
    if rc != 0:
        raise Error("mojoccl: ibv_query_port failed, rc=" + String(rc))
    return ld16(pa, PA_LID)


def vrb_port_mtu(v: VerbsNet) raises -> Int:
    """The port's active MTU (see `vrb_port_lid` for why a failed query
    raises rather than reading 0 out of zeroed scratch)."""
    var pa = alloc_bytes(SZ_PORT_ATTR)
    var rc = v.ibv.query_port(v.ctx, v.port, pa)
    if rc != 0:
        raise Error("mojoccl: ibv_query_port failed, rc=" + String(rc))
    return ld32(pa, PA_ACTIVE_MTU)


# ---- the bootstrap blob --------------------------------------------------
#
#   +0   u64 region base VA
#   +8   u32 rkey
#   +12  u32 lid
#   +16  u32 active_mtu (ibv_mtu enum)
#   +20  u32 number of QPs that follow
#   +24  u32 qpn[MAX_NODES]     -- indexed by the PEER's node
#   +24+4*MAX_NODES  u8 gid[16]

comptime VRB_BLOB_QPN = 24
comptime VRB_BLOB_GID = 24 + 4 * MAX_NODES


def vrb_local_info(v: VerbsNet, blob: P8, nodes: List[Int], region: Int) raises:
    """Fill this rank's half of the bootstrap blob. `nodes[i]` is the node
    queue pair `i` was created for; `region` is the virtual address peers
    write into (InfiniBand RMA always addresses by virtual address)."""
    stu64(blob, 0, UInt64(region))
    stu32(blob, 8, v.rkey)
    stu32(blob, 12, UInt32(vrb_port_lid(v)))
    stu32(blob, 16, UInt32(vrb_port_mtu(v)))
    stu32(blob, 20, UInt32(len(v.qps)))
    for i in range(len(v.qps)):
        stu32(blob, VRB_BLOB_QPN + 4 * nodes[i], v.qpns[i])


def vrb_blob_base(blob: P8) -> Int:
    return Int(blob.unsafe_bitcast[UInt64]()[unsafe_offset=0])


def vrb_blob_key(blob: P8) -> UInt64:
    return UInt64(ldu32(blob, 8))


def vrb_connect_peer(
    mut v: VerbsNet, peer_index: Int, node: Int, my_node: Int, blob: P8
) raises:
    """Move peer `peer_index`'s queue pair to RTS from its blob, then
    pre-post its receive work requests."""
    var lid = ld32(blob, 12)
    var mtu = ld32(blob, 16)
    # The peer's QP for MY node, not for its own.
    var dest_qpn = ldu32(blob, VRB_BLOB_QPN + 4 * my_node)
    var gid = _b(Int(blob) + VRB_BLOB_GID)
    if lid == 0:
        raise Error(
            "mojoccl: peer on node "
            + String(node)
            + " reported LID 0 -- its HCA port is not on an InfiniBand"
            " fabric this library can address"
        )
    qp_to_rtr(
        v.ibv,
        v.qps[peer_index],
        dest_qpn,
        lid,
        min(vrb_port_mtu(v), mtu),
        v.port,
        gid,
        0,
        False,
    )
    qp_to_rts(v.ibv, v.qps[peer_index])
    for _ in range(RECV_DEPTH):
        build_recv_wr(_b(v.rwr), 0)
        if post_recv(v.qps[peer_index], _b(v.rwr), _b(v.bad)) != 0:
            raise Error("mojoccl: ibv_post_recv failed while pre-posting")


def vrb_connect_flush(mut v: VerbsNet) raises:
    """The flush queue pair talks to itself."""
    var gid0 = alloc_bytes(16)
    qp_to_rtr(
        v.ibv,
        v.flush_qp,
        qp_number(v.flush_qp),
        vrb_port_lid(v),
        vrb_port_mtu(v),
        v.port,
        gid0,
        0,
        False,
    )
    qp_to_rts(v.ibv, v.flush_qp)


# ---- the data path -------------------------------------------------------


def vrb_post_payload(
    mut v: VerbsNet,
    peer: Int,
    local_addr: Int,
    nbytes: Int,
    remote_addr: Int,
    remote_key: UInt64,
    immediate: UInt32,
    seq: Int,
) -> Int:
    """One signaled RDMA_WRITE_WITH_IMM: payload and immediate in a single
    operation, which is the thing the libfabric transport has to build out
    of two."""
    build_write_wr(
        _b(v.wr),
        _b(v.sge),
        seq,
        local_addr,
        v.lkey,
        nbytes,
        remote_addr,
        UInt32(remote_key),
        immediate,
        True,
        True,
    )
    return Int(post_send(v.qps[peer], _b(v.wr), _b(v.bad)))


def vrb_post_imm(
    mut v: VerbsNet,
    peer: Int,
    remote_addr: Int,
    remote_key: UInt64,
    nbytes: Int,
    immediate: UInt32,
    seq: Int,
) -> Int:
    """An immediate with nothing to say: an UNSIGNALED short write into the
    peer's credit landing pad. The bytes are never read -- the immediate is
    the message -- but a real address is needed because a zero-length RDMA
    write is not worth relying on across HCAs. Unsignaled because a
    completion here would be indistinguishable from a data send, and the
    data sends are what reclaims the send queue; a failed credit still
    raises a completion with a bad status."""
    build_write_wr(
        _b(v.wr),
        _b(v.sge),
        seq,
        v.flush_host,
        v.flush_lkey,
        nbytes,
        remote_addr,
        UInt32(remote_key),
        immediate,
        True,
        False,
    )
    return Int(post_send(v.qps[peer], _b(v.wr), _b(v.bad)))


def vrb_post_flush(
    mut v: VerbsNet, remote_addr: Int, nbytes: Int, seq: Int
) -> Int:
    build_read_wr(
        _b(v.wr),
        _b(v.sge),
        seq,
        v.flush_host,
        v.flush_lkey,
        nbytes,
        remote_addr,
        v.rkey,
    )
    return Int(post_send(v.flush_qp, _b(v.wr), _b(v.bad)))


def _peer_of_qpn(v: VerbsNet, qpn: UInt32) -> Int:
    for k in range(len(v.qpns)):
        if v.qpns[k] == qpn:
            return k
    return -1


def vrb_poll(mut v: VerbsNet, comps: Int, max_comps: Int) -> Int:
    """Up to `max_comps` completions, translated into `NetCompletion`s.

    Reposting a consumed receive work request happens here rather than in
    the engine: it is the one piece of per-completion bookkeeping that is
    purely about verbs. A receive slot lost is a later RNR the peer retries
    forever (IB_RNR_RETRY = 7), i.e. a silent hang, so a failed repost is
    reported as a failed completion.
    """
    var n = Int(poll_cq(v.cq, min(max_comps, 16), _b(v.wc)))
    if n < 0:
        var c = NetCompletion()
        c.status = 2000 - n
        Pointer[NetCompletion, MutAnyOrigin](unsafe_from_address=comps)[] = c^
        return 1
    for i in range(n):
        var w = _b(v.wc + i * SZ_WC)
        var c = NetCompletion()
        if Int32(ld32(w, WC_STATUS)) != IBV_WC_SUCCESS:
            c.status = 1000 + ld32(w, WC_STATUS) * 1000 + ld32(w, WC_VENDOR_ERR)
        else:
            var op = Int32(ld32(w, WC_OPCODE))
            c.peer = _peer_of_qpn(v, ldu32(w, WC_QP_NUM))
            if op == IBV_WC_RECV_RDMA_WITH_IMM:
                c.kind = NC_RECV
                c.imm = be32(ldu32(w, WC_IMM_DATA))
                if c.peer >= 0:
                    build_recv_wr(_b(v.rwr), 0)
                    if post_recv(v.qps[c.peer], _b(v.rwr), _b(v.bad)) != 0:
                        c.status = 5
            elif op == IBV_WC_RDMA_WRITE:
                # Only data writes are signaled (credits are not), and their
                # wr_id is the exchange number.
                c.kind = NC_SEND
                c.wr_id = ld64(w, WC_WR_ID)
            elif op == IBV_WC_RDMA_READ:
                c.kind = NC_FLUSH
            else:
                c.kind = NC_OTHER
        Pointer[NetCompletion, MutAnyOrigin](
            unsafe_from_address=comps + i * size_of[NetCompletion]()
        )[] = (c^)
    return n


def vrb_teardown(mut v: VerbsNet):
    """Release every ibverbs resource `vrb_setup` may have created -- shared
    by teardown of a live communicator and by `vrb_setup`'s own failure path
    (a later step raised after an earlier one succeeded), which is why every
    field is zero-guarded."""
    try:
        for i in range(len(v.qps)):
            v.ibv.destroy_qp(v.qps[i])
        v.qps.clear()
        if v.flush_qp != 0:
            v.ibv.destroy_qp(v.flush_qp)
            v.flush_qp = 0
        if v.cq != 0:
            v.ibv.destroy_cq(v.cq)
            v.cq = 0
        if v.flush_mr != 0:
            v.ibv.dereg_mr(v.flush_mr)
            v.flush_mr = 0
        if v.mr != 0:
            v.ibv.dereg_mr(v.mr)
            v.mr = 0
        if v.pd != 0:
            v.ibv.dealloc_pd(v.pd)
            v.pd = 0
        if v.ctx != 0:
            v.ibv.close_device(v.ctx)
            v.ctx = 0
    except e:
        # Best-effort teardown: nothing is left to undo, but say what failed.
        print("mojoccl: verbs teardown step failed (ignored):", e)
