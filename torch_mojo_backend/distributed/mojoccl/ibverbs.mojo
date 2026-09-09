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
from std.memory.alloc import unsafe_alloc

comptime P8 = Pointer[UInt8, MutAnyOrigin]

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


@always_inline
def alloc_bytes(n: Int) -> P8:
    var p = P8(unsafe_from_address=Int(unsafe_alloc[UInt8](n)))
    for i in range(n):
        p[unsafe_offset=i] = 0
    return p


@always_inline
def _st8(p: P8, off: Int, v: UInt8):
    p[unsafe_offset=off] = v


@always_inline
def _st16(p: P8, off: Int, v: UInt16):
    p.unsafe_bitcast[UInt16]()[unsafe_offset=off // 2] = v


@always_inline
def _st32(p: P8, off: Int, v: Int32):
    p.unsafe_bitcast[Int32]()[unsafe_offset=off // 4] = v


@always_inline
def _stu32(p: P8, off: Int, v: UInt32):
    p.unsafe_bitcast[UInt32]()[unsafe_offset=off // 4] = v


@always_inline
def _st64(p: P8, off: Int, v: Int):
    p.unsafe_bitcast[Int64]()[unsafe_offset=off // 8] = Int64(v)


@always_inline
def ld8(p: P8, off: Int) -> Int:
    return Int(p[unsafe_offset=off])


@always_inline
def ld16(p: P8, off: Int) -> Int:
    return Int(p.unsafe_bitcast[UInt16]()[unsafe_offset=off // 2])


@always_inline
def ld32(p: P8, off: Int) -> Int:
    return Int(p.unsafe_bitcast[Int32]()[unsafe_offset=off // 4])


@always_inline
def ldu32(p: P8, off: Int) -> UInt32:
    return p.unsafe_bitcast[UInt32]()[unsafe_offset=off // 4]


@always_inline
def ld64(p: P8, off: Int) -> Int:
    return Int(p.unsafe_bitcast[Int64]()[unsafe_offset=off // 8])


@always_inline
def _as_fn[F: TrivialRegisterPassable](addr: Int) -> F:
    """A callable from a raw function address.

    Same shape as `std.ffi._get_dylib_function`'s cache hit: bitcast a
    pointer TO the address variable, then load. A `Pointer.unsafe_bitcast`
    straight to a function type is rejected (function types are not
    `AnyType`), and the type must be `thin` -- a plain `def(...)` type is a
    closure trait, not a function pointer.
    """
    var a = addr
    return Pointer(to=a).unsafe_bitcast[F]()[]


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
        var mr: Int
        try:
            mr = Int(
                self.lib.get_function[Int64]("ibv_reg_mr_iova2")(
                    pd,
                    addr,
                    UInt64(length),
                    UInt64(addr),
                    UInt32(access | IBV_ACCESS_RELAXED_ORDERING),
                )
            )
        except:
            # IBVERBS_1.8 absent: an old rdma-core, RO simply unavailable
            mr = 0
        if mr != 0:
            return mr
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
    var f = _as_fn[def(Int, P8, P8) thin abi("C") -> Int32](
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
    var f = _as_fn[def(Int, P8, P8) thin abi("C") -> Int32](
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
    var f = _as_fn[def(Int, Int32, P8) thin abi("C") -> Int32](
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
    _st64(sge, SGE_ADDR, local_addr)
    _stu32(sge, SGE_LENGTH, UInt32(nbytes))
    _stu32(sge, SGE_LKEY, lkey)
    for i in range(SZ_SEND_WR):
        wr[unsafe_offset=i] = 0
    _st64(wr, WR_ID, wr_id)
    _st64(wr, WR_NEXT, 0)
    _st64(wr, WR_SG_LIST, Int(sge))
    _st32(wr, WR_NUM_SGE, 1)
    _st32(
        wr,
        WR_OPCODE,
        IBV_WR_RDMA_WRITE_WITH_IMM if with_imm else IBV_WR_RDMA_WRITE,
    )
    _st32(wr, WR_SEND_FLAGS, IBV_SEND_SIGNALED if signaled else 0)
    # imm_data is __be32 on both sides (nccl:...:p2p.cc:157 htobe32, :653
    # be32toh). Byte-swapped here so the receiver's plain load reads it back.
    _stu32(wr, WR_IMM_DATA, _bswap32(immediate))
    _st64(wr, WR_RDMA_REMOTE_ADDR, remote_addr)
    _stu32(wr, WR_RDMA_RKEY, rkey)


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
    _st64(sge, SGE_ADDR, local_addr)
    _stu32(sge, SGE_LENGTH, UInt32(nbytes))
    _stu32(sge, SGE_LKEY, lkey)
    for i in range(SZ_SEND_WR):
        wr[unsafe_offset=i] = 0
    _st64(wr, WR_ID, wr_id)
    _st64(wr, WR_SG_LIST, Int(sge))
    _st32(wr, WR_NUM_SGE, 1)
    _st32(wr, WR_OPCODE, IBV_WR_RDMA_READ)
    _st32(wr, WR_SEND_FLAGS, IBV_SEND_SIGNALED)
    _st64(wr, WR_RDMA_REMOTE_ADDR, remote_addr)
    _stu32(wr, WR_RDMA_RKEY, rkey)


def build_recv_wr(wr: P8, wr_id: Int):
    """An EMPTY receive WR (`sg_list = NULL, num_sge = 0`), which is what
    NCCL posts too (nccl:src/transport/net_ib/common.cc:88): its only job is
    to be consumed by an incoming RDMA_WRITE_WITH_IMM so a completion with
    the immediate appears on the CQ. The payload went straight to the
    registered region."""
    for i in range(SZ_RECV_WR):
        wr[unsafe_offset=i] = 0
    _st64(wr, WR_ID, wr_id)


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
    _st64(ia, QIA_SEND_CQ, cq)
    _st64(ia, QIA_RECV_CQ, cq)
    _st32(ia, QIA_MAX_SEND_WR, Int32(max_send_wr))
    _st32(ia, QIA_MAX_RECV_WR, Int32(max_recv_wr))
    _st32(ia, QIA_MAX_SEND_SGE, 1)
    _st32(ia, QIA_MAX_RECV_SGE, 1)
    _st32(ia, QIA_QP_TYPE, IBV_QPT_RC)
    var qp = ibv.create_qp(pd, ia)
    if qp == 0:
        raise Error("mojoccl: ibv_create_qp failed")
    return qp


def qp_number(qp: Int) -> UInt32:
    return ldu32(P8(unsafe_from_address=qp), QP_QP_NUM)


def qp_to_init(ibv: Ibv, qp: Int, port: Int) raises:
    var a = alloc_bytes(SZ_QP_ATTR)
    _st32(a, QA_QP_STATE, IBV_QPS_INIT)
    _st16(a, QA_PKEY_INDEX, 0)
    _st8(a, QA_PORT_NUM, UInt8(port))
    # Both directions on one QP pair: this rank writes into the peer's
    # region and the peer writes into this one, so REMOTE_WRITE is needed
    # on both ends (NCCL splits it because its QPs are one-directional).
    _st32(
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
    _st32(a, QA_QP_STATE, IBV_QPS_RTR)
    _st32(a, QA_PATH_MTU, Int32(mtu))
    _stu32(a, QA_DEST_QP_NUM, dest_qpn)
    _st32(a, QA_RQ_PSN, IB_PSN)
    _st8(a, QA_MAX_DEST_RD_ATOMIC, 1)
    _st8(a, QA_MIN_RNR_TIMER, IB_MIN_RNR_TIMER)
    _st16(a, QA_AH_DLID, UInt16(dlid))
    _st8(a, QA_AH_SL, 0)
    _st8(a, QA_AH_PORT_NUM, UInt8(port))
    if global_route:
        # Only when the two ports are on different IB subnets -- NCCL's rule
        # (connect.cc:456-458); a single-subnet fabric never takes this.
        _st8(a, QA_AH_IS_GLOBAL, 1)
        for i in range(16):
            a[unsafe_offset=QA_AH_DGID + i] = remote_gid[unsafe_offset=i]
        _st8(a, QA_AH_SGID_INDEX, UInt8(local_gid_index))
        _st8(a, QA_AH_HOP_LIMIT, IB_HOP_LIMIT)
    var rc = ibv.modify_qp(qp, a, QP_MASK_RTR)
    if rc != 0:
        raise Error("mojoccl: ibv_modify_qp(RTR) failed, rc=" + String(rc))


def qp_to_rts(ibv: Ibv, qp: Int) raises:
    var a = alloc_bytes(SZ_QP_ATTR)
    _st32(a, QA_QP_STATE, IBV_QPS_RTS)
    _st32(a, QA_SQ_PSN, IB_PSN)
    _st8(a, QA_TIMEOUT, IB_TIMEOUT)
    _st8(a, QA_RETRY_CNT, IB_RETRY_CNT)
    _st8(a, QA_RNR_RETRY, IB_RNR_RETRY)
    _st8(a, QA_MAX_RD_ATOMIC, 1)
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
