"""torchrun worker for tests/test_distributed.py (not a pytest module).

Runs one validation mode per invocation and exits non-zero on failure so the
parent test only has to check the return code. Keep this file importable
without a GPU: everything device-touching happens inside main().
"""

# ruff: noqa: E402 -- use_local_rank_gpu() must run before torch/MAX initialize
import os
import sys

# One GPU per rank, decided before anything can initialize CUDA/MAX.
from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.distributed import nccl
from torch_mojo_backend.distributed.process_group import (
    MojoProcessGroup,
    _nccl_dtype,
    _nccl_red_op,
    _ptr_of,
)
from torch_mojo_backend.mojo_device import torch_mojo_device_module

# mojoccl (torch_mojo_backend/distributed/mojoccl) implements AllReduce/
# Broadcast/AllGather only -- Reduce/ReduceScatter/Send/Recv return
# ncclInvalidUsage (DDP on GPT-2 needs only the first three). Skip the
# checks that need the unimplemented ops rather than fail on them.
_MOJO_CCL = os.environ.get("TORCH_MOJO_BACKEND_CCL") == "mojo"


class ElemwiseNet(torch.nn.Module):
    """Matmul-free so it runs even where GEMM routes are unavailable."""

    def __init__(self, width: int):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(width))
        self.b = torch.nn.Parameter(torch.randn(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gelu, not relu: relu's backward (threshold_backward) is not
        # implemented in mojo eager mode; gelu_backward is.
        return torch.nn.functional.gelu(x * self.w + self.b)


def _check(failures: list[str], name: str, ok: bool):
    rank = dist.get_rank()
    print(f"[rank {rank}] {'OK  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        failures.append(name)


def run_collectives(failures: list[str]):
    rank = dist.get_rank()
    world = dist.get_world_size()
    expected = world * (world + 1) / 2

    t = torch.full((1024,), float(rank + 1), device="mojo")
    dist.all_reduce(t)
    _check(failures, "allreduce.f32", bool((t.cpu() == expected).all()))

    tb = torch.full((257,), float(rank + 1), device="mojo", dtype=torch.bfloat16)
    dist.all_reduce(tb)
    _check(failures, "allreduce.bf16", bool((tb.float().cpu() == expected).all()))

    i64 = torch.tensor([rank + 1], dtype=torch.int64, device="mojo")
    dist.all_reduce(i64)
    _check(failures, "allreduce.int64", i64.cpu().item() == world * (world + 1) // 2)

    b = torch.full((33,), float(rank), device="mojo")
    dist.broadcast(b, src=0)
    _check(failures, "broadcast", bool((b.cpu() == 0.0).all()))

    outs = [torch.zeros(5, device="mojo") for _ in range(world)]
    mine = torch.full((5,), float(rank), device="mojo")
    dist.all_gather(outs, mine)
    ok = all((outs[r].cpu() == float(r)).all().item() for r in range(world))
    _check(failures, "all_gather", ok)

    flat = torch.zeros(world * 5, device="mojo")
    dist.all_gather_into_tensor(flat, mine)
    ok = all(
        (flat.cpu()[r * 5 : (r + 1) * 5] == float(r)).all().item() for r in range(world)
    )
    _check(failures, "all_gather_into_tensor", ok)

    if _MOJO_CCL:
        print(
            f"[rank {rank}] SKIP reduce_scatter_tensor "
            "(mojoccl does not implement ncclReduceScatter)",
            flush=True,
        )
    else:
        src = torch.arange(world * 3, dtype=torch.float32, device="mojo")
        out = torch.zeros(3, device="mojo")
        dist.reduce_scatter_tensor(out, src)
        exp = (
            torch.arange(world * 3, dtype=torch.float32)[rank * 3 : (rank + 1) * 3]
            * world
        )
        _check(failures, "reduce_scatter_tensor", bool((out.cpu() == exp).all()))

    objs: list[dict[str, int] | None] = [None] * world
    dist.all_gather_object(objs, {"rank": rank})
    first, last = objs[0], objs[-1]
    _check(
        failures,
        "all_gather_object",
        first is not None
        and last is not None
        and first["rank"] == 0
        and last["rank"] == world - 1,
    )

    if world > 1:
        if _MOJO_CCL:
            print(
                f"[rank {rank}] SKIP send_recv_ring "
                "(mojoccl does not implement ncclSend/ncclRecv)",
                flush=True,
            )
        else:
            s = torch.full((7,), float(rank), device="mojo")
            r = torch.zeros(7, device="mojo")
            if rank % 2 == 0:
                dist.send(s, (rank + 1) % world)
                dist.recv(r, (rank - 1) % world)
            else:
                dist.recv(r, (rank - 1) % world)
                dist.send(s, (rank + 1) % world)
            _check(
                failures,
                "send_recv_ring",
                bool((r.cpu() == float((rank - 1) % world)).all()),
            )

    dist.barrier()


def run_ddp_parity(failures: list[str]):
    rank = dist.get_rank()
    world = dist.get_world_size()
    width = 4096
    per_rank = 16

    torch.manual_seed(1234 + rank)  # deliberately different per rank...
    model = ElemwiseNet(width).to("mojo")
    ddp = DDP(model, broadcast_buffers=False)
    torch.manual_seed(1234)
    reference = ElemwiseNet(width)
    # ...so a passing check proves DDP's construction-time broadcast synced
    # rank 0's weights everywhere.
    _check(
        failures,
        "ddp.initial_broadcast",
        torch.equal(ddp.module.w.detach().cpu(), reference.w.detach()),
    )

    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-2)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-2)
    torch.manual_seed(999)
    full_batch = torch.randn(world * per_rank, width)
    for _ in range(3):
        shard = full_batch[rank * per_rank : (rank + 1) * per_rank].to("mojo")
        loss = ddp(shard).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        ref_loss = reference(full_batch).pow(2).mean()
        ref_optimizer.zero_grad(set_to_none=True)
        ref_loss.backward()
        ref_optimizer.step()

    _check(
        failures,
        "ddp.step_parity",
        torch.allclose(ddp.module.w.detach().cpu(), reference.w.detach(), atol=1e-5)
        and torch.allclose(
            ddp.module.b.detach().cpu(), reference.b.detach(), atol=1e-5
        ),
    )
    dist.barrier()


def run_lazy_fence(failures: list[str]):
    """The comm-stream collective's result is fenced onto the default stream
    lazily, at its first consumer (mojo_device/comm_fence.py).

    256 MB per collective on purpose: the fence is only under test while the
    allreduce is still running when the host reaches the consumer a few
    microseconds later. A few-KB collective would have finished either way
    and every assertion below would pass with no fence at all.

    Measured against a build with ``comm_fence.mark_pending`` disarmed, (c)
    is the check that catches the missing fence — (a) and (b) allocate a
    256 MB destination first, which is usually long enough for the collective
    to land anyway. They stay because they are the contract users write
    against; (c) is the one with teeth.
    """
    rank = dist.get_rank()
    world = dist.get_world_size()
    expected = float(world * (world + 1) // 2)
    numel = 64 * 1024 * 1024

    # (a) host read straight after the collective.
    a = torch.full((numel,), float(rank + 1), device="mojo")
    dist.all_reduce(a)
    _check(failures, "lazy_fence.host_read", bool((a.cpu() == expected).all()))
    del a

    # (b) device consumer of the result, read back through it.
    b = torch.full((numel,), float(rank + 1), device="mojo")
    dist.all_reduce(b)
    doubled = b + b
    _check(
        failures,
        "lazy_fence.device_consumer",
        bool((doubled.cpu() == 2 * expected).all()),
    )
    del b, doubled

    # (c) write into a VIEW of the reduced buffer. Unfenced, the fill lands
    # first and the collective's output overwrites it.
    c = torch.full((numel,), float(rank + 1), device="mojo")
    dist.all_reduce(c)
    c.narrow(0, 0, 1024).fill_(-7.0)
    out = c.cpu()
    _check(
        failures,
        "lazy_fence.view_write",
        bool((out[:1024] == -7.0).all()) and bool((out[1024:] == expected).all()),
    )
    del c, out

    dist.barrier()


# ---------------------------------------------------------------------------
# stress: mojoccl-only regression coverage ported from the kernel harness
# (/home/gabriel/ddp_work/mojo_collectives/kernel/harness.mojo `mix` and
# `verify`) into the real torchrun/c10d path -- vendor NCCL/RCCL never see
# this mode (nothing here would fail against them; it exists to catch
# mojoccl-specific ABI bugs like the interleaving bug RESULTS.md §6 found).
# ---------------------------------------------------------------------------

_VERIFY_DTYPES: list[torch.dtype] = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.int64,
]
_INT_DTYPES = (torch.int32, torch.int64)


def _fill(dtype: torch.dtype, rank: int, n: int) -> torch.Tensor:
    """Deterministic per-(rank, index) values, no RNG, on the CPU.

    Integers land in [-128, 127]: exact CPU references, no overflow summing
    up to 8 ranks. Floats are k/128 with |k| <= 128: exactly representable
    in fp16/bf16/fp32 (<= 8 significant bits), so only the cross-rank SUM can
    round, never the fill itself -- the same trick harness.mojo's _val_f64
    uses and for the same reason.
    """
    idx = torch.arange(n, dtype=torch.int64)
    h = (rank * 1000003 + idx * 2654435761) & 0xFF
    if dtype in _INT_DTYPES:
        return (h - 128).to(dtype)
    return ((h.to(torch.float64) - 128.0) / 128.0).to(dtype)


def _reference(dtype: torch.dtype, world: int, n: int, avg: bool) -> torch.Tensor:
    """The CPU sum (or, for floats, AVG) of `_fill(dtype, r, n)` over every
    rank r, accumulated wide (int64 / float64) then rounded once to `dtype`
    -- mojoccl ignores `scale` for integer dtypes (collectives_kernels.mojo's
    allreduce docstring: "ignored for integer dtypes"), so AVG on int32/int64
    is SUM, matching what the library actually computes, not what real NCCL's
    ncclAvg would do.
    """
    acc_dtype = torch.int64 if dtype in _INT_DTYPES else torch.float64
    acc = torch.zeros(n, dtype=acc_dtype)
    for r in range(world):
        acc += _fill(dtype, r, n).to(acc_dtype)
    if avg and dtype not in _INT_DTYPES:
        acc = acc / world
    return acc.to(dtype)


def _matches(dtype: torch.dtype, got: torch.Tensor, want: torch.Tensor) -> bool:
    """Exact for float32/int32/int64; half a bf16 ulp for fp16/bf16 -- the
    same tolerance harness.mojo's verify_ar uses, validated there against
    this exact kernel (RESULTS.md §7)."""
    if dtype in (torch.float16, torch.bfloat16):
        got64 = got.to(torch.float64)
        want64 = want.to(torch.float64)
        tol = 0.004 * torch.clamp(want64.abs(), min=1.0)
        return bool((got64 - want64).abs().le(tol).all())
    return torch.equal(got, want)


def _pg_comm(tensor: torch.Tensor) -> tuple[nccl.NcclComm, int]:
    """(NcclComm, default-stream handle) for `tensor`'s device, bypassing
    dist.all_reduce's always-in-place call shape (process_group.py always
    passes the same pointer as sendbuff and recvbuff) so the verify matrix
    below can also exercise a genuine sendbuff != recvbuff ncclAllReduce.
    `dist.group.WORLD` is this backend's own `MojoProcessGroup` instance
    directly -- no C++ PG wraps it (see MojoProcessGroup.__init__).

    Callers MUST synchronize the device before enqueuing on the returned
    stream if a prior collective on this comm went through dist.all_reduce:
    that path can run on a side comm-stream (TORCH_MOJO_BACKEND_COMM_STREAM,
    the default), and NCCL/RCCL-style comms require every op on a
    communicator to execute in issue order across ranks
    (process_group.py's `_fence_default` exists for exactly this) -- this
    raw call bypasses that fencing, so the caller re-establishes the
    ordering with a full sync instead.
    """
    pg = dist.group.WORLD
    assert isinstance(pg, MojoProcessGroup)
    comm, stream, _ = pg._device_state(tensor)
    return comm, stream


def _stress_mix(failures: list[str], rank: int, world: int):
    """Port of `harness.mojo mix`: several hundred generations interleaving a
    4-byte one-shot allreduce, a 27 MiB two-shot allreduce, a broadcast and
    an allgather, all sharing mojoccl's staging arena with different
    layouts -- the pattern that found the interleaving bug (RESULTS.md §6).
    The two allreduces run in place with AVG, which is idempotent once every
    rank holds the mean, so a single corrupted generation anywhere in the
    200 rounds (800 generations) still shows up in the final check. The
    broadcast/allgather payloads are cheap (2 KiB, 8 B/rank) and change every
    round, so they are verified every round instead.
    """
    rounds = 200
    big_n = (27 * 1024 * 1024) // 4  # 27 MiB of float32

    a_small = torch.full((1,), float(rank + 1), device="mojo")
    a_big = torch.full((big_n,), float(rank + 1), device="mojo")
    bcast_n = 500  # 2000 bytes of float32
    bcast_buf = torch.zeros(bcast_n, device="mojo")
    ag_per_rank = 2
    ag_in = torch.zeros(ag_per_rank, device="mojo")
    ag_out = torch.zeros(world * ag_per_rank, device="mojo")

    bcast_ok = True
    gather_ok = True
    for r in range(rounds):
        dist.all_reduce(a_small, op=dist.ReduceOp.AVG)
        dist.all_reduce(a_big, op=dist.ReduceOp.AVG)

        bval = float((r % 97) + 1) * 0.25
        if rank == 0:
            bcast_buf.fill_(bval)
        dist.broadcast(bcast_buf, src=0)
        if not bool((bcast_buf.cpu() == bval).all()):
            bcast_ok = False

        ag_in.fill_(float(r * 1000 + rank))
        dist.all_gather_into_tensor(ag_out, ag_in)
        want = torch.cat(
            [torch.full((ag_per_rank,), float(r * 1000 + p)) for p in range(world)]
        )
        if not torch.equal(ag_out.cpu(), want):
            gather_ok = False

    _check(failures, "stress.mix.broadcast", bcast_ok)
    _check(failures, "stress.mix.allgather", gather_ok)

    expected_mean = float(world + 1) / 2.0
    small_ok = bool(torch.allclose(a_small.cpu(), torch.full((1,), expected_mean)))
    big_ok = bool(torch.allclose(a_big.cpu(), torch.full((big_n,), expected_mean)))
    _check(failures, "stress.mix.allreduce_4B", small_ok)
    _check(failures, "stress.mix.allreduce_27MiB", big_ok)
    dist.barrier()


def _stress_verify_matrix(failures: list[str], rank: int, world: int):
    """dtypes x ragged sizes x {SUM, AVG} x {in place, out of place}.

    Sizes: 1, 1003 (not a multiple of the 16-byte vector width), a size one
    element short of the region cap, and one element past it (forces the ABI
    layer's chunking loop). MOJOCCL_REGION_MB is set small for this mode (see
    test_distributed.py) specifically so "past the cap" stays a small tensor.
    """
    cap_bytes = int(os.environ.get("MOJOCCL_REGION_MB", "256")) * 1024 * 1024

    for dtype in _VERIFY_DTYPES:
        item_bytes = torch.zeros((), dtype=dtype).element_size()
        max_elems = cap_bytes // item_bytes
        sizes = [1, 1003, max(1, (cap_bytes - 1) // item_bytes), max_elems + 1]
        for n in sizes:
            for op, avg in ((dist.ReduceOp.SUM, False), (dist.ReduceOp.AVG, True)):
                want = _reference(dtype, world, n, avg)
                tag = f"stress.verify.{dtype}.n{n}.{op.name}"

                x = _fill(dtype, rank, n).to("mojo")
                dist.all_reduce(x, op=op)
                _check(failures, f"{tag}.inplace", _matches(dtype, x.cpu(), want))

                # Full sync: the in-place step above may have run on the
                # comm side-stream: see _pg_comm's docstring.
                torch_mojo_device_module.synchronize()
                src = _fill(dtype, rank, n).to("mojo")
                dst = torch.zeros(n, dtype=dtype, device="mojo")
                comm, stream = _pg_comm(src)
                comm.all_reduce(
                    _ptr_of(src),
                    _ptr_of(dst),
                    n,
                    _nccl_dtype(dtype),
                    _nccl_red_op(op),
                    stream,
                )
                _check(failures, f"{tag}.outofplace", _matches(dtype, dst.cpu(), want))
    dist.barrier()


def _stress_broadcast_allgather_edges(failures: list[str], rank: int, world: int):
    """Broadcast from a non-zero root; allgather with a per-rank size that is
    not a multiple of 16 bytes (12 bytes of float32)."""
    root = world - 1
    n = 777
    buf = torch.zeros(n, device="mojo")
    if rank == root:
        buf.copy_(_fill(torch.float32, root, n).to("mojo"))
    dist.broadcast(buf, src=root)
    want_bcast = _fill(torch.float32, root, n)
    _check(
        failures, "stress.broadcast_nonzero_root", torch.equal(buf.cpu(), want_bcast)
    )

    per_rank = 3  # 12 bytes -- not a multiple of 16
    mine = _fill(torch.float32, rank, per_rank).to("mojo")
    out = torch.zeros(world * per_rank, device="mojo")
    dist.all_gather_into_tensor(out, mine)
    want_gather = torch.cat([_fill(torch.float32, r, per_rank) for r in range(world)])
    _check(
        failures, "stress.allgather_non16_per_rank", torch.equal(out.cpu(), want_gather)
    )
    dist.barrier()


def _stress_avg_overflow(failures: list[str], rank: int, world: int):
    """AVG over half dtypes must never hold the unscaled sum in the wire dtype.

    Every rank contributes 32768: `world` of them sum past fp16's 65504 while
    the average is exact, so a path that scales after a narrow store reads
    inf (NCCL pre-multiplies, `ncclDevPreMulSum`). Two sizes: 1003 elements
    for the small-message kernels, and 24M elements -- 48 MiB of fp16, the
    NVLS floor on one node and the pipelined hierarchical path on several.
    """
    for dtype in (torch.float16, torch.bfloat16):
        for n in (1003, 24 * 1024 * 1024):
            x = torch.full((n,), 32768.0, dtype=dtype, device="mojo")
            dist.all_reduce(x, op=dist.ReduceOp.AVG)
            want = torch.full((n,), 32768.0, dtype=dtype)
            _check(
                failures,
                f"stress.avg_overflow.{dtype}.n{n}",
                torch.equal(x.cpu(), want),
            )
    dist.barrier()


def run_stress(failures: list[str]):
    rank = dist.get_rank()
    world = dist.get_world_size()
    if not _MOJO_CCL:
        print(
            f"[rank {rank}] SKIP stress (mojoccl-only regression coverage)", flush=True
        )
        return
    _stress_mix(failures, rank, world)
    _stress_verify_matrix(failures, rank, world)
    _stress_broadcast_allgather_edges(failures, rank, world)
    _stress_avg_overflow(failures, rank, world)


def main():
    mode = sys.argv[1]

    register_mojo_devices()
    dist.init_process_group(backend="mojo", timeout=datetime.timedelta(seconds=300))
    failures: list[str] = []
    if mode == "collectives":
        run_collectives(failures)
    elif mode == "ddp_parity":
        run_ddp_parity(failures)
    elif mode == "lazy_fence":
        run_lazy_fence(failures)
    elif mode == "stress":
        run_stress(failures)
    else:
        raise ValueError(f"unknown mode {mode}")
    dist.destroy_process_group()
    if failures:
        print(f"[rank {os.environ['RANK']}] FAILURES: {failures}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
