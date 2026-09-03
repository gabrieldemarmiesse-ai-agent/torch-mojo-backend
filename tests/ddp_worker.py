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

    src = torch.arange(world * 3, dtype=torch.float32, device="mojo")
    out = torch.zeros(3, device="mojo")
    dist.reduce_scatter_tensor(out, src)
    exp = (
        torch.arange(world * 3, dtype=torch.float32)[rank * 3 : (rank + 1) * 3] * world
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
    else:
        raise ValueError(f"unknown mode {mode}")
    dist.destroy_process_group()
    if failures:
        print(f"[rank {os.environ['RANK']}] FAILURES: {failures}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
