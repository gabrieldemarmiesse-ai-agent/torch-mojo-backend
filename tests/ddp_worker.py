"""torchrun worker for tests/test_distributed.py (not a pytest module).

Runs one validation mode per invocation and exits non-zero on failure so the
parent test only has to check the return code. Keep this file importable
without a GPU: everything device-touching happens inside main().
"""

import os
import sys

if "LOCAL_RANK" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    # One GPU per rank, decided before anything can initialize CUDA/MAX.
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
    os.environ.setdefault("HIP_VISIBLE_DEVICES", os.environ["LOCAL_RANK"])

import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class ElemwiseNet(torch.nn.Module):
    """Matmul-free so it runs even where GEMM routes are unavailable."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(width))
        self.b = torch.nn.Parameter(torch.randn(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.w + self.b).relu()


def _check(failures: list[str], name: str, ok: bool) -> None:
    rank = dist.get_rank()
    print(f"[rank {rank}] {'OK  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        failures.append(name)


def run_collectives(failures: list[str]) -> None:
    rank = dist.get_rank()
    world = dist.get_world_size()
    expected = world * (world + 1) / 2

    t = torch.full((1024,), float(rank + 1), device="mojo")
    dist.all_reduce(t)
    _check(failures, "allreduce.f32", (t.cpu() == expected).all().item())

    tb = torch.full((257,), float(rank + 1), device="mojo", dtype=torch.bfloat16)
    dist.all_reduce(tb)
    _check(failures, "allreduce.bf16", (tb.float().cpu() == expected).all().item())

    i64 = torch.tensor([rank + 1], dtype=torch.int64, device="mojo")
    dist.all_reduce(i64)
    _check(failures, "allreduce.int64", i64.cpu().item() == world * (world + 1) // 2)

    b = torch.full((33,), float(rank), device="mojo")
    dist.broadcast(b, src=0)
    _check(failures, "broadcast", (b.cpu() == 0.0).all().item())

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
    exp = torch.arange(world * 3, dtype=torch.float32)[rank * 3 : (rank + 1) * 3] * world
    _check(failures, "reduce_scatter_tensor", (out.cpu() == exp).all().item())

    objs: list[object] = [None] * world
    dist.all_gather_object(objs, {"rank": rank})
    _check(
        failures,
        "all_gather_object",
        objs[0]["rank"] == 0 and objs[-1]["rank"] == world - 1,
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
            failures, "send_recv_ring", (r.cpu() == float((rank - 1) % world)).all().item()
        )

    dist.barrier()


def run_ddp_parity(failures: list[str]) -> None:
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
        and torch.allclose(ddp.module.b.detach().cpu(), reference.b.detach(), atol=1e-5),
    )
    dist.barrier()


def main() -> None:
    mode = sys.argv[1]
    from torch_mojo_backend import register_mojo_devices

    register_mojo_devices()
    dist.init_process_group(backend="mojo", timeout=datetime.timedelta(seconds=300))
    failures: list[str] = []
    if mode == "collectives":
        run_collectives(failures)
    elif mode == "ddp_parity":
        run_ddp_parity(failures)
    else:
        raise ValueError(f"unknown mode {mode}")
    dist.destroy_process_group()
    if failures:
        print(f"[rank {os.environ['RANK']}] FAILURES: {failures}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
