"""mojoccl's NVLS allreduce, at world sizes and sizes ddp_worker misses.

`ddp_worker stress` verifies the multicast path at 8 ranks and 256 MiB.
This runs the same kind of check at whatever `WORLD_SIZE` torchrun gives it
-- 2 and 4 are the interesting ones, since the reducer/copy block split and
the per-rank slice arithmetic are both functions of `world` -- over sizes
that straddle the `MOJOCCL_NVLS_MIN_MB` dispatch threshold and the staging
arena, including one element past it so the ABI layer's chunking loop runs.

    TORCH_MOJO_BACKEND_CCL=mojo torchrun --nproc-per-node=2 tests/nvls_check.py

Exits nonzero on the first mismatch. It is not a pytest module (it needs
torchrun, like tests/ddp_worker.py) and pytest does not collect it.
"""

# ruff: noqa: E402 -- use_local_rank_gpu() must run before torch/MAX initialize
import os

# One GPU per rank, decided before anything can initialize CUDA/MAX.
from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import torch
import torch.distributed as dist

from torch_mojo_backend import register_mojo_devices

register_mojo_devices()

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])


def main():
    dist.init_process_group(backend="mojo")
    cap = int(os.environ.get("MOJOCCL_REGION_MB", "256")) * 1024 * 1024
    bad = 0
    # (op, fill, expected). The third case fills every rank with 32768: the
    # unscaled sum overflows fp16 (world x 32768 > 65504) while the average is
    # exact, so it fails unless AVG's 1/world is applied to the inputs before
    # the switch narrows the sum (NCCL's PreMulSum), and it is the reason
    # fp16 is in the dtype list.
    cases = (
        (dist.ReduceOp.SUM, float(RANK + 1), WORLD * (WORLD + 1) / 2),
        (dist.ReduceOp.AVG, float(RANK + 1), (WORLD + 1) / 2),
        (dist.ReduceOp.AVG, 32768.0, 32768.0),
    )
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        item = torch.zeros((), dtype=dtype).element_size()
        for n in (
            47 * 1024 * 1024 // item,  # just below the NVLS floor
            48 * 1024 * 1024 // item,  # just at it
            64 * 1024 * 1024 // item + 3,  # ragged, above it
            2 * cap // item + 1,  # one element past the NVLS staging arena
        ):
            for op, fill, want in cases:
                x = torch.full((n,), fill, dtype=dtype, device="mojo")
                dist.all_reduce(x, op=op)
                got = x.cpu()
                ok = bool((got == torch.tensor(want, dtype=dtype)).all())
                tag = f"world{WORLD}.{dtype}.n{n}.{op.name}.fill{int(fill)}"
                if not ok:
                    bad += 1
                    print(
                        f"[rank {RANK}] FAIL {tag} first={got[0]} want={want}",
                        flush=True,
                    )
                else:
                    print(f"[rank {RANK}] OK   {tag}", flush=True)
                del x, got
    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {RANK}] {'PASS' if bad == 0 else 'FAILURES ' + str(bad)}", flush=True)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
