"""Tests for the mojo distributed backend (torch_mojo_backend/distributed).

Single-process pieces (registration, dtype maps, CPU/gloo delegation) run
anywhere. Real multi-rank NCCL coverage launches torchrun subprocesses and
needs multiple GPUs, so those tests skip on smaller machines; they are also
exercised on the cluster by the SLURM jobs in demo_scripts/nanogpt_ddp.
"""

import datetime
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from torch_mojo_backend import get_accelerators, register_mojo_devices

_WORKER = Path(__file__).parent / "ddp_worker.py"


def _gpu_count() -> int:
    return len(list(get_accelerators())) - 1  # get_accelerators appends the CPU


def test_backend_name_registered():
    register_mojo_devices()
    assert "mojo" in dist.Backend.backend_list
    assert dist.Backend.default_device_backend_map.get("mojo") == "mojo"


def test_nccl_dtype_and_op_maps():
    from torch_mojo_backend.distributed import nccl
    from torch_mojo_backend.distributed.process_group import _nccl_dtype, _nccl_red_op

    assert _nccl_dtype(torch.bfloat16) == nccl.NCCL_BFLOAT16 == 9
    assert _nccl_dtype(torch.float32) == nccl.NCCL_FLOAT32 == 7
    assert _nccl_dtype(torch.int64) == nccl.NCCL_INT64 == 4
    assert _nccl_dtype(torch.bool) == nccl.NCCL_UINT8 == 1
    with pytest.raises(TypeError):
        _nccl_dtype(torch.complex64)
    assert _nccl_red_op(dist.ReduceOp.SUM) == nccl.NCCL_SUM
    assert _nccl_red_op(dist.ReduceOp.AVG) == nccl.NCCL_AVG


def test_cpu_collectives_through_gloo_delegation():
    """World-size-1 init with backend="mojo": CPU tensors ride the private gloo."""
    register_mojo_devices()
    store = dist.TCPStore("127.0.0.1", 29517, 1, is_master=True)
    dist.init_process_group(
        backend="mojo",
        store=store,
        rank=0,
        world_size=1,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        t = torch.arange(4.0)
        dist.all_reduce(t)
        assert t.tolist() == [0.0, 1.0, 2.0, 3.0]
        objs = [None]
        dist.all_gather_object(objs, {"hello": "world"})
        assert objs[0] == {"hello": "world"}
        dist.barrier()
        assert dist.get_backend() == "mojo"
    finally:
        dist.destroy_process_group()


def _run_torchrun(nproc: int, mode: str, extra_env: dict[str, str] | None = None):
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)  # the worker pins per-rank visibility
    env.update(extra_env or {})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={nproc}",
            str(_WORKER),
            mode,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        cwd=Path(__file__).parent.parent,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"torchrun {mode} failed (rc={result.returncode})\n"
            f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
        )


@pytest.mark.parametrize("comm_stream", ["1", "0"], ids=["side-stream", "same-stream"])
@pytest.mark.parametrize("mode", ["collectives", "ddp_parity"])
def test_two_rank_nccl(mode: str, comm_stream: str):
    if _gpu_count() < 2:
        pytest.skip("needs at least 2 GPUs")
    _run_torchrun(2, mode, {"TORCH_MOJO_BACKEND_COMM_STREAM": comm_stream})
