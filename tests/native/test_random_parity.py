"""Bit-exactness of the device RNG against stock PyTorch CUDA.

Every case in rng_parity_cases runs from the same seed on the mojo GPU and is
compared, digest and generator state, against stock CUDA: the checked-in
`rng_golden.json` for the device-independent cases (recorded on an H100 with
torch 2.11.0+cu128), and a live CUDA device in this process when one is
available, or a record made with `rng_parity_dump.py dump cuda` on the same
GPU model named by `TMB_RNG_REFERENCE=<path>`.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from tests.native import rng_parity_cases as cases
from torch_mojo_backend.native import device_module
from tests.native.rng_parity_dump import Record, digest_of, run_case
from torch_mojo_backend import get_accelerators

_GOLDEN = json.loads((Path(__file__).with_name("rng_golden.json")).read_text())


@pytest.fixture
def cuda_like_gpu(mojo_gpu):
    """The parity target is CUDA: off NVIDIA the libdevice fast paths fall
    back to std.math and bit-exactness is not claimed."""
    if get_accelerators()[0].api != "cuda":
        pytest.skip("CUDA bit-parity is only claimed on NVIDIA GPUs")
    return mojo_gpu


def _assert_same(name: str, got: Record, want: Record, who: str = "ours"):
    assert got["dtype"] == want["dtype"], name
    assert list(got["shape"]) == list(want["shape"]), name
    if got["sha256"] != want["sha256"]:
        pytest.fail(
            f"{name}: {who} sha256 {got['sha256']} != golden {want['sha256']}; "
            f"head {who}={got['head'][:6]} golden={want['head'][:6]} "
            "(refresh with rng_parity_dump.py golden <cuda record>)"
        )
    assert got["state"] == want["state"], (
        f"{name}: {who} generator state {got['state']} != golden {want['state']}"
    )


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device: the golden file stands in for stock CUDA")
    return "cuda:0"


@pytest.mark.parametrize("name", cases.GOLDEN)
def test_mojo_output_matches_golden(cuda_like_gpu, name):
    """Same seed, same bits, same counter as stock CUDA (recorded)."""
    got = run_case(cases.CASES[name], cuda_like_gpu)
    _assert_same(name, got, _GOLDEN["cases"][name], "mojo")


@pytest.mark.parametrize("name", cases.GOLDEN)
def test_cuda_output_matches_golden(cuda_device, name):
    """The recorded digests are still what stock CUDA produces on this box."""
    got = run_case(cases.CASES[name], cuda_device)
    _assert_same(name, got, _GOLDEN["cases"][name], "cuda")


def _live_reference():
    """A reference for the device-dependent cases: live CUDA, or a record."""
    if torch.cuda.is_available():
        return "cuda"
    path = os.environ.get("TMB_RNG_REFERENCE")
    if path:
        return torch.load(path, weights_only=False)
    return None


def _device_dependent_ids():
    ids = sorted(cases.DEVICE_DEPENDENT)
    if os.environ.get("TMB_RNG_BIG"):
        ids += sorted(cases.BIG_CASES)
    return ids


@pytest.mark.parametrize("name", _device_dependent_ids())
def test_matches_cuda_above_grid_cap(cuda_like_gpu, name):
    """Above the grid cap the draw depends on the GPU model, as on CUDA: the
    same model (the record's `props` name it) must still agree bit for bit."""
    ref = _live_reference()
    if ref is None:
        pytest.skip("needs a CUDA device or TMB_RNG_REFERENCE=<record of the same GPU>")
    fn = cases.CASES.get(name) or cases.BIG_CASES[name]
    got = run_case(fn, cuda_like_gpu)
    if ref == "cuda":
        want = run_case(fn, "cuda:0")
    else:
        want = ref["results"].get(name)
        if want is None:
            pytest.skip(f"{name} is not in the reference record")
        assert want["ok"], want
    _assert_same(name, got, want, "mojo")


def test_state_layout_matches_cuda(mojo_gpu):
    """get_rng_state is CUDA's 16 bytes: seed then offset, little-endian."""
    torch.manual_seed(cases.SEED)
    torch.rand(1000, device=mojo_gpu)
    state = device_module.get_rng_state().tolist()
    assert int.from_bytes(bytes(state[:8]), "little") == cases.SEED
    assert int.from_bytes(bytes(state[8:]), "little") == 4


def test_digest_is_stable():
    """The digest the golden file keys on is a plain SHA-256 of the bytes."""
    t = torch.arange(4, dtype=torch.float32)
    assert digest_of(t) == digest_of(t.clone())
    assert digest_of(t) != digest_of(t + 1)
