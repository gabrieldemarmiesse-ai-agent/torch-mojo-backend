"""The torch.Generator Python API on the mojo device.

`torch.Generator(device="mojo")` goes through the shim's `getNewGenerator`
(`MojoGeneratorImpl` in shim_runtime.cpp), whose state is CUDA's: a Philox
seed plus an offset counted in curand units, serialized as 16 little-endian
bytes. These tests pin the Python-facing Generator methods to that contract;
whether the drawn bits themselves match stock CUDA is test_random_parity.py's
job. `philox_state()` from the upstream docs does not exist in this torch
(2.11 exposes the Philox counter as `get_offset`/`set_offset`), and the
`graphsafe_*` pair is intentionally unsupported -- see the last test.
"""

import struct

import pytest
import torch

from torch_mojo_backend.native import device_module

SEED = 20260914


def _draw(device: str, generator: torch.Generator, n: int = 64) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(generator=generator).cpu()


def test_generator_device(mojo_device):
    g = torch.Generator(device=mojo_device)
    assert g.device == torch.device(mojo_device)


def test_generator_bare_device_string_defaults_to_index_0(mojo_gpu):
    assert torch.Generator(device="mojo").device == torch.device("mojo", 0)


def test_manual_seed_and_initial_seed(mojo_device):
    g = torch.Generator(device=mojo_device)
    assert g.manual_seed(SEED) is g
    assert g.initial_seed() == SEED
    _draw(mojo_device, g)
    assert g.initial_seed() == SEED, "drawing must not change the seed"


def test_seed_draws_a_fresh_nondeterministic_seed(mojo_device):
    g = torch.Generator(device=mojo_device)
    _draw(mojo_device, g)
    s = g.seed()
    assert g.initial_seed() == s
    assert g.get_offset() == 0, "seeding resets the Philox counter"
    assert g.seed() != s


def test_get_state_is_cudas_16_bytes(mojo_device):
    """seed then offset, little-endian, on a CPU uint8 tensor -- the same
    wire format as CUDA generators and torch.mojo.get_rng_state()."""
    g = torch.Generator(device=mojo_device)
    g.manual_seed(SEED)
    _draw(mojo_device, g)
    state = g.get_state()
    assert state.device.type == "cpu"
    assert state.dtype == torch.uint8
    assert state.shape == (16,)
    seed, offset = struct.unpack("<QQ", bytes(state.tolist()))
    assert seed == SEED
    assert offset == g.get_offset()


def test_set_state_round_trips_mid_stream(mojo_device):
    g = torch.Generator(device=mojo_device)
    g.manual_seed(SEED)
    _draw(mojo_device, g)
    state = g.get_state()
    second = _draw(mojo_device, g)
    g.set_state(state)
    torch.testing.assert_close(_draw(mojo_device, g), second)


def test_set_state_transfers_to_a_fresh_generator(mojo_device):
    g = torch.Generator(device=mojo_device)
    g.manual_seed(SEED)
    _draw(mojo_device, g)
    other = torch.Generator(device=mojo_device)
    other.set_state(g.get_state())
    assert other.initial_seed() == SEED
    assert other.get_offset() == g.get_offset()
    torch.testing.assert_close(_draw(mojo_device, other), _draw(mojo_device, g))


def test_set_state_rejects_malformed_states(mojo_device):
    g = torch.Generator(device=mojo_device)
    for bad in (
        torch.zeros(8, dtype=torch.uint8),  # too short
        torch.zeros(16, dtype=torch.int32),  # wrong dtype
    ):
        with pytest.raises(RuntimeError, match="16-byte uint8"):
            g.set_state(bad)


def test_offset_is_the_philox_counter(mojo_device):
    """`get_offset`/`set_offset` expose the Philox counter (torch 2.11 has no
    `philox_state()`): a (seed, offset) pair fully names a stream position, so
    a fresh generator fast-forwarded to the offset resumes it exactly."""
    g = torch.Generator(device=mojo_device)
    g.manual_seed(SEED)
    assert g.get_offset() == 0
    first = _draw(mojo_device, g)
    offset = g.get_offset()
    assert offset > 0 and offset % 4 == 0, "counted in curand units of 4"
    second = _draw(mojo_device, g)

    resumed = torch.Generator(device=mojo_device)
    resumed.manual_seed(SEED)
    torch.testing.assert_close(_draw(mojo_device, resumed), first)
    resumed.manual_seed(SEED)
    resumed.set_offset(offset)
    torch.testing.assert_close(_draw(mojo_device, resumed), second)

    with pytest.raises(RuntimeError, match="multiple of 4"):
        g.set_offset(3)


def test_clone_state(mojo_device):
    g = torch.Generator(device=mojo_device)
    g.manual_seed(SEED)
    _draw(mojo_device, g)
    clone = g.clone_state()
    assert clone is not g
    assert clone.device == g.device
    assert clone.initial_seed() == SEED
    assert clone.get_offset() == g.get_offset()

    # The clone is an independent copy: drawing from it advances only it,
    # and both produce the original's next values.
    offset = g.get_offset()
    from_clone = _draw(mojo_device, clone)
    assert g.get_offset() == offset
    torch.testing.assert_close(_draw(mojo_device, g), from_clone)


def test_explicit_generator_leaves_the_default_untouched(mojo_device):
    device_module.manual_seed_all(SEED)
    before = device_module.get_rng_state(mojo_device)
    g = torch.Generator(device=mojo_device)
    g.manual_seed(SEED + 1)
    _draw(mojo_device, g)
    torch.testing.assert_close(device_module.get_rng_state(mojo_device), before)


def test_explicit_generator_matches_the_default_stream(mojo_device):
    """Same seed, same stream: an explicit generator is the same Philox
    scheme as the device's default one, not a different RNG."""
    torch.manual_seed(SEED)
    from_default = torch.rand(64, device=mojo_device).cpu()
    g = torch.Generator(device=mojo_device)
    g.manual_seed(SEED)
    assert torch.equal(
        torch.rand(64, device=mojo_device, generator=g).cpu(), from_default
    )


def test_graphsafe_state_is_unsupported(mojo_device):
    """`graphsafe_get_state`/`graphsafe_set_state` exist for RNG use inside
    CUDA-graph capture, which the mojo device does not have; they are left
    on the base class's NotImplementedError on purpose. Implementing them
    faithfully means moving (seed, offset) into a state object shared
    between generators -- do that when graph capture arrives, then replace
    this test with real ones."""
    g = torch.Generator(device=mojo_device)
    with pytest.raises(NotImplementedError):
        g.graphsafe_get_state()
    with pytest.raises(NotImplementedError):
        g.graphsafe_set_state(torch.Generator(device=mojo_device))
