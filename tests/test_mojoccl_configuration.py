"""Keep the collective configuration surface limited to vendor analogs."""

import re
from pathlib import Path

from scripts.compare_kernel_asm import find_entry_modules
from torch_mojo_backend import env_vars

PACKAGE = Path(__file__).resolve().parents[1] / "torch_mojo_backend"
# Purpose analogs, not promises of identical value syntax/units. Reviewed
# against docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html.
SUPPORTED = {
    "MOJOCCL_SOCKET_IFNAME": "NCCL_SOCKET_IFNAME",
    "MOJOCCL_BOOTSTRAP_TIMEOUT_S": "NCCL_SOCKET_RETRY_CNT",
    "MOJOCCL_IB_HCA": "NCCL_IB_HCA",
    "MOJOCCL_IB_TIMEOUT_S": "NCCL_IB_TIMEOUT",
    "MOJOCCL_IB_RELAXED_ORDERING": "NCCL_IB_PCI_RELAXED_ORDERING",
    "MOJOCCL_IB_TRACE": "NCCL_DEBUG",
    "MOJOCCL_NET": "NCCL_NET",
    "MOJOCCL_LIBFABRIC": "NCCL_NET_PLUGIN",
    "MOJOCCL_FABRIC_PROVIDER": "NCCL_NET",
    "MOJOCCL_FABRIC_DOMAIN": "NCCL_IB_HCA",
    "MOJOCCL_REGION_MB": "NCCL_BUFFSIZE",
    "MOJOCCL_NVLS": "NCCL_NVLS_ENABLE",
}


def test_mojoccl_has_only_reviewed_environment_controls():
    found = set()
    for path in PACKAGE.rglob("*"):
        if path.suffix not in {".mojo", ".py"}:
            continue
        # The project-wide table also indexes standalone probe artifact paths.
        # The control surface here is what the shipped library itself uses.
        if path == PACKAGE / "env_vars.py":
            continue
        source = path.read_text()
        found.update(re.findall(r"\bMOJOCCL_[A-Z0-9_]+", source))
    # This is a compiler-runtime global key, never read with getenv.
    found.discard("MOJOCCL_ROOT_")
    assert found == set(SUPPORTED), (
        "New collective controls need a reviewed NCCL/RCCL purpose analog; "
        f"unexpected={found - SUPPORTED.keys()}, missing={SUPPORTED.keys() - found}"
    )
    registry = PACKAGE / "distributed/mojoccl/env_vars.mojo"
    registered = set(
        re.findall(r"^comptime (MOJOCCL_\w+) =", registry.read_text(), re.M)
    )
    assert registered == set(SUPPORTED)
    assert registered <= env_vars.OWN_ENV_VARS.keys()


def test_asm_discovery_covers_collective_entry_and_skips_helpers(tmp_path: Path):
    kernel_dir = Path("collectives")
    directory = tmp_path / kernel_dir
    directory.mkdir()
    (directory / "mojoccl.mojo").write_text("def ncclAllReduce(\n")
    (directory / "family.mojo").write_text("def tmb_call(\n")
    (directory / "helper.mojo").write_text("def helper(\n")
    assert find_entry_modules(tmp_path, kernel_dir) == {
        "mojoccl": kernel_dir / "mojoccl.mojo",
        "family": kernel_dir / "family.mojo",
    }
