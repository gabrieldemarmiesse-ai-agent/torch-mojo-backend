"""torchrun-free worker for test_distributed.py's per-op overhead check.

Prints two numbers:

* ``us_per_op`` — one tiny eager op end to end, with or (``--no-hook``)
  without the comm-fence wrapper `register_mojo_devices()` installs in front
  of every op. A whole process per leg, because op registration happens once
  per process. At ~20 us of eager dispatch per op the wrapper is under the
  run-to-run noise here, hence the second number.
* ``wrapper_us`` — the wrapper frame in isolation, against a callable that
  does nothing. This is the cost the hook actually adds.
"""

import sys
import time

import torch


def _time_add(x: torch.Tensor, y: torch.Tensor, iterations: int) -> float:
    from torch_mojo_backend.mojo_device import torch_mojo_device_module

    start = time.perf_counter()
    for _ in range(iterations):
        torch.add(x, y)
    torch_mojo_device_module.synchronize()
    return (time.perf_counter() - start) / iterations * 1e6


def _time_wrapper_frame(x: torch.Tensor) -> float:
    from torch_mojo_backend.mojo_device.register import _fence_pending_collectives

    def target(a: object, b: object) -> object:
        return a

    wrapped = _fence_pending_collectives(target)
    iterations = 200000
    start = time.perf_counter()
    for _ in range(iterations):
        target(x, x)
    raw = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(iterations):
        wrapped(x, x)
    hooked = time.perf_counter() - start
    return (hooked - raw) / iterations * 1e6


def main():
    from torch_mojo_backend.mojo_device import register

    if "--no-hook" in sys.argv:
        # The wrapper is applied by register_mojo_devices()'s install loop
        # through this global, so the identity gives the same process minus
        # the hook and nothing else.
        register._fence_pending_collectives = lambda func: func  # ty: ignore[invalid-assignment]
    register.register_mojo_devices()

    x = torch.ones(16, device="mojo")
    y = torch.ones(16, device="mojo")
    _time_add(x, y, 2000)  # warm the kernel cache and the dispatch caches
    # Min of three: this is a host-cost measurement, so the fastest run is
    # the one least polluted by whatever else the machine was doing.
    print(f"us_per_op={min(_time_add(x, y, 10000) for _ in range(3)):.4f}")
    print(f"wrapper_us={min(_time_wrapper_frame(x) for _ in range(3)):.4f}")


if __name__ == "__main__":
    main()
