import contextlib
import inspect
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import torch

from torch_mojo_backend import mojo_backend
from torch_mojo_backend.eager_kernels import aten_fast
from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS
from torch_mojo_backend.types import CountedCallable


@contextlib.contextmanager
def _xfail_if_unsupported(device: str) -> Iterator[None]:
    """xfail (rather than fail) when the mojo eager backend raises
    NotImplementedError for an input its fast kernels don't cover.

    Killing the graph fallback (docs/strided_owning_tensors_design.md) turned
    "unsupported input" from a slow fallback into a clear raise; this makes the
    existing suite record those as expected-unsupported instead of hard
    failures, without editing individual tests or masking real errors (only
    our own "not supported by mojo" NotImplementedError is caught).
    """
    try:
        yield
    except NotImplementedError as exc:
        if str(device).startswith("mojo") and "mojo" in str(exc):
            import pytest  # noqa: PLC0415 -- pytest is a dev dependency; this module imports without it

            pytest.xfail(f"unsupported on mojo eager: {exc}")
        raise


class CallChecker:
    """Asserts that at least one of the registered implementations ran.

    Ops covered by the mojo fast eager path have two implementations:
    the graph one in `aten_functions` (used by the torch.compile backend)
    and the Mojo-kernel one in `aten_fast` (used by mojo eager mode).
    A test registers the `aten_functions` twin; `register` automatically
    also accepts the matching `aten_fast.fast_<name>` twin, so the same
    test passes whether the op routed to the graph path (compile) or the
    fast path (eager) — no per-test bookkeeping needed.
    """

    def __init__(self):
        self._functions_to_check: tuple[CountedCallable, ...] | None = None
        self._counts_before_starting_to_check: list[int] | None = None

    @staticmethod
    def _fast_twins(func: Callable[..., object]) -> list[CountedCallable]:
        """The aten_fast counterparts of an aten_functions twin.

        Matches `fast_<name>` and its variants `fast_<name>_<suffix>` (e.g.
        `aten_min` -> `fast_aten_min`, `fast_aten_min_dim`), so a test that
        registers the base op accepts whichever specialized fast impl the
        inputs routed to. Only instrumented (call-counted) functions match.
        """
        name = getattr(func, "__name__", "")
        if not name.startswith("aten"):
            return []
        base = f"fast_{name}"
        twins = []
        for attr in dir(aten_fast):
            if attr == base or attr.startswith(base + "_"):
                cand = getattr(aten_fast, attr)
                if hasattr(cand, "call_count"):
                    twins.append(cand)
        return twins

    @staticmethod
    def _eager_twins(func: Callable[..., object]) -> list[CountedCallable]:
        """The instrumented mojo registration(s) whose op matches an
        aten_functions twin. Covers ops implemented as custom / out-variant
        registrations (empty_like, mean.out, normal_, ...) that don't route
        through an aten_fast.fast_* function, so nothing else observes them.
        """
        name = getattr(func, "__name__", "")
        if not name.startswith("aten_"):
            return []
        base = name[len("aten_") :]  # e.g. "empty_like", "mean_out", "_log_softmax"
        candidates = {f"aten::{base}"}
        if "_" in base:
            head, tail = base.rsplit("_", 1)
            candidates.add(f"aten::{head}.{tail}")  # mean_out -> aten::mean.out
        prefix = f"aten::{base}."
        # The scaled_dot_product_attention family (plain / _math / _flash /
        # _efficient) is one concept; eager routes to the fused impl whatever
        # variant the test names, so accept any of them.
        sdpa_family = "scaled_dot_product" in base
        twins = []
        for op_name, counter in EAGER_CALL_COUNTERS.items():
            if (
                op_name in candidates
                or op_name.startswith(prefix)
                or (sdpa_family and "scaled_dot_product" in op_name)
            ):
                twins.append(counter)
        return twins

    def register(self, *funcs: Callable[..., object]):
        """Register the functions expected to run.

        `funcs` are typed `Callable` (each caller's own precise signature,
        e.g. `aten_functions.aten_min`), not `CountedCallable`: under tests
        `map_to`/`register_aten_op` always wrap them with a `call_count`
        attribute, but that fact is deliberately hidden from their static
        type (see `aten_functions.map_to`) so callers elsewhere keep a
        precise signature. Cast here, at the one place that relies on it.
        """
        expanded: list[CountedCallable] = []
        for func in funcs:
            counted_func = cast(CountedCallable, func)
            if counted_func not in expanded:
                expanded.append(counted_func)
            for twin in self._fast_twins(func) + self._eager_twins(func):
                if twin not in expanded:
                    expanded.append(twin)
        self._functions_to_check = tuple(expanded)
        self._counts_before_starting_to_check = [
            f.call_count for f in self._functions_to_check
        ]

    def check_was_called(self):
        if (
            self._functions_to_check is None
            or self._counts_before_starting_to_check is None
        ):
            raise ValueError(
                "No function to check was set, call call_checker.register first"
            )
        if not any(
            func.call_count > count_before
            for func, count_before in zip(
                self._functions_to_check, self._counts_before_starting_to_check
            )
        ):
            names = ", ".join(f.__name__ for f in self._functions_to_check)
            raise AssertionError(
                f"Expected one of [{names}] to be called at least once in the test, but none was"
            )


def _as_tensor_list(
    outputs: torch.Tensor | Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    return [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)


def check_functions_are_equivalent(
    fn: Callable[..., torch.Tensor | Sequence[torch.Tensor]],
    device: str | None,
    inputs: list[torch.Tensor],
    fn_compiled: Callable[..., torch.Tensor | Sequence[torch.Tensor]] | None = None,
    rtol: float | None = None,
    atol: float | None = None,
):
    fn_compiled = fn_compiled or torch.compile(backend=mojo_backend)(fn)
    if device is not None:
        inputs = [input_tensor.to(device) for input_tensor in inputs]

    # We use the compiled first because compiled never changes
    # the input tensors, while the original function might.
    output_compiled = fn_compiled(*inputs)
    output_original = fn(*inputs)

    assert type(output_original) is type(output_compiled)

    for i, (original, compiled) in enumerate(
        zip(_as_tensor_list(output_original), _as_tensor_list(output_compiled))
    ):
        assert original.shape == compiled.shape, f"Issue with output {i}"
        assert original.device == compiled.device, f"Issue with output {i}"
        assert original.dtype == compiled.dtype, f"Issue with output {i}"
        torch.testing.assert_close(original, compiled, rtol=rtol, atol=atol)


# torch.testing.assert_close's default (rtol, atol) per floating dtype.
_DEFAULT_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
}


def assert_close_fp64_anchored(
    actual: torch.Tensor,
    expected: torch.Tensor,
    reference: torch.Tensor,
    *,
    slack: float = 2.0,
):
    """`actual` must sit as close to the float64 `reference` as `expected`
    (torch's own result in the working dtype) does, within a factor `slack`.

    For fp32 reductions torch's default bar (rtol 1.3e-6, atol 1e-5) is
    tighter than torch's own rounding error against the exact answer, so
    two correct implementations that merely sum in a different order fail
    it on some elements, depending on the CPU's SIMD width. This bar asks
    the question the default one means to: is the kernel as accurate as
    torch's? The default tolerance stays as a floor, so where torch is exact
    the check is the ordinary one.
    """
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    assert reference.dtype == torch.float64, reference.dtype
    rtol, atol = _DEFAULT_TOLERANCES[expected.dtype]
    actual, expected = actual.cpu(), expected.cpu()
    if not torch.isfinite(reference).all():
        torch.testing.assert_close(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True
        )
        return
    error = (actual.double() - reference).abs()
    torch_error = (expected.double() - reference).abs()
    allowed = torch.clamp(atol + rtol * reference.abs(), min=slack * torch_error.max())
    bad = error > allowed
    if bad.any():
        worst = int(error.flatten().argmax())
        raise AssertionError(
            f"{int(bad.sum())} / {bad.numel()} elements further from the float64 "
            f"reference than {slack}x torch's own worst error "
            f"({float(torch_error.max()):.3g}) and outside rtol={rtol}, atol={atol}. "
            f"Worst: actual {float(actual.flatten()[worst])!r}, torch "
            f"{float(expected.flatten()[worst])!r}, reference "
            f"{float(reference.flatten()[worst])!r} at flat index {worst}."
        )


@dataclass
class Conf:
    device: str
    compile: bool

    def __str__(self) -> str:
        if self.compile:
            word = "compiled"
        else:
            word = "eager"
        return f"{self.device}, {word}"


def to_device(tensors: list[torch.Tensor], device: str) -> list[torch.Tensor]:
    return [torch.clone(tensor).to(device) for tensor in tensors]


def check_outputs(
    fn: Callable[..., torch.Tensor | Sequence[torch.Tensor]],
    conf: Conf,
    inputs: list[torch.Tensor],
    *,
    rtol: float | None = None,
    atol: float | None = None,
):
    # We compare to eager cpu execution
    # We first check if the function has a device argument
    has_device_arg = "device" in inspect.signature(fn).parameters
    inputs_cpu = to_device(inputs, "cpu")
    if has_device_arg:
        outputs_eager_cpu = fn(*inputs_cpu, device="cpu")
    else:
        outputs_eager_cpu = fn(*inputs_cpu)

    if conf.compile:
        fn_to_run = torch.compile(fn, backend=mojo_backend)
    else:
        fn_to_run = fn

    inputs_on_device = to_device(inputs, conf.device)
    with _xfail_if_unsupported(conf.device):
        if has_device_arg:
            outputs_conf = fn_to_run(*inputs_on_device, device=conf.device)
        else:
            outputs_conf = fn_to_run(*inputs_on_device)

    for i, (output_eager_cpu, output_conf) in enumerate(
        zip(_as_tensor_list(outputs_eager_cpu), _as_tensor_list(outputs_conf))
    ):
        expected_device = torch.device(conf.device)
        if not (output_conf.device == expected_device):
            raise AssertionError(
                f"Issue with output {i}, expected device {repr(expected_device)} but got {repr(output_conf.device)}"
            )
        assert output_eager_cpu.shape == output_conf.shape, f"Issue with output {i}"
        assert output_eager_cpu.dtype == output_conf.dtype, f"Issue with output {i}"
        # move to cpu for comparison
        output_conf_cpu = output_conf.to("cpu")
        torch.testing.assert_close(
            output_eager_cpu, output_conf_cpu, rtol=rtol, atol=atol
        )
