"""Every OpInfo in PyTorch's `op_db`, executed on the mojo device and compared
against the same call on the CPU.

What comes from OpInfo rather than from us:

* the sample inputs (`sample_inputs_func`) and, with `--reference-inputs`, the
  larger `reference_inputs_func` corpus of awkward shapes, non-contiguous
  layouts, broadcasts and scalar overloads;
* the dtypes each operator is expected to support, per `op.dtypesIfCUDA` —
  the accelerator set, not the CPU one;
* the accuracy bar: `op.precisionOverride` when the operator declares one,
  otherwise `torch.testing.assert_close`'s dtype defaults, applied through
  `TestCase.assertEqual`.  We never invent a tolerance;
* the corner cases: `error_inputs_func`, i.e. the inputs PyTorch asserts must
  RAISE, and the exception type and message it requires.

An operator we have not implemented raises NotImplementedError from the
dispatch layer.  That is reported as a distinct outcome (skip with a
"not implemented" reason), never as a pass and never as a wrong answer,
because the point of this suite is to tell those three apart.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase, run_tests

# The dtypes worth exercising on an accelerator backend.  Deliberately not the
# full OpInfo set: float64 is absent on some GPUs we target and complex is not
# implemented at all, so including them would report a backend-wide gap once
# per operator instead of once.
_DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.bool)


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list | tuple):
        return type(value)(_to_cpu(v) for v in value)
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    return value


def _not_implemented(exc: BaseException) -> bool:
    """True when the backend declined the op rather than got it wrong."""
    if isinstance(exc, NotImplementedError):
        return True
    text = str(exc).lower()
    return "not implemented" in text or "no fast implementation" in text


def _harness_cannot_construct(exc: BaseException) -> bool:
    """True when the harness's build-on-CPU-then-move strategy created a
    call the operator's own contract forbids on every backend, not just
    ours -- e.g. tensor_split's indices/sections argument must stay on
    CPU; moving it because it happens to be a Tensor is a property of
    "move every tensor found in the sample" being simpler than knowing
    each op's per-argument device rules, not a mojo bug.
    """
    return "but it's on" in str(exc) and "to be on cpu" in str(exc)


def _cross_device_comparison_skip_reason(op: Any, dtype: torch.dtype) -> str | None:
    """OpInfo already flags operators whose output is legitimately allowed
    to differ across devices -- uninitialized memory for the `empty`
    family, a per-device RNG stream for `dropout` -- by skipping
    `TestCommon.test_compare_cpu`, upstream's own CPU-vs-other-device
    consistency test. `test_matches_cpu` is the same kind of comparison,
    so it honors the same skip instead of hand-maintaining a duplicate
    list that would drift from upstream's.
    """
    for skip in op.skips:
        if skip.test_name != "test_compare_cpu":
            continue
        if skip.cls_name not in (None, "TestCommon"):
            continue
        if skip.device_type not in (None, "privateuse1"):
            continue
        if skip.dtypes is not None and dtype not in skip.dtypes:
            continue
        if not skip.active_if:
            continue
        return (
            "OpInfo skips TestCommon.test_compare_cpu for this op "
            "(output not expected to match across devices)"
        )
    return None


class TestOpInfoConformance(TestCase):
    """One test per (operator, dtype), driven entirely by OpInfo metadata."""

    @ops(op_db, allowed_dtypes=_DTYPES)
    def test_matches_cpu(self, device: str, dtype: torch.dtype, op: Any) -> None:
        """Same operator, same inputs, mojo vs CPU, at OpInfo's own tolerance.

        Samples are built on the CPU and moved, never built on the device.
        `sample_inputs(device=...)` constructs its tensors THERE, so a backend
        missing any op that `make_tensor` reaches fails during input
        construction and reports as if the operator under test were broken --
        which is a property of the harness, not of the operator.  Moving also
        makes both legs read bit-identical inputs.
        """
        skip_reason = _cross_device_comparison_skip_reason(op, dtype)
        if skip_reason is not None:
            self.skipTest(skip_reason)
        checked = 0
        for sample in op.sample_inputs("cpu", dtype, requires_grad=False):
            try:
                moved = sample.transform(
                    lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
                )
                actual = op(moved.input, *moved.args, **moved.kwargs)
            except Exception as exc:  # noqa: BLE001 - triaging is the point
                if _not_implemented(exc) or _harness_cannot_construct(exc):
                    self.skipTest(f"not implemented on mojo: {exc}")
                raise
            expected = op(sample.input, *sample.args, **sample.kwargs)
            # assertEqual carries the OpInfo precisionOverride for this dtype
            # when the operator declares one; otherwise assert_close defaults.
            self.assertEqual(_to_cpu(actual), expected, exact_dtype=True)
            checked += 1
        if checked == 0:
            self.skipTest("OpInfo produced no sample inputs for this dtype")

    @ops(
        [op for op in op_db if op.error_inputs_func is not None],
        allowed_dtypes=(torch.float32,),
    )
    def test_errors_match(self, device: str, dtype: torch.dtype, op: Any) -> None:
        """The inputs PyTorch says must raise, must raise here too.

        A backend that silently accepts a malformed call is a worse failure
        than one that cannot run it at all, and only OpInfo knows which calls
        those are per operator.
        """
        checked = 0
        for error_input in op.error_inputs(device):
            sample = error_input.sample_input
            with self.assertRaises(error_input.error_type):
                try:
                    op(sample.input, *sample.args, **sample.kwargs)
                except NotImplementedError as exc:
                    self.skipTest(f"not implemented on mojo: {exc}")
            checked += 1
        if checked == 0:
            self.skipTest("OpInfo declared no error inputs for this operator")


instantiate_device_type_tests(
    TestOpInfoConformance, globals(), only_for=("privateuse1",)
)


if __name__ == "__main__":
    run_tests()
