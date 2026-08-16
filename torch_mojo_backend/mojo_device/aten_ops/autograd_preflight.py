"""Ops that preflight their native autograd node from the forward.

Each op here has a fast forward but records a backward node the autograd
engine may not be able to run. A raise from inside the engine aborts the
process on this backend instead of propagating (the engine restores streams
through PyTorch's noexcept Python device guard, which std::terminates when the
Python round-trip fails during unwind), so the refusal has to happen in the
forward, where it still has a traceback naming the op. Each function's
docstring spells out its own case.
"""

import torch

from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

from .support import _fast, _unsupported


def mojo_device_embedding(
    weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False
):
    """Native-autograd embedding with a forward-time autograd-mode preflight.

    An exception raised while the autograd engine runs a backward node aborts
    the process on this backend: the engine's stream guard restores streams
    through PyTorch's noexcept Python device guard, which std::terminates
    when the Python round-trip fails during unwind. Unsupported autograd
    modes must therefore be rejected before EmbeddingBackward0 is recorded,
    not when the engine reaches the sparse or dense backward.
    """
    aten_fast = _fast()
    if torch.is_grad_enabled() and weight.requires_grad:
        if sparse or scale_grad_by_freq:
            mode = "sparse=True" if sparse else "scale_grad_by_freq=True"
            raise NotImplementedError(
                f"Mojo eager embedding autograd does not yet support {mode}"
            )
        # The recorded backward's atomic accumulation is nondeterministic;
        # alerting there is too late for the same unwind-abort reason.
        aten_fast._alert_not_deterministic("embedding_dense_backward on Mojo")
    result = aten_fast.fast_aten_embedding(
        weight, indices, padding_idx, scale_grad_by_freq, sparse
    )
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::embedding", (weight, indices))
    return result


def _linear_backward_unsupported_reason(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    output_mask: tuple[bool, bool, bool],
) -> str | None:
    """Why `aten::linear_backward` will decline these operands, or None.

    The verdict has to come from the kernel layer rather than from a copy of
    its conditions here, because those conditions move: the fp32 weight-
    gradient GEMM only recently learned to materialize its transposed operand,
    and a stale copy would reject calls that now run. So we ask `aten_fast`
    for a predicate, and until it grows one (see the follow-ups) we fall back
    to reading its own dtype constant and applying only the *necessary*
    conditions from `fast_aten_linear_backward`'s entry gate.

    That fallback is deliberately incomplete: it names regimes the kernel
    provably cannot take, and stays silent about the GEMM-path questions it
    cannot answer without launching one. Under-reporting leaves the pre-
    existing abort in place for regimes it misses; over-reporting would break
    working models, so every uncertainty resolves to None.
    """
    aten_fast = _fast()
    predicate = getattr(aten_fast, "fast_aten_linear_backward_supported", None)
    if predicate is not None:
        try:
            if predicate(input, weight, output_mask):
                return None
        except Exception:
            return None
        return (
            "the eager kernel layer reports no path for this dtype and layout "
            "combination"
        )

    # `fast_aten_linear_backward` tests every operand dtype against this tuple
    # before it does anything else, so reading the tuple (rather than
    # restating its contents) keeps the check current if the kernels widen.
    supported = getattr(aten_fast, "_FLOAT_DTYPES", None)
    if supported is None:
        return None
    try:
        if input._dtype not in supported:
            covered = ", ".join(
                str(dtype).removeprefix("DType.") for dtype in supported
            )
            return (
                f"linear_backward covers {covered} only, and these operands "
                f"are {input.dtype}"
            )
        if weight._dtype != input._dtype or (
            bias is not None and bias._dtype != input._dtype
        ):
            return (
                "linear_backward requires input, weight and bias to share one "
                f"dtype, and these are {input.dtype}/{weight.dtype}"
                + ("" if bias is None else f"/{bias.dtype}")
            )
    except AttributeError:
        # A non-mojo operand: the forward below reports that far better.
        return None
    return None


def mojo_device_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> TorchMojoTensor:
    """Linear with a forward-time preflight of its native autograd node.

    Registering `aten::linear` keeps `nn.Linear` from decomposing to addmm, so
    its backward is the fused `aten::linear_backward` node. That node runs
    inside the autograd engine, where an exception aborts the process on this
    backend: the engine's stream guard restores streams through PyTorch's
    noexcept Python device guard, which std::terminates when the Python
    round-trip fails during unwind — "terminate called without an active
    exception", no traceback, no frame naming the op. A backward that cannot
    run has to be reported before LinearBackward is recorded, exactly as
    `mojo_device_embedding` above rejects its unsupported autograd modes.

    `requires_grad` is the engine's `task_should_compute_output` mask for an
    ordinary `loss.backward()`; a partial `torch.autograd.grad(inputs=...)`
    can narrow it further, so the preflight is asked about the widest mask the
    engine could request.
    """
    if torch.is_grad_enabled() and (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        reason = _linear_backward_unsupported_reason(
            input,
            weight,
            bias,
            (
                bool(input.requires_grad),
                bool(weight.requires_grad),
                bias is not None and bool(bias.requires_grad),
            ),
        )
        if reason is not None:
            raise NotImplementedError(
                "aten::linear would record an autograd node "
                "(aten::linear_backward) that mojo eager mode cannot run: "
                f"{reason} (input {tuple(input.shape)} {input.dtype}, weight "
                f"{tuple(weight.shape)} {weight.dtype}, device "
                f"{input.device}). linear_backward forms the weight gradient "
                "as mm(grad_output.transpose(0, 1), input) — a GEMM whose "
                "left operand is a non-contiguous transposed view — so not "
                "every dtype and layout has a path. Workarounds: run the "
                "layer in bfloat16, float16 or float32; for float32, "
                'torch.set_float32_matmul_precision("high") additionally '
                "opens the TF32 GEMM, which takes arbitrary 2-D layouts. "
                "Raised from the forward on purpose: raised from the backward "
                "node instead, this aborts the process without a traceback."
            )
    aten_fast = _fast()
    result = aten_fast.fast_aten_linear(input, weight, bias)
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::linear", (input, weight, bias))
    return result


def mojo_device_native_batch_norm(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[TorchMojoTensor, TorchMojoTensor, TorchMojoTensor]:
    """Batch norm with a forward-time preflight of its native autograd node.

    The training forward exists here; `aten::native_batch_norm_backward` does
    not (docs/optimization_backlog.md N2). Letting the forward record
    NativeBatchNormBackward anyway would move the failure into the autograd
    engine, where an exception aborts the process on this backend rather than
    raising — the same unwind hazard `mojo_device_linear` above documents — so
    a training call that would need a gradient is refused here, from the
    forward, with a traceback that names the op.

    Inference needs no preflight: `training=False` records a backward this
    backend never has to run.
    """
    if (
        training
        and torch.is_grad_enabled()
        and (
            input.requires_grad
            or (weight is not None and weight.requires_grad)
            or (bias is not None and bias.requires_grad)
        )
    ):
        raise NotImplementedError(
            "aten::native_batch_norm (training=True) would record an autograd "
            "node (aten::native_batch_norm_backward) that mojo eager mode "
            f"does not implement (input {tuple(input.shape)} {input.dtype}, "
            f"device {input.device}). The forward itself is supported: run it "
            "under torch.no_grad(), or put the module in eval() mode. Raised "
            "from the forward on purpose: raised from the backward node "
            "instead, this aborts the process without a traceback."
        )
    aten_fast = _fast()
    result = aten_fast.fast_aten_native_batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported(
            "aten::native_batch_norm", (input, weight, bias, running_mean, running_var)
        )
    return result
