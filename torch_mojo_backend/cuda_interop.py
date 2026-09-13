"""Run CUDA-only code -- a package with its own compiled kernels -- on mojo
tensors.

The mojo device owns its memory and its streams, and a package written
against `torch.cuda` (causal-conv1d, mamba-ssm, apex, ...) will not look at
either: its C++ asks `x.is_cuda()`, takes `at::cuda::getCurrentCUDAStream()`
and launches there. Triton needs none of this because it launches through its
own driver and only *asks* torch which stream is current
(`triton_driver.py`); a compiled extension instead calls into libtorch_cuda,
so it needs a CUDA build of torch, and it needs to be handed CUDA tensors and
a CUDA stream.

Both are aliases rather than copies:

* **Memory.** MAX allocates through the CUDA driver on the device's *primary*
  context -- the same context `torch.cuda` uses -- so one device pointer is
  valid in both worlds and a tensor can be retagged rather than copied.
  `as_cuda` / `as_mojo` do that by taking torch's own DLPack export and
  rewriting the device code in the capsule (`dlpack.retag_capsule`), which
  keeps shape, strides, offset and dtype exactly and leaves the source tensor
  pinned by the capsule's deleter. Nothing is allocated and nothing is
  copied; `t.data_ptr()` is equal on both sides.
* **Ordering.** `on_mojo_stream()` makes torch.cuda's current stream an
  `ExternalStream` over the mojo current stream's vendor handle, so the
  package's launches queue behind our kernels and ours behind its, with no
  synchronization.

`call_cuda(fn, *args)` is the two together: convert, call, convert back.
`enable_cuda_fallback()` is the same conversion installed as a dispatcher
fallback, so an op with a CUDA kernel and no Mojo op runs this way -- with
the two exceptions `_convolution_backward_overrideable` describes.

Gradients do not cross an alias (DLPack carries no autograd history), so
`call_cuda` is a leaf: wrap a package's forward and backward entry points in
`CudaAutogradFunction` to get a differentiable mojo-level op.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import torch
import torch.utils.dlpack

from torch_mojo_backend.mojo_device.dlpack import retag_capsule
from torch_mojo_backend.native import device_module

if TYPE_CHECKING:
    from torch.autograd.function import FunctionCtx

# DLPack device codes (ATen/dlpack.h): kDLCUDA=2, kDLROCM=10, kDLExtDev=12.
_KDL_CUDA = 2
_KDL_ROCM = 10
_KDL_EXT_DEV = 12

_MOJO = "mojo"


def _vendor_dlpack_code() -> int:
    """ROCm builds of torch spell their device "cuda" but export kDLROCM."""
    return _KDL_ROCM if torch.version.hip is not None else _KDL_CUDA


def is_available() -> bool:
    """Whether this process can run the CUDA (or ROCm) leg at all: a vendor
    build of torch whose driver actually initializes."""
    return torch.cuda.is_available()


def as_cuda(t: torch.Tensor) -> torch.Tensor:
    """A `cuda` tensor aliasing a mojo tensor's memory. Zero copy.

    Same shape, strides, storage offset and dtype; the mojo device index is
    the CUDA ordinal (MAX enumerates accelerators in vendor order, as
    `triton_driver` relies on too). The returned tensor holds the mojo one
    alive through the DLPack deleter, so the alias may outlive every other
    reference to the source. It is a leaf with `requires_grad=False`: see
    `CudaAutogradFunction` for gradients.
    """
    if t.device.type != _MOJO:
        raise ValueError(f"expected a mojo tensor, got {t.device}")
    index = t.device.index if t.device.index is not None else 0
    if index >= device_module.device_count() - 1:
        raise ValueError(f"mojo:{index} is the MAX CPU device, which has no CUDA alias")
    capsule = torch.utils.dlpack.to_dlpack(t.detach())
    return torch.from_dlpack(retag_capsule(capsule, _vendor_dlpack_code(), index))


def as_mojo(t: torch.Tensor, index: int | None = None) -> torch.Tensor:
    """A `mojo` tensor aliasing a CUDA tensor's memory. Zero copy.

    The reverse of `as_cuda`, and the direction a package's *outputs* take:
    the memory then belongs to torch's CUDA caching allocator rather than to
    MAX, and stays alive because the capsule's deleter pins the CUDA tensor.
    """
    if t.device.type != "cuda":
        raise ValueError(f"expected a cuda tensor, got {t.device}")
    if index is None:
        index = t.device.index if t.device.index is not None else 0
    capsule = torch.utils.dlpack.to_dlpack(t.detach())
    return torch.from_dlpack(retag_capsule(capsule, _KDL_EXT_DEV, index))


# The two traversals below are hand-rolled rather than `tree_map`: this runs
# once per converted call (and once per op under the fallback), and pytree
# costs ~3 us per container -- more than a conversion itself. Only the exact
# built-in container types are descended into, so a tuple subclass such as
# `torch.Size` is passed through whole, which is what an op wants anyway.


class _Call:
    """The state of one converted call: which mojo device it is on, and which
    mojo tensor each CUDA alias came from.

    The second half is what makes in-place and `out=` ops behave: those return
    an argument, and torch's contract is that they return *that object*. An
    alias built fresh from the result would share the memory but not the
    identity, so `torch.add(a, b, out=c)` would stop satisfying `result is c`
    and an autograd in-place check would see a different tensor.
    """

    __slots__ = ("index", "originals")

    index: int | None
    originals: dict[int, torch.Tensor]

    def __init__(self):
        self.index = None
        self.originals = {}


def _to_cuda(x: object, call: _Call) -> object:
    if isinstance(x, torch.Tensor):
        if x.device.type != _MOJO:
            return x
        if call.index is None:
            call.index = x.device.index if x.device.index is not None else 0
        alias = as_cuda(x)
        call.originals[id(alias)] = x
        return alias
    if type(x) is list:
        return [_to_cuda(v, call) for v in x]
    if type(x) is tuple:
        return tuple(_to_cuda(v, call) for v in x)
    if type(x) is dict:
        return {k: _to_cuda(v, call) for k, v in x.items()}
    if isinstance(x, torch.device) and x.type == _MOJO:
        return torch.device("cuda", x.index if x.index is not None else 0)
    return x


def _to_mojo(x: object, originals: dict[int, torch.Tensor]) -> object:
    if isinstance(x, torch.Tensor):
        if x.device.type != "cuda":
            return x
        original = originals.get(id(x))
        # not `or`: a multi-element tensor has no truth value
        return as_mojo(x) if original is None else original
    if type(x) is list:
        return [_to_mojo(v, originals) for v in x]
    if type(x) is tuple:
        return tuple(_to_mojo(v, originals) for v in x)
    if type(x) is dict:
        return {k: _to_mojo(v, originals) for k, v in x.items()}
    return x


def _convert_args(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[tuple[object, ...], dict[str, object], _Call]:
    """Everything to CUDA in one pass."""
    call = _Call()
    cuda_args = tuple(_to_cuda(a, call) for a in args)
    cuda_kwargs = (
        {k: _to_cuda(v, call) for k, v in kwargs.items()} if kwargs else kwargs
    )
    return cuda_args, cuda_kwargs, call


# One ExternalStream per mojo stream: constructing one, and asking the shim for
# its vendor handle, cost more than everything else `on_mojo_stream` does, and
# a mojo stream keeps its handle for life.
_external_streams: dict[tuple[int, int], torch.cuda.Stream] = {}


def _external_stream(index: int) -> torch.cuda.Stream:
    stream = device_module.current_stream(index)
    cached = _external_streams.get((index, stream.stream_id))
    if cached is None:
        handle = device_module.stream_native_handle(stream)
        if handle == 0:
            raise RuntimeError(
                f"mojo:{index} has no vendor stream handle (the MAX CPU device "
                "cannot back a CUDA ExternalStream)"
            )
        cached = torch.cuda.ExternalStream(handle, device=index)
        _external_streams[(index, stream.stream_id)] = cached
    return cached


@contextlib.contextmanager
def on_mojo_stream(
    device: int | str | torch.device | None = None,
) -> Iterator[torch.cuda.Stream]:
    """Make torch.cuda's current stream the mojo current stream.

    A compiled extension launches on `at::cuda::getCurrentCUDAStream()`, so
    this is what orders its kernels with ours -- both sides then enqueue on
    one vendor stream and neither has to synchronize. The previous CUDA
    stream (and device) are restored on exit; the mojo current stream is not
    touched.
    """
    external = _external_stream(device_module._index(device))
    with torch.cuda.stream(external):
        yield external


def call_cuda(fn: Callable[..., object], *args: object, **kwargs: object) -> object:
    """Run a CUDA-only callable on mojo tensors.

    Every mojo tensor in `args`/`kwargs` becomes a CUDA alias, every
    `torch.device("mojo", i)` becomes `cuda:i`, the call runs with the mojo
    current stream installed as torch.cuda's, and every CUDA tensor coming
    back becomes a mojo alias. Tensors the callable mutates in place need no
    conversion back: an alias shares the memory, so the mojo tensor already
    holds the result.

    This is a leaf with respect to autograd (see the module docstring).
    """
    cuda_args, cuda_kwargs, call = _convert_args(args, kwargs)
    if call.index is None:
        raise ValueError("call_cuda needs at least one mojo tensor among its arguments")
    with on_mojo_stream(call.index):
        out = fn(*cuda_args, **cuda_kwargs)
    return _to_mojo(out, call.originals)


class CudaAutogradFunction(torch.autograd.Function):
    """Differentiable `call_cuda`, for a package that exposes its forward and
    backward entry points separately (causal-conv1d's
    `causal_conv1d_fwd_function` / `causal_conv1d_bwd_function`, and most
    kernel packages, because their own `autograd.Function` is written that
    way).

    `torch.autograd.grad` over the CUDA aliases would be the general answer,
    but it is not available here: registering a PrivateUse1 backend makes it
    torch's one accelerator, and from then on the engine's
    `TORCH_INTERNAL_ASSERT(opt_ready_stream && opt_parent_stream)` fires for
    any backward over CUDA tensors (see `require_cuda_autograd` in
    tests/conftest.py). Driving the two halves by hand keeps the autograd
    graph entirely on mojo tensors, where it works.

    `backward` returns one gradient per forward input, so `bwd` is called as
    ``bwd(grad_out, *saved)`` and must return that tuple (None for inputs
    that take no gradient).
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        fwd: Callable[..., torch.Tensor],
        bwd: Callable[..., tuple[torch.Tensor | None, ...]],
        *args: torch.Tensor,
    ) -> torch.Tensor:
        out = call_cuda(fwd, *args)
        if not isinstance(out, torch.Tensor):
            raise TypeError("the forward callable must return one tensor")
        ctx.bwd = bwd  # ty: ignore[unresolved-attribute] -- FunctionCtx takes arbitrary attributes
        ctx.save_for_backward(*args)
        return out

    @staticmethod
    def backward(
        ctx: FunctionCtx, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        grads = call_cuda(
            ctx.bwd,  # ty: ignore[unresolved-attribute] -- set in forward
            grad_out.contiguous(),
            *ctx.saved_tensors,  # ty: ignore[unresolved-attribute] -- FunctionCtx stub lacks it
        )
        if not isinstance(grads, tuple):
            raise TypeError("the backward callable must return a tuple of gradients")
        return (None, None, *grads)


def cuda_autograd(
    fwd: Callable[..., torch.Tensor],
    bwd: Callable[..., tuple[torch.Tensor | None, ...]],
) -> Callable[..., torch.Tensor]:
    """`fwd`/`bwd`, a package's two CUDA entry points, as one differentiable
    function of mojo tensors."""

    @functools.wraps(fwd)
    def call(*args: torch.Tensor) -> torch.Tensor:
        return CudaAutogradFunction.apply(fwd, bwd, *args)

    return call


# ---------------------------------------------------------------------------
# The generic fallback


# Every Library whose registrations must stay: a `torch.library.Library` has a
# weakref finalizer that calls `m.reset()`, so dropping one takes its
# registrations with it.
_fallback_libs: list[torch.library.Library] = []
_fallback_counts: dict[str, int] = {}


def _cuda_fallback(
    op: torch._ops.OpOverload, *args: object, **kwargs: object
) -> object:
    """Boxed fallback for the mojo dispatch key: run the op's CUDA kernel on
    CUDA aliases of the mojo arguments.

    The dispatcher only reaches a fallback for an op that has no kernel
    registered at this key, so every op the native backend implements is
    untouched -- this is exactly the complement of its registration list.
    """
    _fallback_counts[str(op)] = _fallback_counts.get(str(op), 0) + 1
    cuda_args, cuda_kwargs, call = _convert_args(args, kwargs)
    if call.index is None:
        # Nothing to alias; redispatching would land back here forever.
        raise NotImplementedError(
            f"{op} reached the mojo CUDA fallback with no mojo tensor to convert"
        )
    with on_mojo_stream(call.index):
        out = op(*cuda_args, **cuda_kwargs)
    return _to_mojo(out, call.originals)


def _convolution_backward_overrideable(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
    output_mask: list[bool],
) -> tuple[torch.Tensor | None, ...]:
    """`aten::convolution_backward` never reaches the fallback.

    It is CompositeExplicitAutograd and branches on the backend itself,
    sending anything that is not CPU / CUDA / MKLDNN to this stub -- which
    has a CompositeExplicitAutograd kernel of its own that only raises "use
    TORCH_LIBRARY_IMPL to override this function". A fallback fires where
    *no* kernel is registered, so it never sees either name. Registering the
    stub by hand is that TORCH_LIBRARY_IMPL, and it is what makes a
    convolution trainable on the mojo device while `convolution_backward`
    has no Mojo op.
    """
    bias_sizes = [weight.shape[1] * groups if transposed else weight.shape[0]]
    index = input.device.index if input.device.index is not None else 0
    with on_mojo_stream(index):
        grads = torch.ops.aten.convolution_backward(
            as_cuda(grad_output),
            as_cuda(input),
            as_cuda(weight),
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )
    return tuple(None if g is None else as_mojo(g) for g in grads)


# Ops that need a registration of their own rather than the fallback, because
# ATen already put a kernel at the mojo key for them (see the docstring above).
_EXPLICIT_ROUTES = {
    "aten::convolution_backward_overrideable": _convolution_backward_overrideable
}


def _install(lib: torch.library.Library, aten: torch.library.Library) -> None:
    lib.fallback(_cuda_fallback, "PrivateUse1")
    for name, fn in _EXPLICIT_ROUTES.items():
        aten.impl(name, fn, "PrivateUse1", allow_override=True)


def enable_cuda_fallback() -> None:
    """Route every op the mojo device does not implement through CUDA, for
    the rest of the process. Needs a CUDA build of torch; idempotent."""
    if _fallback_libs:
        return
    if not is_available():
        raise RuntimeError("the CUDA fallback needs a CUDA build of torch")
    lib = torch.library.Library("_", "IMPL")  # noqa: TOR901 -- a process-lifetime registration, by design
    aten = torch.library.Library("aten", "IMPL")  # noqa: TOR901 -- idem
    _install(lib, aten)
    _fallback_libs.extend((lib, aten))


@contextlib.contextmanager
def cuda_fallback() -> Iterator[None]:
    """The same fallback, only for the duration of a block.

    A dispatcher registration lives as long as the `Library` object that
    carries it, so scoping the libraries scopes the fallback. Use this rather
    than `enable_cuda_fallback` wherever silently routing an unimplemented op
    to CUDA would hide a missing mojo kernel from the code that follows.
    """
    if not is_available():
        raise RuntimeError("the CUDA fallback needs a CUDA build of torch")
    with (
        torch.library._scoped_library("_", "IMPL") as lib,
        torch.library._scoped_library("aten", "IMPL") as aten,
    ):
        _install(lib, aten)
        yield


def fallback_counts() -> dict[str, int]:
    """How many times each op went through the CUDA fallback (for tests and
    for finding out what a model is missing)."""
    return dict(_fallback_counts)
