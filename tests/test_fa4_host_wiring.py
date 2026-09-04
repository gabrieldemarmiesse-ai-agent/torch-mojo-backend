"""Host-only contracts for the vendored H100 FA4 integration (bf16 and f16)."""

import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

import torch_mojo_backend.eager_flash_attention as package
from torch_mojo_backend.eager_kernels import aten_fast
from torch_mojo_backend.mojo_device import mojo_device_autograd as autograd
from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor


def _device(*, arch: str = "sm_90a") -> SimpleNamespace:
    return SimpleNamespace(api="cuda", architecture_name=arch, label="gpu", id=0)


def _tensor(
    name: str,
    shape=(2, 4, 128, 64),
    *,
    dtype=None,
    device=None,
    ptr: int = 100,
    strides=None,
    contiguous: bool | None = None,
) -> SimpleNamespace:
    shape = tuple(shape)
    strides = aten_fast._row_major_strides(shape) if strides is None else tuple(strides)
    if contiguous is None:
        contiguous = strides == aten_fast._row_major_strides(shape)
    return SimpleNamespace(
        name=name,
        _shape=shape,
        _mojo_strides=strides,
        _offset=0,
        _dtype=dtype or aten_fast.DType.bfloat16,
        _device=device or _device(),
        _ptr=ptr,
        _itemsize=(dtype or aten_fast.DType.bfloat16).size_in_bytes,
        _numel=math.prod(shape),
        _is_contiguous=contiguous,
        _holder=object(),
    )


def _mt(tensor: SimpleNamespace | None) -> TorchMojoTensor:
    """`_tensor()` fakes carry every payload attribute the Tensor-/
    TorchMojoTensor-typed signatures under test read; SimpleNamespace isn't
    nominally either, so cast at the call boundary. Also covers the rare
    literal-``None`` call: `_fa4_bhsd_layout`/`_fa4_strided_bthd_layout`
    check ``tensor is None`` at runtime despite their non-Optional hint."""
    return cast(TorchMojoTensor, tensor)


def test_fa4_rejects_ineligible_regimes_before_loading_or_device_work(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an ineligible FA4 call reached device/compiler work")

    monkeypatch.setattr(package, "load_fa4_ops", forbidden)
    monkeypatch.setattr(aten_fast, "_fa4_native_bthd", forbidden)
    monkeypatch.setattr(aten_fast, "_alloc", forbidden)

    cases = [
        ((_tensor("q", dtype=aten_fast.DType.float32),), {}),
        ((_tensor("q", device=_device(arch="sm_80")),), {}),
        ((_tensor("q", shape=(2, 4, 96, 64)),), {}),
        ((_tensor("q", shape=(2, 4, 128, 32)),), {}),
        ((_tensor("q"),), {"is_causal": False}),
        ((_tensor("q"),), {"dropout_p": 0.1}),
        ((_tensor("q"),), {"attn_mask": object()}),
        ((_tensor("q"),), {"enable_gqa": True}),
    ]
    for (query,), overrides in cases:
        key = _tensor("k", dtype=query._dtype, device=query._device)
        value = _tensor("v", dtype=query._dtype, device=query._device)
        kwargs = {
            "attn_mask": None,
            "dropout_p": 0.0,
            "is_causal": True,
            "scale": None,
            "enable_gqa": False,
        }
        kwargs.update(overrides)
        assert (
            aten_fast.fast_fa4_16bit_d64_causal_forward(
                _mt(query), _mt(key), _mt(value), **kwargs
            )
            is aten_fast.NOT_HANDLED
        )


def test_fa4_forward_bridge_uses_dynamic_bthd_allocations(
    monkeypatch: pytest.MonkeyPatch,
):
    device = _device()
    # A fixed 3-tuple (not a list/tuple(generator)) so *public below unpacks
    # to a fixed arity instead of tuple[SimpleNamespace, ...].
    q_pub, k_pub, v_pub = (
        _tensor(name, shape=(3, 12, 256, 64), device=device, ptr=ptr)
        for name, ptr in zip(("q", "k", "v"), (10, 20, 30), strict=True)
    )
    public = (q_pub, k_pub, v_pub)
    native = {
        tensor.name: _tensor(
            f"{tensor.name}_native",
            shape=(3, 256, 12, 64),
            device=device,
            ptr=tensor._ptr + 100,
        )
        for tensor in public
    }
    allocations = []
    bridge_calls = []

    def alloc(shape, dtype, actual_device):
        result = _tensor(
            f"alloc{len(allocations)}",
            shape=shape,
            dtype=dtype,
            device=actual_device,
            ptr=1000 + len(allocations),
        )
        allocations.append(result)
        return result

    def transpose(tensor, dim0, dim1):
        assert (dim0, dim1) == (1, 2)
        shape = list(tensor._shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        return _tensor(
            "output", shape, dtype=tensor._dtype, device=device, ptr=tensor._ptr
        )

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(
        aten_fast, "_fa4_native_bthd", lambda tensor: native[tensor.name]
    )
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "fast_aten_transpose", transpose)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 9090)
    monkeypatch.setattr(
        package,
        "load_fa4_ops",
        lambda: SimpleNamespace(
            flash_attention_fwd_bf16_d64_causal=lambda *args: bridge_calls.append(args)
        ),
    )

    result = aten_fast.fast_fa4_16bit_d64_causal_forward(
        _mt(q_pub), _mt(k_pub), _mt(v_pub), is_causal=True, scale=0.125
    )
    assert isinstance(result, tuple)
    output, logsumexp, q_native, k_native, v_native = result

    assert output._shape == (3, 12, 256, 64)
    assert logsumexp._shape == (3, 12, 256)
    assert (q_native, k_native, v_native) == tuple(native[name] for name in "qkv")
    assert [(item._shape, item._dtype) for item in allocations] == [
        ((3, 256, 12, 64), aten_fast.DType.bfloat16),
        ((3, 12, 256), aten_fast.DType.float32),
    ]
    assert bridge_calls == [
        (
            110,
            120,
            130,
            allocations[0]._ptr,
            allocations[1]._ptr,
            3,
            256,
            12,
            0.125,
            9090,
        )
    ]


def test_fa4_forward_bridge_selects_f16_kernel_symbol_and_allocation(
    monkeypatch: pytest.MonkeyPatch,
):
    """f16 Q/K/V select the f16 bridge symbol and f16 output allocation.

    Mirrors ``test_fa4_forward_bridge_uses_dynamic_bthd_allocations`` above,
    only with f16 inputs: the bf16 bridge symbol must not be touched.
    """
    device = _device()
    # A fixed 3-tuple (not a list/tuple(generator)) so *public below unpacks
    # to a fixed arity instead of tuple[SimpleNamespace, ...].
    q_pub, k_pub, v_pub = (
        _tensor(
            name,
            shape=(3, 12, 256, 64),
            device=device,
            ptr=ptr,
            dtype=aten_fast.DType.float16,
        )
        for name, ptr in zip(("q", "k", "v"), (10, 20, 30), strict=True)
    )
    public = (q_pub, k_pub, v_pub)
    native = {
        tensor.name: _tensor(
            f"{tensor.name}_native",
            shape=(3, 256, 12, 64),
            device=device,
            ptr=tensor._ptr + 100,
            dtype=aten_fast.DType.float16,
        )
        for tensor in public
    }
    allocations = []
    bridge_calls = []

    def alloc(shape, dtype, actual_device):
        result = _tensor(
            f"alloc{len(allocations)}",
            shape=shape,
            dtype=dtype,
            device=actual_device,
            ptr=1000 + len(allocations),
        )
        allocations.append(result)
        return result

    def transpose(tensor, dim0, dim1):
        assert (dim0, dim1) == (1, 2)
        shape = list(tensor._shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        return _tensor(
            "output", shape, dtype=tensor._dtype, device=device, ptr=tensor._ptr
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("f16 inputs reached the bf16 bridge symbol")

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(
        aten_fast, "_fa4_native_bthd", lambda tensor: native[tensor.name]
    )
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "fast_aten_transpose", transpose)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 9090)
    monkeypatch.setattr(
        package,
        "load_fa4_ops",
        lambda: SimpleNamespace(
            flash_attention_fwd_bf16_d64_causal=forbidden,
            flash_attention_fwd_f16_d64_causal=lambda *args: bridge_calls.append(args),
        ),
    )

    result = aten_fast.fast_fa4_16bit_d64_causal_forward(
        _mt(q_pub), _mt(k_pub), _mt(v_pub), is_causal=True, scale=0.125
    )
    assert isinstance(result, tuple)
    output, logsumexp, q_native, k_native, v_native = result

    assert output._shape == (3, 12, 256, 64)
    assert output._dtype == aten_fast.DType.float16
    assert logsumexp._shape == (3, 12, 256)
    assert (q_native, k_native, v_native) == tuple(native[name] for name in "qkv")
    assert [(item._shape, item._dtype) for item in allocations] == [
        ((3, 256, 12, 64), aten_fast.DType.float16),
        ((3, 12, 256), aten_fast.DType.float32),
    ]
    assert bridge_calls == [
        (
            110,
            120,
            130,
            allocations[0]._ptr,
            allocations[1]._ptr,
            3,
            256,
            12,
            0.125,
            9090,
        )
    ]


def test_fa4_rejects_mixed_bf16_f16_inputs(monkeypatch: pytest.MonkeyPatch):
    """Q/K/V must share one dtype from {bf16, f16} -- mixing is not eligible."""
    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    device = _device()
    q = _tensor("q", dtype=aten_fast.DType.bfloat16, device=device)
    k = _tensor("k", dtype=aten_fast.DType.float16, device=device)
    v = _tensor("v", dtype=aten_fast.DType.bfloat16, device=device)
    assert (
        aten_fast._fa4_16bit_d64_causal_inputs(
            _mt(q), _mt(k), _mt(v), None, 0.0, True, None, False
        )
        is None
    )


def test_fa4_strided_layout_contract_is_strict():
    shape = (2, 256, 12, 64)
    token_stride = 3 * shape[2] * shape[3]
    strides = (shape[1] * token_stride, token_stride, 64, 1)
    eligible = _tensor(
        "q_native", shape=shape, ptr=0x1000, strides=strides, contiguous=False
    )
    assert aten_fast._fa4_strided_bthd_layout(_mt(eligible))
    # f16 is the same 2-byte width as bf16, so the same layout is eligible.
    eligible_f16 = _tensor(
        "q_native_f16",
        shape=shape,
        ptr=0x1000,
        strides=strides,
        contiguous=False,
        dtype=aten_fast.DType.float16,
    )
    assert aten_fast._fa4_strided_bthd_layout(_mt(eligible_f16))

    invalid = []
    for updates in (
        {"_ptr": eligible._ptr + 2},
        {"_shape": (2, 255, 12, 64)},
        {"_shape": (2, 256, 12, 32)},
        {"_mojo_strides": (strides[0], strides[1], 128, 1)},
        {"_mojo_strides": (strides[0], 12 * 64 - 8, 64, 1)},
        {"_mojo_strides": (strides[0] + 8, strides[1], 64, 1)},
        {"_mojo_strides": (strides[0], strides[1], 64, 2)},
        {"_mojo_strides": (strides[0], strides[1] + 2, 64, 1)},
        {"_dtype": aten_fast.DType.float32, "_itemsize": 4},
    ):
        candidate = SimpleNamespace(**vars(eligible))
        for name, value in updates.items():
            setattr(candidate, name, value)
        invalid.append(candidate)
    assert not any(
        aten_fast._fa4_strided_bthd_layout(_mt(tensor)) for tensor in invalid
    )


def test_fa4_canonical_fused_qkv_uses_zero_copy_strided_forward_bridge(
    monkeypatch: pytest.MonkeyPatch,
):
    device = _device()
    batch, heads, seqlen, head_dim = 2, 12, 256, 64
    token_stride = 3 * heads * head_dim
    batch_stride = seqlen * token_stride
    public_strides = (batch_stride, head_dim, token_stride, 1)
    pointers = (0x1000, 0x1600, 0x1C00)
    q_pub, k_pub, v_pub = (
        _tensor(
            name,
            shape=(batch, heads, seqlen, head_dim),
            device=device,
            ptr=ptr,
            strides=public_strides,
            contiguous=False,
        )
        for name, ptr in zip("qkv", pointers, strict=True)
    )
    allocations = []
    strided_calls = []

    def alloc(shape, dtype, actual_device):
        result = _tensor(
            f"alloc{len(allocations)}",
            shape=shape,
            dtype=dtype,
            device=actual_device,
            ptr=0x8000 + len(allocations) * 0x1000,
        )
        allocations.append(result)
        return result

    def transpose(tensor, dim0, dim1):
        assert (dim0, dim1) == (1, 2)
        shape = list(tensor._shape)
        strides = list(tensor._mojo_strides)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
        return _tensor(
            f"{tensor.name}_transpose",
            shape=shape,
            dtype=tensor._dtype,
            device=tensor._device,
            ptr=tensor._ptr,
            strides=strides,
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("canonical fused QKV reached a copy or old FA4 bridge")

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(aten_fast, "_tc", forbidden)
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "fast_aten_transpose", transpose)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 9090)
    monkeypatch.setattr(
        package,
        "load_fa4_ops",
        lambda: SimpleNamespace(
            flash_attention_fwd_bf16_d64_causal=forbidden,
            flash_attention_fwd_bf16_d64_causal_strided_qkv=lambda *args: (
                strided_calls.append(args)
            ),
        ),
    )

    result = aten_fast.fast_fa4_16bit_d64_causal_forward(
        _mt(q_pub), _mt(k_pub), _mt(v_pub), is_causal=True, scale=0.125
    )
    assert isinstance(result, tuple)
    output, logsumexp, q_native, k_native, v_native = result

    physical_strides = (batch_stride, token_stride, head_dim, 1)
    assert output._shape == (batch, heads, seqlen, head_dim)
    assert logsumexp._shape == (batch, heads, seqlen)
    assert tuple(tensor._ptr for tensor in (q_native, k_native, v_native)) == pointers
    assert all(
        tensor._mojo_strides == physical_strides
        for tensor in (q_native, k_native, v_native)
    )
    assert strided_calls == [
        (
            pointers[0],
            *physical_strides,
            pointers[1],
            *physical_strides,
            pointers[2],
            *physical_strides,
            allocations[0]._ptr,
            allocations[1]._ptr,
            batch,
            seqlen,
            heads,
            0.125,
            9090,
        )
    ]


def test_fa4_offset_view_public_layout_copies_and_uses_contiguous_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    """A public (B, H, S, D) tensor that is fully contiguous but whose base
    pointer is NOT 16-byte aligned (e.g. a sliced/offset view into a larger
    buffer) must be rejected by ``_fa4_bhsd_layout`` and fall back to the
    existing BTHD-materialize-and-copy path -- TMA descriptor creation for
    the BHSD-native path requires 16-byte alignment and a misaligned base
    pointer violates that even though the tensor is otherwise eligible.
    """
    device = _device()
    # +2 bytes (one bf16/f16 element) off a 16-byte-aligned address: still
    # fully contiguous, but ptr % 16 != 0.
    q_pub, k_pub, v_pub = (
        _tensor(name, shape=(2, 12, 256, 64), device=device, ptr=ptr)
        for name, ptr in zip("qkv", (0x1002, 0x2002, 0x3002), strict=True)
    )
    copied = []
    old_calls = []
    allocations = []

    def transpose(tensor, dim0, dim1):
        shape = list(tensor._shape)
        strides = list(tensor._mojo_strides)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
        return _tensor(
            tensor.name,
            shape=shape,
            dtype=tensor._dtype,
            device=tensor._device,
            ptr=tensor._ptr,
            strides=strides,
        )

    def make_contiguous(tensor):
        copied.append(tensor)
        return _tensor(
            f"{tensor.name}_copy",
            shape=tensor._shape,
            dtype=tensor._dtype,
            device=tensor._device,
            ptr=0x4000 + len(copied) * 0x1000,
        )

    def alloc(shape, dtype, actual_device):
        result = _tensor(
            f"alloc{len(allocations)}",
            shape=shape,
            dtype=dtype,
            device=actual_device,
            ptr=0x8000 + len(allocations) * 0x1000,
        )
        allocations.append(result)
        return result

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unsupported QKV selected the strided FA4 bridge")

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(aten_fast, "_tc", make_contiguous)
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "fast_aten_transpose", transpose)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 7070)
    monkeypatch.setattr(
        package,
        "load_fa4_ops",
        lambda: SimpleNamespace(
            flash_attention_fwd_bf16_d64_causal=lambda *args: old_calls.append(args),
            flash_attention_fwd_bf16_d64_causal_strided_qkv=forbidden,
        ),
    )

    result = aten_fast.fast_fa4_16bit_d64_causal_forward(
        _mt(q_pub), _mt(k_pub), _mt(v_pub), is_causal=True, scale=0.125
    )

    assert result is not aten_fast.NOT_HANDLED
    assert [tensor.name for tensor in copied] == ["q", "k", "v"]
    assert len(old_calls) == 1
    assert old_calls[0][:3] == (0x5000, 0x6000, 0x7000)


def test_fa4_forward_bridge_uses_bhsd_native_path_for_aligned_contiguous_public_qkv(
    monkeypatch: pytest.MonkeyPatch,
):
    """A public (B, H, S, D) tensor that is fully contiguous AND 16-byte
    aligned skips BTHD materialization entirely: no transpose, no copy, and
    the output is allocated directly in the (B, H, S, D) layout and
    returned as-is."""
    device = _device()
    # A fixed 3-tuple (not tuple(generator)) so *public below unpacks to a
    # fixed arity instead of tuple[SimpleNamespace, ...].
    q_pub, k_pub, v_pub = (
        _tensor(name, shape=(3, 12, 256, 64), device=device, ptr=ptr)
        for name, ptr in zip("qkv", (0x1000, 0x2000, 0x3000), strict=True)
    )
    public = (q_pub, k_pub, v_pub)
    allocations = []
    bridge_calls = []

    def alloc(shape, dtype, actual_device):
        result = _tensor(
            f"alloc{len(allocations)}",
            shape=shape,
            dtype=dtype,
            device=actual_device,
            ptr=0x9000 + len(allocations) * 0x1000,
        )
        allocations.append(result)
        return result

    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "aligned contiguous public QKV reached a copy or a non-bhsd bridge"
        )

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(aten_fast, "_fa4_native_bthd", forbidden)
    monkeypatch.setattr(aten_fast, "_tc", forbidden)
    monkeypatch.setattr(aten_fast, "fast_aten_transpose", forbidden)
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 5050)
    monkeypatch.setattr(
        package,
        "load_fa4_ops",
        lambda: SimpleNamespace(
            flash_attention_fwd_bf16_d64_causal=forbidden,
            flash_attention_fwd_bf16_d64_causal_strided_qkv=forbidden,
            flash_attention_fwd_bf16_d64_causal_bhsd=lambda *args: bridge_calls.append(
                args
            ),
        ),
    )

    result = aten_fast.fast_fa4_16bit_d64_causal_forward(
        _mt(q_pub), _mt(k_pub), _mt(v_pub), is_causal=True, scale=0.125
    )
    assert isinstance(result, tuple)
    output, logsumexp, q_native, k_native, v_native = result

    assert output is allocations[0]
    assert output._shape == (3, 12, 256, 64)
    assert logsumexp._shape == (3, 12, 256)
    assert (q_native, k_native, v_native) == public
    assert bridge_calls == [
        (
            0x1000,
            0x2000,
            0x3000,
            allocations[0]._ptr,
            allocations[1]._ptr,
            3,
            256,
            12,
            0.125,
            5050,
        )
    ]


def test_fa4_bhsd_layout_requires_contiguity_and_16_byte_alignment():
    base = _tensor("q", shape=(3, 12, 256, 64), ptr=0x1000)
    assert aten_fast._fa4_bhsd_layout(_mt(base))

    misaligned = SimpleNamespace(**vars(base))
    misaligned._ptr = base._ptr + 2
    assert not aten_fast._fa4_bhsd_layout(_mt(misaligned))

    non_contiguous = SimpleNamespace(**vars(base))
    non_contiguous._is_contiguous = False
    assert not aten_fast._fa4_bhsd_layout(_mt(non_contiguous))

    wrong_dtype = SimpleNamespace(**vars(base))
    wrong_dtype._dtype = aten_fast.DType.float32
    wrong_dtype._itemsize = 4
    assert not aten_fast._fa4_bhsd_layout(_mt(wrong_dtype))

    assert not aten_fast._fa4_bhsd_layout(_mt(None))


def test_direct_flash_aten_returns_real_lse_and_cuda_shaped_auxiliaries(
    monkeypatch: pytest.MonkeyPatch,
):
    device = _device()
    query, key, value = (
        _tensor(name, shape=(2, 8, 256, 64), device=device, ptr=ptr)
        for name, ptr in zip(("q", "k", "v"), (10, 20, 30), strict=True)
    )
    output = _tensor("output", shape=query._shape, device=device, ptr=40)
    logsumexp = _tensor(
        "lse", shape=(2, 8, 256), dtype=aten_fast.DType.float32, device=device, ptr=50
    )
    physical = tuple(
        _tensor(name, shape=(2, 256, 8, 64), device=device, ptr=ptr)
        for name, ptr in zip(("qn", "kn", "vn"), (60, 70, 80), strict=True)
    )
    allocations = []

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(
        aten_fast,
        "fast_fa4_16bit_d64_causal_forward",
        lambda *_args: (output, logsumexp, *physical),
    )

    def alloc(shape, dtype, actual_device):
        allocations.append((tuple(shape), dtype, actual_device))
        return _tensor(
            f"alloc{len(allocations)}", shape=shape, dtype=dtype, device=actual_device
        )

    monkeypatch.setattr(aten_fast, "_alloc", alloc)

    result = aten_fast.fast_aten__scaled_dot_product_flash_attention(
        _mt(query), _mt(key), _mt(value), dropout_p=0.0, is_causal=True
    )
    assert isinstance(result, tuple)

    assert result[:6] == (output, logsumexp, None, None, 256, 256)
    assert result[6]._dtype == aten_fast.DType.uint64
    assert result[7]._dtype == aten_fast.DType.uint64
    assert result[8]._shape == (0,)
    assert allocations == [
        ((2,), aten_fast.DType.uint64, device),
        ((), aten_fast.DType.uint64, device),
        ((0,), aten_fast.DType.bfloat16, device),
    ]
    assert (
        aten_fast.fast_aten__scaled_dot_product_flash_attention(
            _mt(query),
            _mt(key),
            _mt(value),
            dropout_p=0.0,
            is_causal=True,
            return_debug_mask=True,
        )
        is aten_fast.NOT_HANDLED
    )


def test_direct_flash_backward_materializes_strided_logsumexp(
    monkeypatch: pytest.MonkeyPatch,
):
    device = _device()
    shape = (2, 4, 128, 64)
    query, key, value = (
        _tensor(name, shape=shape, device=device, ptr=ptr)
        for name, ptr in zip(("q", "k", "v"), (0x1000, 0x2000, 0x3000), strict=True)
    )
    grad = _tensor("grad", shape=shape, device=device, ptr=0x4000)
    output = _tensor("output", shape=shape, device=device, ptr=0x5000)
    lse = _tensor(
        "lse",
        shape=(2, 4, 128),
        dtype=aten_fast.DType.float32,
        device=device,
        ptr=0x6000,
        strides=(4 * 128 * 2, 128 * 2, 2),
        contiguous=False,
    )
    contiguous_lse = _tensor(
        "lse_contiguous",
        shape=lse._shape,
        dtype=aten_fast.DType.float32,
        device=device,
        ptr=0x7000,
    )
    native = {
        tensor.name: _tensor(
            f"{tensor.name}_native",
            shape=(2, 128, 4, 64),
            device=device,
            ptr=tensor._ptr,
        )
        for tensor in (query, key, value)
    }
    backward_calls = []

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(
        aten_fast, "_fa4_native_bthd", lambda tensor: native[tensor.name]
    )

    def make_contiguous(tensor):
        assert tensor is lse
        return contiguous_lse

    monkeypatch.setattr(aten_fast, "_tc", make_contiguous)
    monkeypatch.setattr(
        aten_fast,
        "fast_fa4_16bit_d64_causal_backward",
        lambda *args: backward_calls.append(args) or "gradients",
    )

    result = aten_fast.fast_aten__scaled_dot_product_flash_attention_backward(
        _mt(grad),
        _mt(query),
        _mt(key),
        _mt(value),
        _mt(output),
        _mt(lse),
        None,
        None,
        128,
        128,
        0.0,
        True,
        # Dropout is 0.0, so philox_seed/philox_offset are read but never
        # dereferenced -- production accepts this despite the non-Optional
        # hint (aten_fast.py:8414, `_ = philox_seed, philox_offset`).
        cast(torch.Tensor, None),
        cast(torch.Tensor, None),
        scale=0.125,
    )

    assert result == "gradients"
    assert len(backward_calls) == 1
    assert backward_calls[0][4] is contiguous_lse


def test_fa4_saved_variable_recompute_rederives_natives_independently(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression test for the SavedVariable-recompute fallback (``out``/
    ``logsumexp`` unpacked as bare tensors without their Python payload).

    Before this fix, the recompute branch trusted the *natives* returned by
    the inner ``fast_fa4_16bit_d64_causal_forward`` recompute call. Now that
    the BHSD-eligible fast path returns the untouched PUBLIC q/k/v as those
    same tuple slots (see ``fast_fa4_16bit_d64_causal_forward``), blindly
    reusing them here would feed BHSD-shaped tensors into the BTHD-only bwd
    kernel ABI. This test simulates exactly that: the mocked recompute
    returns the public q/k/v unchanged (mirroring the BHSD fast path's
    contract), and asserts the backward bridge is handed independently
    rederived BTHD natives instead -- never the recompute's raw tuple
    elements.
    """
    device = _device()
    shape = (2, 4, 128, 64)
    query, key, value = (
        _tensor(name, shape=shape, device=device, ptr=ptr)
        for name, ptr in zip(("q", "k", "v"), (0x1000, 0x2000, 0x3000), strict=True)
    )
    grad = _tensor("grad", shape=shape, device=device, ptr=0x4000)
    recomputed_output = _tensor(
        "recomputed_output", shape=shape, device=device, ptr=0x5000
    )
    recomputed_lse = _tensor(
        "recomputed_lse",
        shape=(2, 4, 128),
        dtype=aten_fast.DType.float32,
        device=device,
        ptr=0x6000,
    )
    bthd_native = {
        tensor.name: _tensor(
            f"{tensor.name}_bthd_native",
            shape=(2, 128, 4, 64),
            device=device,
            ptr=tensor._ptr,
        )
        for tensor in (query, key, value)
    }
    native_calls = []
    backward_calls = []

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(
        aten_fast, "_fa4_16bit_d64_causal_inputs", lambda *args: (query, key, value)
    )

    def fake_recompute(q, k, v, *_args):
        # Mirrors the real BHSD fast path's contract: the "natives" slots
        # are the untouched PUBLIC q/k/v, not BTHD-shaped tensors.
        assert (q, k, v) == (query, key, value)
        return recomputed_output, recomputed_lse, q, k, v

    monkeypatch.setattr(aten_fast, "fast_fa4_16bit_d64_causal_forward", fake_recompute)

    def fake_native_bthd(tensor):
        native_calls.append(tensor)
        return bthd_native[tensor.name]

    monkeypatch.setattr(aten_fast, "_fa4_native_bthd", fake_native_bthd)
    monkeypatch.setattr(aten_fast, "_tc", lambda tensor: tensor)
    monkeypatch.setattr(
        aten_fast,
        "fast_fa4_16bit_d64_causal_backward",
        lambda *args: backward_calls.append(args) or "gradients",
    )

    result = aten_fast.fast_aten__scaled_dot_product_flash_attention_backward(
        _mt(grad),
        _mt(query),
        _mt(key),
        _mt(value),
        # out/logsumexp: unpacked as a bare tensor, triggers the recompute
        # path -- `_t(None) is None` once _t is monkeypatched to identity
        # above, despite the non-Optional hint.
        cast(torch.Tensor, None),
        cast(torch.Tensor, None),
        None,
        None,
        128,
        128,
        0.0,
        True,
        # Dropout is 0.0, so philox_seed/philox_offset are read but never
        # dereferenced -- production accepts this despite the non-Optional
        # hint (aten_fast.py:8414, `_ = philox_seed, philox_offset`).
        cast(torch.Tensor, None),
        cast(torch.Tensor, None),
        scale=0.125,
    )

    assert result == "gradients"
    # Every one of q/k/v must have been independently rederived...
    assert native_calls == [query, key, value]
    assert len(backward_calls) == 1
    q_native, k_native, v_native = backward_calls[0][:3]
    # ...and the backward call must have received those rederived BTHD
    # natives, never the recompute's raw (public-shaped) return values. This
    # is the exact bug the fix closes: pre-fix, q_native/k_native/v_native
    # here would have been `query`/`key`/`value` themselves.
    assert (q_native, k_native, v_native) == (
        bthd_native["q"],
        bthd_native["k"],
        bthd_native["v"],
    )
    assert q_native is not query and k_native is not key and v_native is not value


def test_fa4_combined_backward_bridge_allocates_exact_scratch(
    monkeypatch: pytest.MonkeyPatch,
):
    device = _device()
    q_native, k_native, v_native = [
        _tensor(name, shape=(2, 384, 4, 64), device=device, ptr=ptr)
        for name, ptr in zip(("q", "k", "v"), (11, 22, 33), strict=True)
    ]
    output = _tensor("output", shape=(2, 4, 384, 64), device=device, ptr=44)
    logsumexp = _tensor(
        "lse", shape=(2, 4, 384), dtype=aten_fast.DType.float32, device=device, ptr=55
    )
    grad_output = _tensor("grad", shape=(2, 4, 384, 64), device=device, ptr=66)
    physical = {
        "output": _tensor("out_native", shape=(2, 384, 4, 64), device=device, ptr=77),
        "grad": _tensor("dout_native", shape=(2, 384, 4, 64), device=device, ptr=88),
    }
    allocations = []
    bridge_calls = []

    def alloc(shape, dtype, actual_device):
        result = _tensor(
            f"alloc{len(allocations)}",
            shape=shape,
            dtype=dtype,
            device=actual_device,
            ptr=1000 + len(allocations),
        )
        allocations.append(result)
        return result

    def transpose(tensor, dim0, dim1):
        shape = list(tensor._shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        return _tensor(
            f"public_{tensor.name}",
            shape,
            dtype=tensor._dtype,
            device=device,
            ptr=tensor._ptr,
        )

    monkeypatch.setattr(
        aten_fast, "_fa4_native_bthd", lambda tensor: physical[tensor.name]
    )
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "fast_aten_transpose", transpose)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 8080)
    monkeypatch.setattr(
        package,
        "load_fa4_ops",
        lambda: SimpleNamespace(
            flash_attention_bwd_bf16_d64_causal=lambda *args: bridge_calls.append(args)
        ),
    )

    gradients = aten_fast.fast_fa4_16bit_d64_causal_backward(
        _mt(q_native),
        _mt(k_native),
        _mt(v_native),
        _mt(output),
        _mt(logsumexp),
        _mt(grad_output),
        0.125,
    )
    assert isinstance(gradients, tuple)

    assert [gradient._shape for gradient in gradients] == [(2, 4, 384, 64)] * 3
    assert [(item._shape, item._dtype) for item in allocations] == [
        ((2, 384, 4, 64), aten_fast.DType.bfloat16),
        ((2, 384, 4, 64), aten_fast.DType.bfloat16),
        ((2, 384, 4, 64), aten_fast.DType.bfloat16),
        ((2, 4, 384), aten_fast.DType.float32),
        ((2, 4, 384), aten_fast.DType.float32),
        ((2 * 4 * 384 * 64,), aten_fast.DType.float32),
    ]
    assert bridge_calls == [
        (
            11,
            22,
            33,
            77,
            88,
            55,
            1000,
            1001,
            1002,
            1003,
            1004,
            1005,
            2,
            384,
            4,
            0.125,
            8080,
        )
    ]


def test_fa4_canonical_fused_qkv_uses_strided_backward_bridge(
    monkeypatch: pytest.MonkeyPatch,
):
    device = _device()
    batch, seqlen, heads, head_dim = 2, 256, 12, 64
    token_stride = 3 * heads * head_dim
    qkv_strides = (seqlen * token_stride, token_stride, head_dim, 1)
    qkv_pointers = (0x1000, 0x1600, 0x1C00)
    q_native, k_native, v_native = tuple(
        _tensor(
            name,
            shape=(batch, seqlen, heads, head_dim),
            device=device,
            ptr=ptr,
            strides=qkv_strides,
            contiguous=False,
        )
        for name, ptr in zip("qkv", qkv_pointers, strict=True)
    )
    public_shape = (batch, heads, seqlen, head_dim)
    public_strides = (seqlen * heads * head_dim, head_dim, heads * head_dim, 1)
    output = _tensor(
        "output",
        shape=public_shape,
        device=device,
        ptr=0x3000,
        strides=public_strides,
        contiguous=False,
    )
    grad_output = _tensor(
        "grad",
        shape=public_shape,
        device=device,
        ptr=0x4000,
        strides=public_strides,
        contiguous=False,
    )
    logsumexp = _tensor(
        "lse",
        shape=(batch, heads, seqlen),
        dtype=aten_fast.DType.float32,
        device=device,
        ptr=0x5000,
    )
    allocations = []
    strided_calls = []

    def alloc(shape, dtype, actual_device):
        result = _tensor(
            f"alloc{len(allocations)}",
            shape=shape,
            dtype=dtype,
            device=actual_device,
            ptr=0x8000 + len(allocations) * 0x1000,
        )
        allocations.append(result)
        return result

    def transpose(tensor, dim0, dim1):
        shape = list(tensor._shape)
        strides = list(tensor._mojo_strides)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
        return _tensor(
            f"{tensor.name}_transpose",
            shape=shape,
            dtype=tensor._dtype,
            device=tensor._device,
            ptr=tensor._ptr,
            strides=strides,
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("strided QKV reached a copy or old FA4 bridge")

    monkeypatch.setattr(aten_fast, "_tc", forbidden)
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "fast_aten_transpose", transpose)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 6060)
    monkeypatch.setattr(
        package,
        "load_fa4_ops",
        lambda: SimpleNamespace(
            flash_attention_bwd_bf16_d64_causal=forbidden,
            flash_attention_bwd_bf16_d64_causal_strided_qkv=lambda *args: (
                strided_calls.append(args)
            ),
        ),
    )

    gradients = aten_fast.fast_fa4_16bit_d64_causal_backward(
        _mt(q_native),
        _mt(k_native),
        _mt(v_native),
        _mt(output),
        _mt(logsumexp),
        _mt(grad_output),
        0.125,
    )
    assert isinstance(gradients, tuple)

    assert [gradient._shape for gradient in gradients] == [public_shape] * 3
    assert strided_calls == [
        (
            qkv_pointers[0],
            *qkv_strides,
            qkv_pointers[1],
            *qkv_strides,
            qkv_pointers[2],
            *qkv_strides,
            output._ptr,
            grad_output._ptr,
            logsumexp._ptr,
            allocations[0]._ptr,
            allocations[1]._ptr,
            allocations[2]._ptr,
            allocations[3]._ptr,
            allocations[4]._ptr,
            allocations[5]._ptr,
            batch,
            seqlen,
            heads,
            0.125,
            6060,
        )
    ]


def test_fa4_eligible_backward_routes_to_native_flash_with_public_inputs(
    monkeypatch: pytest.MonkeyPatch,
):
    """An eligible FA4 call that needs gradients goes to the lower ATen pair.

    The custom SDPA Function used to own the FA4 saves itself; PyTorch's
    generated autograd owns them now, so the eligibility gate must hand the
    *public* BHTD q/k/v to ``aten::_scaled_dot_product_flash_attention`` (the
    physical BTHD copies are made inside the kernel bridge and are not what
    gets saved) and neither custom Function may be entered.
    """
    # A literal 3-tuple (not tuple(generator)) so *public below unpacks to a
    # fixed arity instead of tuple[T, ...].
    q_pub, k_pub, v_pub = (
        _tensor(name, ptr=index) for index, name in enumerate("qkv", 1)
    )
    public = (q_pub, k_pub, v_pub)
    for tensor in public:
        tensor.requires_grad = True
    output = _tensor("output", ptr=200)
    lse = _tensor("lse", shape=(2, 4, 128), dtype=aten_fast.DType.float32, ptr=201)
    eligible_with = []
    flash_calls = []

    def eligible(*args):
        eligible_with.append(args)
        return public

    def flash(*args, **kwargs):
        flash_calls.append((args, kwargs))
        return (output, lse)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an eligible FA4 call built a custom autograd node")

    monkeypatch.setattr(aten_fast, "_fa4_16bit_d64_causal_inputs", eligible)
    monkeypatch.setattr(
        torch.ops.aten._scaled_dot_product_flash_attention, "default", flash
    )
    monkeypatch.setattr(autograd._ScaledDotProductAttentionAutograd, "apply", forbidden)
    monkeypatch.setattr(autograd._FusedFlashAttentionAutograd, "apply", forbidden)

    actual = autograd._scaled_dot_product_attention_autograd(
        _mt(q_pub), _mt(k_pub), _mt(v_pub), None, 0.0, True, scale=0.125
    )

    assert actual is output
    assert eligible_with == [(*public, None, 0.0, True, 0.125, False)]
    assert flash_calls == [((*public, 0.0, True, False), {"scale": 0.125})]


def test_vendored_fa4_sources_have_no_torch_cuda_or_internal_sync():
    source_dir = Path(aten_fast.__file__).parents[1] / "eager_flash_attention"
    source_by_name = {path.name: path.read_text() for path in source_dir.glob("*.mojo")}
    sources = "\n".join(source_by_name.values())
    assert "torch.cuda" not in sources
    assert "ctx.synchronize()" not in sources

    # All seven launches -- dense fwd, strided_qkv fwd, and BHSD-native fwd
    # in fa4_fwd_launch.mojo, the phase-2c self-loading BHSD-native fwd in
    # fa4_fwd_selfload_launch.mojo, plus bwd preprocess/main/convert in
    # fa4_bwd_launch.mojo -- share compiled code by specialization and raw
    # context identity.  B/S/H, pointers, descriptors, and launch grids stay
    # runtime values, so changing model or batch shapes does not force
    # recompilation.
    launchers = (
        source_by_name["fa4_fwd_launch.mojo"]
        + source_by_name["fa4_fwd_selfload_launch.mojo"]
        + source_by_name["fa4_bwd_launch.mojo"]
    )
    cache = source_by_name["fa4_launch_cache.mojo"]
    assert launchers.count("enqueue_fa4_cached[") == 7
    assert ".compile_function[" not in launchers
    assert "_CTX{context_identity}" in cache
