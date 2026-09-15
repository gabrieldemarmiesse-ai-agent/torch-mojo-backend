"""Torchvision detection kernels through public APIs on the mojo GPU."""

from __future__ import annotations

import fcntl
import subprocess
import sys
import textwrap

import pytest
import torch

from tests.native.conftest import skip_if_metal

vision = pytest.importorskip("torchvision")
DTYPES = (torch.float32, torch.float16, torch.float64)


def _dtype_supported(device: str, dtype: torch.dtype):
    if dtype == torch.float64:
        skip_if_metal(device, "Metal does not support float64")


def _boxes(count: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(731)
    starts = torch.rand(count, 2, generator=generator) * 20
    sizes = torch.rand(count, 2, generator=generator) * 30 + 1
    return torch.cat((starts, starts + sizes), dim=1).to(dtype)


def _rois(scale: float = 1.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    result = torch.tensor(
        [
            [0, 1.25, 2.5, 18.75, 20.25],
            [1, -4, -3, 15, 17],
            [2, 45, 29, 58, 42],
            [0, -20, -15, -8, -4],
            [2, 60, 45, 71, 56],
            [1, 8, 8, 8, 8],
        ],
        dtype=dtype,
    )
    result[:, 1:] /= scale
    return result


def _roi_op(
    kind: str,
    x: torch.Tensor,
    rois: torch.Tensor,
    output: tuple[int, int] = (7, 5),
    scale: float = 1.0,
    sampling: int = 2,
    aligned: bool = False,
) -> torch.Tensor:
    if kind == "align":
        return vision.ops.roi_align(x, rois, output, scale, sampling, aligned=aligned)
    return vision.ops.roi_pool(x, rois, output, scale)


def _assert_close(got: torch.Tensor, want: torch.Tensor, dtype: torch.dtype):
    tolerance = {torch.float16: 2e-2, torch.float32: 3e-5, torch.float64: 2e-12}[dtype]
    torch.testing.assert_close(
        got.cpu(), want.to(dtype), rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("count", [0, 1, 47, 3073])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_nms(mojo_gpu: str, dtype: torch.dtype, count: int, threshold: float):
    _dtype_supported(mojo_gpu, dtype)
    boxes = _boxes(count, dtype)
    scores = torch.rand(count, generator=torch.Generator().manual_seed(171)).to(dtype)
    reference_dtype = torch.float32 if dtype == torch.float16 else dtype
    want = vision.ops.nms(
        boxes.to(reference_dtype), scores.to(reference_dtype), threshold
    )
    got = vision.ops.nms(boxes.to(mojo_gpu), scores.to(mojo_gpu), threshold)
    assert got.dtype == torch.int64
    assert got.device.type == "mojo"
    torch.testing.assert_close(got.cpu(), want)


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_nms_ties_and_degenerate_boxes(mojo_gpu: str, threshold: float):
    boxes = torch.cat((_boxes(127), torch.zeros(3, 4), torch.ones(4, 4)))
    scores = torch.ones(boxes.shape[0])
    want = vision.ops.nms(boxes, scores, threshold)
    got = vision.ops.nms(boxes.to(mojo_gpu), scores.to(mojo_gpu), threshold)
    # Equal scores use the CPU's stable input order, including zero-area boxes.
    torch.testing.assert_close(got.cpu(), want)


def test_batched_nms(mojo_gpu: str):
    boxes = _boxes(301)
    scores = torch.linspace(0.001, 1.0, boxes.shape[0])
    labels = torch.arange(boxes.shape[0]) % 4
    want = vision.ops.batched_nms(boxes, scores, labels, 0.5)
    got = vision.ops.batched_nms(
        boxes.to(mojo_gpu), scores.to(mojo_gpu), labels.to(mojo_gpu), 0.5
    )
    torch.testing.assert_close(got.cpu(), want)


@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("sampling", [-1, 0, 2])
@pytest.mark.parametrize("scale", [1.0, 0.25, 1 / 16])
def test_roi_align_forward_backward(
    mojo_gpu: str, aligned: bool, sampling: int, scale: float
):
    _check_roi(mojo_gpu, "align", torch.float32, scale, sampling, aligned)


@pytest.mark.parametrize("kind", ["align", "pool"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_roi_dtypes(mojo_gpu: str, kind: str, dtype: torch.dtype):
    _dtype_supported(mojo_gpu, dtype)
    _check_roi(mojo_gpu, kind, dtype, 1.0, 2, True)


@pytest.mark.parametrize("scale", [1.0, 0.25, 1 / 16])
def test_roi_pool_forward_backward(mojo_gpu: str, scale: float):
    _check_roi(mojo_gpu, "pool", torch.float32, scale, 2, False)


def _check_roi(
    device: str,
    kind: str,
    dtype: torch.dtype,
    scale: float,
    sampling: int,
    aligned: bool,
):
    data = torch.randn(3, 7, 37, 53, generator=torch.Generator().manual_seed(172)).to(
        dtype
    )
    reference_dtype = torch.float32 if dtype == torch.float16 else dtype
    x = data.to(reference_dtype).detach().requires_grad_()
    ours = data.to(device).detach().requires_grad_()
    rois = _rois(scale, dtype)
    want = _roi_op(
        kind,
        x,
        rois.to(reference_dtype),
        scale=scale,
        sampling=sampling,
        aligned=aligned,
    )
    got = _roi_op(
        kind, ours, rois.to(device), scale=scale, sampling=sampling, aligned=aligned
    )
    assert got.dtype == dtype
    _assert_close(got, want, dtype)
    grad = torch.randn(got.shape, generator=torch.Generator().manual_seed(173)).to(
        dtype
    )
    want.backward(grad.to(reference_dtype))
    got.backward(grad.to(device))
    assert ours.grad is not None and x.grad is not None
    _assert_close(ours.grad, x.grad, dtype)
    first_grad = ours.grad.cpu()
    ours.grad = None
    _roi_op(
        kind, ours, rois.to(device), scale=scale, sampling=sampling, aligned=aligned
    ).backward(grad.to(device))
    assert ours.grad is not None
    torch.testing.assert_close(ours.grad.cpu(), first_grad, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["align", "pool"])
def test_roi_empty_forward_backward(mojo_gpu: str, kind: str):
    x = torch.randn(2, 3, 13, 17, requires_grad=True)
    ours = x.detach().to(mojo_gpu).requires_grad_()
    rois = torch.empty(0, 5)
    want = _roi_op(kind, x, rois)
    got = _roi_op(kind, ours, rois.to(mojo_gpu))
    torch.testing.assert_close(got.cpu(), want)
    want.sum().backward()
    got.sum().backward()
    assert ours.grad is not None and x.grad is not None
    torch.testing.assert_close(ours.grad.cpu(), x.grad)


@pytest.mark.parametrize("kind", ["align", "pool"])
def test_roi_noncontiguous(mojo_gpu: str, kind: str):
    data = torch.randn(3, 7, 53, 37)
    rois = _rois()
    want = _roi_op(kind, data.transpose(2, 3), rois)
    try:
        got = _roi_op(kind, data.to(mojo_gpu).transpose(2, 3), rois.to(mojo_gpu))
    except NotImplementedError as exc:
        assert "contigu" in str(exc).lower()
    else:
        _assert_close(got, want, torch.float32)


def test_roi_pool_argmax(mojo_gpu: str):
    x = torch.arange(2 * 3 * 9 * 11, dtype=torch.float32).reshape(2, 3, 9, 11)
    rois = torch.tensor([[0, -3, -2, 8, 7], [1, 1, 2, 9, 8], [0, -9, -8, -6, -4]])
    want, want_argmax = torch.ops.torchvision.roi_pool(x, rois.float(), 1.0, 3, 4)
    got, got_argmax = torch.ops.torchvision.roi_pool(
        x.to(mojo_gpu), rois.float().to(mojo_gpu), 1.0, 3, 4
    )
    torch.testing.assert_close(got.cpu(), want)
    torch.testing.assert_close(got_argmax.cpu(), want_argmax)


@pytest.mark.parametrize("kind", ["align", "pool"])
def test_roi_numerical_gradient(mojo_gpu: str, kind: str):
    _dtype_supported(mojo_gpu, torch.float64)
    data = torch.randn(
        1, 1, 4, 5, generator=torch.Generator().manual_seed(74), dtype=torch.float64
    )
    rois = torch.tensor([[0, 0.2, 0.3, 3.5, 3.2]], dtype=torch.float64)
    ours = data.to(mojo_gpu).requires_grad_()
    output = _roi_op(kind, ours, rois.to(mojo_gpu), output=(2, 2))
    output.backward(torch.ones(output.shape, dtype=output.dtype).to(mojo_gpu))
    assert ours.grad is not None
    gradient = ours.grad.cpu().flatten()
    eps = 1e-5
    for index in range(data.numel()):
        plus, minus = data.clone(), data.clone()
        plus.flatten()[index] += eps
        minus.flatten()[index] -= eps
        numerical = (
            _roi_op(kind, plus, rois, output=(2, 2)).sum()
            - _roi_op(kind, minus, rois, output=(2, 2)).sum()
        ) / (2 * eps)
        torch.testing.assert_close(gradient[index], numerical, atol=1e-8, rtol=1e-8)


def test_roi_align_module(mojo_gpu: str):
    module = vision.ops.RoIAlign((7, 5), 0.25, 2, aligned=True)
    x = torch.randn(3, 7, 37, 53)
    rois = _rois(0.25)
    torch.testing.assert_close(
        module(x.to(mojo_gpu), rois.to(mojo_gpu)).cpu(),
        module(x, rois),
        rtol=3e-5,
        atol=3e-5,
    )


def test_detection_roi_heads(mojo_gpu: str):
    pool = vision.ops.MultiScaleRoIAlign(["0", "1"], output_size=3, sampling_ratio=2)
    heads = vision.models.detection.roi_heads.RoIHeads(
        pool,
        torch.nn.Identity(),
        torch.nn.Identity(),
        0.5,
        0.5,
        32,
        0.25,
        None,
        0.05,
        0.5,
        20,
    )
    features = {"0": torch.randn(2, 3, 32, 40), "1": torch.randn(2, 3, 16, 20)}
    proposals = [_boxes(13) * 2, _boxes(11) * 2]
    for boxes in proposals:
        boxes[0] = torch.tensor([0, 0, 150, 120])
    shapes = [(128, 160), (128, 160)]
    ours_features = {name: value.to(mojo_gpu) for name, value in features.items()}
    ours_proposals = [boxes.to(mojo_gpu) for boxes in proposals]
    want = heads.box_roi_pool(features, proposals, shapes)
    got = heads.box_roi_pool(ours_features, ours_proposals, shapes)
    torch.testing.assert_close(got.cpu(), want, atol=3e-5, rtol=3e-5)
    logits = torch.randn(24, 3, generator=torch.Generator().manual_seed(32))
    regression = torch.zeros(24, 12)
    want_result = heads.postprocess_detections(logits, regression, proposals, shapes)
    got_result = heads.postprocess_detections(
        logits.to(mojo_gpu), regression.to(mojo_gpu), ours_proposals, shapes
    )
    for got_list, want_list in zip(got_result, want_result, strict=True):
        for result, reference in zip(got_list, want_list, strict=True):
            torch.testing.assert_close(result.cpu(), reference, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_autocast(mojo_gpu: str, dtype: torch.dtype):
    x = torch.randn(3, 7, 37, 53).to(dtype)
    rois = _rois(dtype=dtype)
    boxes = _boxes(71, dtype)
    scores = torch.linspace(0.01, 1, 71).to(dtype)
    with torch.autocast("mojo", dtype=torch.float16):
        got = vision.ops.roi_align(
            x.to(mojo_gpu), rois.to(mojo_gpu), (7, 5), sampling_ratio=2
        )
        keep = vision.ops.nms(boxes.to(mojo_gpu), scores.to(mojo_gpu), 0.5)
    # Torchvision 0.26 executes ROIAlign in fp32, then restores the input dtype.
    want = vision.ops.roi_align(x.float(), rois.float(), (7, 5), sampling_ratio=2).to(
        dtype
    )
    assert got.dtype == dtype
    _assert_close(got, want, dtype)
    assert keep.dtype == torch.int64
    torch.testing.assert_close(
        keep.cpu(), vision.ops.nms(boxes.float(), scores.float(), 0.5)
    )


@pytest.mark.parametrize("vision_first", [False, True])
def test_import_order(mojo_gpu: str, vision_first: bool):
    script = textwrap.dedent(f"""
        import torch
        if {vision_first!r}:
            import torchvision
        from torch_mojo_backend import register_mojo_devices
        register_mojo_devices()
        import torchvision
        boxes = torch.tensor([[0., 0., 4., 4.], [1., 1., 3., 3.]])
        scores = torch.tensor([0.9, 0.8])
        expected = torchvision.ops.nms(boxes, scores, 0.5)
        actual = torchvision.ops.nms(boxes.to({mojo_gpu!r}), scores.to({mojo_gpu!r}), 0.5)
        torch.testing.assert_close(actual.cpu(), expected)
        x = torch.randn(1, 2, 8, 8)
        rois = torch.tensor([[0., 1., 1., 6., 6.]])
        expected = torchvision.ops.roi_align(x, rois, (3, 3), sampling_ratio=2)
        actual = torchvision.ops.roi_align(x.to({mojo_gpu!r}), rois.to({mojo_gpu!r}), (3, 3), sampling_ratio=2)
        torch.testing.assert_close(actual.cpu(), expected)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["align", "pool", "nms"])
def test_unsupported_dtype(mojo_gpu: str, kind: str):
    if kind == "nms":
        with pytest.raises(NotImplementedError, match="dtype|float"):
            vision.ops.nms(
                _boxes(4).to(torch.bfloat16).to(mojo_gpu),
                torch.ones(4, dtype=torch.bfloat16, device=mojo_gpu),
                0.5,
            )
    else:
        with pytest.raises(NotImplementedError, match="dtype|float"):
            _roi_op(
                kind,
                torch.ones(3, 2, 37, 53, dtype=torch.bfloat16, device=mojo_gpu),
                _rois(dtype=torch.bfloat16).to(mojo_gpu),
            )


@pytest.mark.parametrize("kind", ["align", "pool", "nms"])
def test_wrong_device(mojo_gpu: str, kind: str):
    with pytest.raises(NotImplementedError, match="same mojo|same.*device"):
        if kind == "nms":
            vision.ops.nms(_boxes(4).to(mojo_gpu), torch.ones(4), 0.5)
        else:
            _roi_op(kind, torch.ones(3, 2, 37, 53, device=mojo_gpu), _rois())


@pytest.mark.parametrize("kind", ["align", "pool", "nms"])
def test_malformed_shape(mojo_gpu: str, kind: str):
    with pytest.raises(NotImplementedError, match="NCHW|expected boxes|shape"):
        if kind == "nms":
            vision.ops.nms(
                torch.ones(4, 5, device=mojo_gpu), torch.ones(4, device=mojo_gpu), 0.5
            )
        else:
            _roi_op(kind, torch.ones(3, 37, 53, device=mojo_gpu), _rois().to(mojo_gpu))


@pytest.mark.parametrize("operand", ["boxes", "scores"])
def test_nms_noncontiguous(mojo_gpu: str, operand: str):
    boxes = _boxes(24).to(mojo_gpu)
    scores = torch.linspace(0.01, 1, 24).to(mojo_gpu)
    if operand == "boxes":
        boxes = boxes[::2]
        scores = scores[:12]
    else:
        boxes = boxes[:12]
        scores = scores[::2]
    with pytest.raises(NotImplementedError, match="contigu"):
        vision.ops.nms(boxes, scores, 0.5)


def test_second_gpu(mojo_gpu: str):
    if getattr(torch, "mojo").device_count() < 3:
        pytest.skip("requires two GPUs (the final mojo device is CPU)")
    with open("/tmp/gpu_lock_1.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            test_nms("mojo:1", torch.float32, 47, 0.5)
            _check_roi("mojo:1", "align", torch.float32, 1.0, 2, True)
            _check_roi("mojo:1", "pool", torch.float32, 1.0, 2, False)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_roi_pool_autocast_forward_backward(mojo_gpu: str, dtype: torch.dtype):
    data = torch.randn(2, 3, 9, 11, generator=torch.Generator().manual_seed(91)).to(
        dtype
    )
    rois = torch.tensor([[0, -2, -1, 7, 6], [1, 2, 1, 10, 8]], dtype=dtype)
    reference = data.float().detach().requires_grad_()
    ours = data.to(mojo_gpu).detach().requires_grad_()
    device_rois = rois.to(mojo_gpu)
    want = vision.ops.roi_pool(reference, rois.float(), (3, 4))
    _, expected_argmax = torch.ops.torchvision.roi_pool(
        reference.detach(), rois.float(), 1.0, 3, 4
    )
    with torch.autocast("mojo", dtype=torch.float16):
        got = vision.ops.roi_pool(ours, device_rois, (3, 4))
        tuple_output, argmax = torch.ops.torchvision.roi_pool(
            ours, device_rois, 1.0, 3, 4
        )
    assert got.dtype == dtype
    assert argmax.dtype == dtype
    assert not argmax.requires_grad
    _assert_close(got, want, dtype)
    _assert_close(tuple_output, want, dtype)
    torch.testing.assert_close(argmax.cpu(), expected_argmax.to(dtype))
    grad = torch.randn(got.shape, generator=torch.Generator().manual_seed(92)).to(dtype)
    want.backward(grad.float())
    got.backward(grad.to(mojo_gpu))
    assert ours.grad is not None and reference.grad is not None
    _assert_close(ours.grad, reference.grad, dtype)


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_nms_identical_boxes(
    mojo_gpu: str, threshold: float, tied: bool, dtype: torch.dtype
):
    _dtype_supported(mojo_gpu, dtype)
    boxes = torch.tensor([[0.125, 0.3125, 13.234, 31.445]], dtype=dtype).repeat(9, 1)
    scores = (
        torch.ones(9, dtype=dtype) if tied else torch.linspace(0.1, 0.9, 9).to(dtype)
    )
    reference_dtype = torch.float32 if dtype == torch.float16 else dtype
    want = vision.ops.nms(
        boxes.to(reference_dtype), scores.to(reference_dtype), threshold
    )
    got = vision.ops.nms(boxes.to(mojo_gpu), scores.to(mojo_gpu), threshold)
    assert want.numel() == (9 if threshold == 1.0 else 1)
    torch.testing.assert_close(got.cpu(), want)


@pytest.mark.parametrize("sampling", [-1, 0, 2])
@pytest.mark.parametrize("aligned", [False, True])
def test_roi_align_zero_spatial_scale(mojo_gpu: str, sampling: int, aligned: bool):
    data = torch.randn(3, 2, 5, 7, generator=torch.Generator().manual_seed(93))
    reference = data.detach().requires_grad_()
    ours = data.to(mojo_gpu).requires_grad_()
    rois = _rois()
    want = vision.ops.roi_align(reference, rois, (3, 2), 0.0, sampling, aligned)
    got = vision.ops.roi_align(ours, rois.to(mojo_gpu), (3, 2), 0.0, sampling, aligned)
    _assert_close(got, want, torch.float32)
    grad = torch.randn(want.shape, generator=torch.Generator().manual_seed(94))
    want.backward(grad)
    got.backward(grad.to(mojo_gpu))
    assert ours.grad is not None and reference.grad is not None
    _assert_close(ours.grad, reference.grad, torch.float32)


def test_multiscale_roi_align_backward(mojo_gpu: str):
    pool = vision.ops.MultiScaleRoIAlign(["0", "1"], output_size=3, sampling_ratio=2)
    features = {
        "0": torch.randn(2, 3, 32, 40, requires_grad=True),
        "1": torch.randn(2, 3, 16, 20, requires_grad=True),
    }
    proposals = [_boxes(13) * 2, _boxes(11) * 2]
    for boxes in proposals:
        boxes[0] = torch.tensor([0, 0, 150, 120])
    shapes = [(128, 160), (128, 160)]
    ours = {
        name: tensor.detach().to(mojo_gpu).requires_grad_()
        for name, tensor in features.items()
    }
    expected = pool(features, proposals, shapes)
    result = pool(ours, [boxes.to(mojo_gpu) for boxes in proposals], shapes)
    torch.testing.assert_close(result.cpu(), expected, atol=3e-5, rtol=3e-5)
    grad = torch.randn(expected.shape, generator=torch.Generator().manual_seed(95))
    expected.backward(grad)
    result.backward(grad.to(mojo_gpu))
    for name, reference in features.items():
        actual_grad = ours[name].grad
        assert reference.grad is not None and actual_grad is not None
        torch.testing.assert_close(
            actual_grad.cpu(), reference.grad, atol=3e-5, rtol=3e-5
        )


@pytest.mark.parametrize("kind", ["align", "pool"])
def test_roi_backward_grid_stride(mojo_gpu: str, kind: str):
    generator = torch.Generator().manual_seed(96)
    data = torch.randn(2, 5, 241, 317, generator=generator)
    starts = torch.rand(17, 2, generator=generator) * torch.tensor([220, 170])
    sizes = torch.rand(17, 2, generator=generator) * torch.tensor([80, 60]) + 1
    batches = (torch.arange(17) % 2).float().unsqueeze(1)
    rois = torch.cat((batches, starts, starts + sizes), dim=1)
    reference = data.detach().requires_grad_()
    ours = data.to(mojo_gpu).detach().requires_grad_()
    expected = _roi_op(kind, reference, rois, aligned=True)
    result = _roi_op(kind, ours, rois.to(mojo_gpu), aligned=True)
    _assert_close(result, expected, torch.float32)
    grad = torch.randn(expected.shape, generator=torch.Generator().manual_seed(97))
    expected.backward(grad)
    result.backward(grad.to(mojo_gpu))
    assert reference.grad is not None and ours.grad is not None
    _assert_close(ours.grad, reference.grad, torch.float32)
