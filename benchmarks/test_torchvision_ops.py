"""Detection ROI sampling, pooling and IoU suppression device times."""

from __future__ import annotations

from types import ModuleType

import pytest
import torch
from bench_lib.cases import DTYPES, both
from bench_lib.check import Bench
from bench_lib.hw import Hardware
from bench_lib.measure import gpu_lock

# N, C, H, W, K, pooled H, pooled W, sampling ratio.
ALIGN_SHAPES = {
    "N2C256H200W272K1000_o7_s2": (2, 256, 200, 272, 1000, 7, 7, 2),
    "N2C256H200W272K1000_o14_s2": (2, 256, 200, 272, 1000, 14, 14, 2),
    "N2C256H200W272K1000_o7_adaptive": (2, 256, 200, 272, 1000, 7, 7, -1),
    "N2C256H200W272K1000_o14_adaptive": (2, 256, 200, 272, 1000, 14, 14, -1),
    "N3C7H37W53K19_o7x5_s2": (3, 7, 37, 53, 19, 7, 5, 2),
}
POOL_SHAPES = {name: shape for name, shape in ALIGN_SHAPES.items() if shape[-1] == 2}
NMS_SHAPES = {"K1000": 1000, "K5000": 5000, "K20000": 20000}

COVERS = {
    "torchvision::nms": "test_nms",
    "torchvision::roi_align": "test_roi_align",
    "torchvision::_roi_align_backward": "test_roi_align_backward",
    "torchvision::roi_pool": "test_roi_pool",
    "torchvision::_roi_pool_backward": "test_roi_pool_backward",
}
SKIPPED = {}


@pytest.fixture
def vision() -> ModuleType:
    return pytest.importorskip("torchvision")


def _roi_inputs(
    shape: tuple[int, int, int, int, int, int, int, int], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    n, c, h, w, k, _, _, _ = shape
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(n, c, h, w, dtype=dtype, generator=generator)
    starts = torch.rand(k, 2, generator=generator) * torch.tensor([w * 0.7, h * 0.7])
    sizes = (torch.rand(k, 2, generator=generator) * 0.2 + 0.05) * torch.tensor([w, h])
    batches = (torch.arange(k) % n).float().unsqueeze(1)
    return x, torch.cat((batches, starts, starts + sizes), dim=1).to(dtype)


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", ALIGN_SHAPES)
@pytest.mark.bench_op("torchvision::roi_align")
def test_roi_align(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    shape = ALIGN_SHAPES[shape_id]
    _, c, _, _, k, ph, pw, sampling = shape
    x, rois = _roi_inputs(shape, DTYPES[dtype_id])
    with gpu_lock():
        x_ref, x_our = both(x, hw, mojo_device)
        r_ref, r_our = both(rois, hw, mojo_device)
    bench.run(
        lambda: vision.ops.roi_align(x_ref, r_ref, (ph, pw), 1.0, sampling, True),
        lambda: vision.ops.roi_align(x_our, r_our, (ph, pw), 1.0, sampling, True),
        flops=float(k * c * ph * pw * 32),
    )


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", ALIGN_SHAPES)
@pytest.mark.bench_op("torchvision::_roi_align_backward")
def test_roi_align_backward(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    shape = ALIGN_SHAPES[shape_id]
    n, c, h, w, k, ph, pw, sampling = shape
    _, rois = _roi_inputs(shape, DTYPES[dtype_id])
    grad = torch.randn(
        k, c, ph, pw, dtype=DTYPES[dtype_id], generator=torch.Generator().manual_seed(1)
    )
    with gpu_lock():
        g_ref, g_our = both(grad, hw, mojo_device)
        r_ref, r_our = both(rois, hw, mojo_device)
    bench.run(
        lambda: torch.ops.torchvision._roi_align_backward(
            g_ref, r_ref, 1.0, ph, pw, n, c, h, w, sampling, True
        ),
        lambda: torch.ops.torchvision._roi_align_backward(
            g_our, r_our, 1.0, ph, pw, n, c, h, w, sampling, True
        ),
        flops=float(k * c * ph * pw * 32),
    )


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", POOL_SHAPES)
@pytest.mark.bench_op("torchvision::roi_pool")
def test_roi_pool(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    shape = POOL_SHAPES[shape_id]
    _, c, _, _, k, ph, pw, _ = shape
    x, rois = _roi_inputs(shape, DTYPES[dtype_id])
    with gpu_lock():
        x_ref, x_our = both(x, hw, mojo_device)
        r_ref, r_our = both(rois, hw, mojo_device)
    bench.run(
        lambda: vision.ops.roi_pool(x_ref, r_ref, (ph, pw)),
        lambda: vision.ops.roi_pool(x_our, r_our, (ph, pw)),
        flops=float(k * c * ph * pw * 32),
    )


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", POOL_SHAPES)
@pytest.mark.bench_op("torchvision::_roi_pool_backward")
def test_roi_pool_backward(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    shape = POOL_SHAPES[shape_id]
    n, c, h, w, k, ph, pw, _ = shape
    x, rois = _roi_inputs(shape, DTYPES[dtype_id])
    grad = torch.randn(
        k, c, ph, pw, dtype=DTYPES[dtype_id], generator=torch.Generator().manual_seed(1)
    )
    with gpu_lock():
        x_ref, x_our = both(x, hw, mojo_device)
        r_ref, r_our = both(rois, hw, mojo_device)
        g_ref, g_our = both(grad, hw, mojo_device)
        _, a_ref = torch.ops.torchvision.roi_pool(x_ref, r_ref, 1.0, ph, pw)
        _, a_our = torch.ops.torchvision.roi_pool(x_our, r_our, 1.0, ph, pw)
    bench.run(
        lambda: torch.ops.torchvision._roi_pool_backward(
            g_ref, r_ref, a_ref, 1.0, ph, pw, n, c, h, w
        ),
        lambda: torch.ops.torchvision._roi_pool_backward(
            g_our, r_our, a_our, 1.0, ph, pw, n, c, h, w
        ),
        flops=float(k * c * ph * pw * 4),
    )


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", NMS_SHAPES)
@pytest.mark.bench_op("torchvision::nms")
def test_nms(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    k = NMS_SHAPES[shape_id]
    generator = torch.Generator().manual_seed(0)
    starts = torch.rand(k, 2, generator=generator) * 800
    sizes = torch.rand(k, 2, generator=generator) * 200 + 1
    boxes = torch.cat((starts, starts + sizes), dim=1).to(DTYPES[dtype_id])
    scores = torch.rand(k, generator=generator).to(DTYPES[dtype_id])
    with gpu_lock():
        b_ref, b_our = both(boxes, hw, mojo_device)
        s_ref, s_our = both(scores, hw, mojo_device)
    bench.run(
        lambda: vision.ops.nms(b_ref, s_ref, 0.5),
        lambda: vision.ops.nms(b_our, s_our, 0.5),
        flops=float(k * k * 12),
    )
