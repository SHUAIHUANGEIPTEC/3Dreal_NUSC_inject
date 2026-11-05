"""LiDAR payload adapter and renderer dispatch."""
from __future__ import annotations

import numpy as np
import pytest

from realinject.gauss.lidar_splat import (
    cloud_to_lidar_payload,
    estimate_normals,
    sample_indices,
)
from realinject.gauss.model import GaussianCloud, identity_quat
from realinject.gauss.raster_cpu import render_gaussians
from realinject.gauss.render import get_renderer, render, save_rgba_png


def _cloud(n=500):
    rng = np.random.default_rng(0)
    return GaussianCloud(
        xyz=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        rot=identity_quat(n),
        scale=np.full((n, 3), 0.02, np.float32),
        opacity=np.full(n, 0.9, np.float32),
        rgb=np.full((n, 3), 0.5, np.float32),
    )


def test_sample_indices_deterministic_and_capped():
    a = sample_indices(1000, 100, seed=0)
    b = sample_indices(1000, 100, seed=0)
    assert np.array_equal(a, b) and len(a) == 100
    assert np.array_equal(sample_indices(50, 100), np.arange(50))


def test_estimate_normals_unit_and_outward():
    # a sphere shell: normals must point away from the centroid
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(400, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    normals = estimate_normals(pts, k=12)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
    outward = np.sum(pts * normals, axis=1)  # centroid ~ 0
    assert np.mean(outward > 0) > 0.95


def test_payload_uses_mesh_normals_when_present():
    cloud = _cloud()
    cloud.extra["normal"] = np.tile([0.0, 0.0, 1.0], (len(cloud), 1)).astype(np.float32)
    payload = cloud_to_lidar_payload(cloud, gt={"x": 1})
    assert payload["pts_raw"].shape == (len(cloud), 3)
    assert np.allclose(payload["normals_raw"], [0, 0, 1])
    assert payload["gt"] == {"x": 1}


def test_payload_samples_down_to_n_dense():
    cloud = _cloud(5000)
    payload = cloud_to_lidar_payload(cloud, cfg={"gaussian": {"lidar": {"n_dense": 250}}})
    assert len(payload["pts_raw"]) == 250
    assert np.allclose(np.linalg.norm(payload["normals_raw"], axis=1), 1.0, atol=1e-6)


def test_get_renderer_dispatch():
    assert get_renderer(None) is render_gaussians
    assert get_renderer({"gaussian": {"render": {"backend": "cpu"}}}) is render_gaussians
    with pytest.raises(ValueError):
        get_renderer({"gaussian": {"render": {"backend": "vulkan"}}})


def test_gsplat_backend_raises_actionable_error():
    from realinject.gauss.raster_gpu import render_gaussians_gsplat

    if not _gsplat_available():
        with pytest.raises(RuntimeError, match="gsplat"):
            render_gaussians_gsplat({}, _cloud(10))


def _gsplat_available() -> bool:
    try:
        import gsplat  # noqa: F401

        return True
    except ImportError:
        return False


def test_save_rgba_png_roundtrip(tmp_path):
    cam = {
        "name": "CAM", "K": [[500, 0, 32], [0, 500, 24], [0, 0, 1]],
        "T_cam2global": np.eye(4).tolist(), "width": 64, "height": 48,
    }
    cloud = _cloud(200)
    out = render(cam, cloud, T_model2global=np.eye(4))
    path = tmp_path / "fg.png"
    save_rgba_png(path, out)
    import cv2

    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert rgba.shape == (48, 64, 4)
    assert np.array_equal(rgba[..., 3], (np.clip(out.alpha, 0, 1) * 255 + 0.5).astype(np.uint8))
    # straight alpha: rgb channel matches the unpremultiplied foreground
    bgr = rgba[..., :3][out.mask]
    assert np.abs(bgr.astype(int) - out.rgb[out.mask][:, ::-1].astype(int)).max() <= 1
