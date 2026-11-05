"""Canonical asset -> GaussianCloud conversion."""
from __future__ import annotations

import numpy as np
import trimesh

from realinject.assets import convert


def _box_canon():
    box = trimesh.creation.box(extents=(2.0, 1.0, 0.6))
    verts = np.asarray(box.vertices, dtype=np.float32)
    faces = np.asarray(box.faces, dtype=np.int32)
    colors = np.tile(np.array([[0.8, 0.2, 0.1]], np.float32), (len(verts), 1))
    return {"verts": verts, "faces": faces, "colors": colors}


def test_mesh_to_gaussians_shapes_and_ranges():
    canon = _box_canon()
    cfg = {"gaussian_init": {"mesh_samples": 500}}
    cloud = convert.mesh_to_gaussians(canon, cfg)
    assert len(cloud) == 500
    assert cloud.xyz.shape == (500, 3)
    assert cloud.scale.shape == (500, 3)
    assert np.all(cloud.opacity > 0) and np.all(cloud.opacity <= 1)
    assert cloud.rgb.min() >= 0 and cloud.rgb.max() <= 1
    # samples stay on the box surface
    assert np.isclose(cloud.xyz.min(axis=0), [-1.0, -0.5, -0.3], atol=1e-5).all()
    assert np.isclose(cloud.xyz.max(axis=0), [1.0, 0.5, 0.3], atol=1e-5).all()


def test_points_to_gaussians_intensity_colors():
    xyz = np.random.default_rng(0).uniform(0, 1, (200, 3)).astype(np.float32)
    canon = {"xyz": xyz, "intensity": np.full(200, 255, np.uint8)}
    cloud = convert.points_to_gaussians(canon)
    assert len(cloud) == 200
    assert np.allclose(cloud.rgb, 1.0)
    assert np.all(cloud.scale > 0)


def test_points_to_gaussians_density_mode():
    xyz = np.random.default_rng(0).uniform(0, 1, (200, 3)).astype(np.float32)
    canon = {"xyz": xyz, "intensity": np.zeros(200, np.uint8)}
    cfg = {"gaussian_init": {"color_mode": "density"}}
    cloud = convert.points_to_gaussians(canon, cfg)
    assert cloud.rgb.min() >= 0 and cloud.rgb.max() <= 1


def test_knn_scale_monotonic_in_spacing():
    dense = np.random.default_rng(1).uniform(0, 0.1, (500, 3))
    sparse = np.random.default_rng(1).uniform(0, 1.0, (500, 3))
    s_dense = convert.knn_scale(dense, 8, 1.2, [0.001, 1.0]).mean()
    s_sparse = convert.knn_scale(sparse, 8, 1.2, [0.001, 1.0]).mean()
    assert s_sparse > s_dense


def test_scale_is_clipped():
    xyz = np.zeros((10, 3)) + np.linspace(0, 5, 10)[:, None]
    scale = convert.knn_scale(xyz, 2, 1.0, [0.003, 0.15])
    assert scale.max() <= 0.15 + 1e-6
    assert scale.min() >= 0.003 - 1e-6


def test_gaussian_cloud_roundtrip(tmp_path):
    from realinject.gauss.model import GaussianCloud

    n = 50
    cloud = GaussianCloud(
        xyz=np.zeros((n, 3), np.float32), rot=np.tile([1, 0, 0, 0], (n, 1)),
        scale=np.full((n, 3), 0.01, np.float32), opacity=np.full(n, 0.9, np.float32),
        rgb=np.full((n, 3), 0.5, np.float32), extra={"intensity": np.zeros(n, np.uint8)})
    path = tmp_path / "c.npz"
    cloud.save(path)
    loaded = GaussianCloud.load(path)
    assert len(loaded) == n
    assert np.allclose(loaded.scale, 0.01)
    assert "intensity" in loaded.extra
