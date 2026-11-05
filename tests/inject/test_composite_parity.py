"""Composite parity: composite_gauss vs upstream composite_onto."""
from __future__ import annotations

import numpy as np
import pytest

from realinject.inject.bridge import bootstrap
from realinject.inject.composite_ext import composite_gauss


@pytest.fixture(scope="module")
def cfg():
    return {
        "composite": {"alpha_threshold": 0.05, "depth_margin_m": 0.2,
                      "depth_margin_rel": 0.02},
        "scene_occluders": {"enabled": False},
        "dataset": {"root": ".", "version": "v1.0-mini"},
        "inject_nusc": {"root": str(_inject_root())},
    }


def _inject_root():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    return root.parent / "nusc-inject"


def _scene():
    """Synthetic scene: object plane at z=10, wall at z=20, 640x480 camera."""
    w, h = 640, 480
    k = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    cam = {"name": "CAM", "K": k.tolist(), "T_cam2global": np.eye(4).tolist(),
           "width": w, "height": h}
    # object: fronto-parallel plane at z=10 covering the central rectangle
    # (metric coords chosen so the projections land inside the rectangle)
    us = np.arange(220, 420, 2.0)
    vs = np.arange(140, 340, 2.0)
    gx, gy = np.meshgrid(us, vs)
    obj_pts = np.stack([((gx - 320.0) * 10.0 / 500.0).ravel(),
                        ((gy - 240.0) * 10.0 / 500.0).ravel(),
                        np.full(gx.size, 10.0)], axis=1)
    # scene wall at z=20 (behind) — sparse, like a LiDAR return pattern
    us2 = np.arange(0, w, 12.0)
    vs2 = np.arange(0, h, 12.0)
    sgx, sgy = np.meshgrid(us2, vs2)
    scene = np.stack([((sgx - 320.0) * 20.0 / 500.0).ravel(),
                      ((sgy - 240.0) * 20.0 / 500.0).ravel(),
                      np.full(sgx.size, 20.0)], axis=1)
    bg = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    fg = np.zeros((h, w, 4), dtype=np.uint8)
    fg[140:340, 220:420, :3] = 200
    fg[140:340, 220:420, 3] = 255
    obj_depth = np.full((h, w), np.inf, np.float32)
    obj_depth[140:340, 220:420] = 10.0
    return cam, obj_pts, scene, bg, fg, obj_depth


def test_blend_matches_upstream(cfg):
    bootstrap(cfg)
    from nusc_inject import composite as up

    cam, obj_pts, scene, bg, fg, obj_depth = _scene()
    objects = [{"pts_raw": obj_pts, "gt": {"T_model2global": np.eye(4).tolist()}}]
    ref = up.composite_onto("CAM", fg, bg, objects, cam, cfg, scene,
                            T_lidar2global=np.eye(4))
    ours = composite_gauss("CAM", fg, bg, obj_depth, cam, cfg, scene,
                           T_lidar2global=np.eye(4))
    ref_rgb, ref_vis, ref_alpha, ref_od, ref_sd, _ = ref
    our_rgb, our_vis, our_alpha, our_od, our_sd, _ = ours
    assert np.array_equal(ref_vis, our_vis)
    assert np.allclose(ref_alpha, our_alpha, atol=1e-5)
    assert np.allclose(ref_od, our_od, atol=1e-4, equal_nan=True)
    assert np.allclose(ref_sd, our_sd, atol=1e-4, equal_nan=True)
    assert np.abs(ref_rgb.astype(int) - our_rgb.astype(int)).max() <= 1


def test_occlusion_by_nearer_wall(cfg):
    bootstrap(cfg)
    cam, obj_pts, scene, bg, fg, obj_depth = _scene()
    # move the wall in front of the object (z=5): object must be hidden
    scene_front = scene.copy()
    scene_front[:, 2] = 5.0
    out_rgb, visible, alpha, _, scene_depth, info = composite_gauss(
        "CAM", fg, bg, obj_depth, cam, cfg, scene_front,
        T_lidar2global=np.eye(4))
    assert info["n_alpha_px"] > 0
    assert info["n_occluded_px"] > 0
    # hidden pixels keep the background
    hidden = (alpha > 0.05) & ~visible
    assert np.array_equal(out_rgb[hidden], bg[hidden])
    assert np.all(scene_depth[hidden] <= 5.5)
