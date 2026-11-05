"""Parity with the vendored stag3d point renderer (P2 acceptance).

The Gaussian renderer must agree with ``stag3d.renderer.render_rgbd`` on a
synthetic scene once the Gaussians degenerate to tiny opaque splats: same lit
pixels, same colors, same depth. Projection math must match exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from realinject.gauss.model import GaussianCloud, identity_quat
from realinject.gauss.raster_cpu import render_gaussians

W, H = 640, 480
FX = FY = 1000.0
EYE = np.array([0.0, -12.0, 0.0])
TARGET = np.array([0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def stag3d():
    """Vendored stag3d from the sibling stag1_inject repo (read-only)."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    src = root.parent / "stag1_inject" / "src"
    if not (src / "stag3d").is_dir():
        pytest.skip("stag3d (stag1_inject) not available")
    sys.path.insert(0, str(src))
    try:
        from stag3d import geometry, renderer  # noqa: PLC0415
    finally:
        sys.path.pop(0)
    return geometry, renderer


def _lookat_cam():
    """OpenCV camera at EYE looking at TARGET, up +Z."""
    f = TARGET - EYE
    f = f / np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x = x / np.linalg.norm(x)
    y = np.cross(f, x)  # points down in OpenCV
    rot = np.column_stack([x, y, f])
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = EYE
    cam = {
        "name": "CAM_PARITY",
        "K": [[FX, 0, W / 2], [0, FY, H / 2], [0, 0, 1]],
        "T_cam2global": t.tolist(),
        "width": W,
        "height": H,
    }
    return cam, t


def _grid_points(spacing=0.4, half=2.0):
    xs = np.arange(-half, half + 1e-9, spacing)
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)


def _front_cap(radius=2.0, n=2000):
    """Front hemisphere (facing the camera at -Y): one depth per pixel ray.

    A full 3D grid stacks many points along the same pixel ray, where the
    1px point renderer (painter) and the Gaussian soft-z attribution
    legitimately disagree; the convex front cap removes that ambiguity so the
    parity check stays tight.
    """
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 1.0 * i / n)  # 0..pi/2 -> front cap only
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([radius * np.cos(theta) * np.sin(phi),
                     -radius * np.cos(phi),  # y < 0 faces the camera
                     radius * np.sin(theta) * np.sin(phi)], axis=1).astype(np.float32)


def test_projection_matches_stag3d(stag3d):
    geometry, _ = stag3d
    cam, t = _lookat_cam()
    pts = _grid_points()
    k = np.asarray(cam["K"], dtype=float)
    intr = geometry.CameraIntrinsics.from_matrix(k, W, H)
    # stag3d: cam frame = inverse of cam2world applied to world points
    pts_cam = geometry.transform_points(geometry.invert_pose(t), pts)
    uv, depth, valid = geometry.project_points(pts_cam, intr)

    # our renderer's projection, recomputed the same way
    t_g2c = np.linalg.inv(t)
    mu_c = pts.astype(float) @ t_g2c[:3, :3].T + t_g2c[:3, 3]
    u = FX * mu_c[:, 0] / mu_c[:, 2] + W / 2
    v = FY * mu_c[:, 1] / mu_c[:, 2] + H / 2
    m = valid & (mu_c[:, 2] > 1e-3)
    assert np.allclose(u[m], uv[m, 0], atol=1e-9)
    assert np.allclose(v[m], uv[m, 1], atol=1e-9)
    assert np.allclose(mu_c[m, 2], depth[m], atol=1e-9)


def test_render_matches_stag3d_point_render(stag3d):
    geometry, renderer = stag3d
    cam, t = _lookat_cam()
    pts = _front_cap()
    cols = (np.random.default_rng(0).uniform(0.2, 1.0, (len(pts), 3)) * 255).astype(np.uint8)

    # degenerate Gaussians: tiny isotropic splats, near-opaque
    cloud = GaussianCloud(
        xyz=pts,
        rot=identity_quat(len(pts)),
        scale=np.full((len(pts), 3), 0.005, np.float32),
        opacity=np.full(len(pts), 0.999, np.float32),
        rgb=cols.astype(np.float32) / 255.0,
    )
    out = render_gaussians(cam, cloud)

    intr = geometry.CameraIntrinsics.from_matrix(np.asarray(cam["K"], float), W, H)
    ref_img, ref_depth = renderer.render_rgbd(
        pts.astype(np.float64), cols, t, intr, point_size=1, background=0)

    ref_mask = ref_depth > 0
    from scipy import ndimage

    # The point renderer paints one pixel per projected point; a degenerate
    # Gaussian covers that core plus a <=2 px halo (3-sigma footprint). Parity
    # therefore means: every ref core is lit by us, and our mask never extends
    # beyond a 2 px dilation of the ref cores.
    # square structuring element: chebyshev radius-2 halo (default is a cross)
    dil2 = ndimage.binary_dilation(ref_mask, structure=np.ones((3, 3), bool),
                                   iterations=2)
    recall = (out.mask & ref_mask).sum() / max(ref_mask.sum(), 1)
    precision = (out.mask & dil2).sum() / max(out.mask.sum(), 1)
    assert recall > 0.95, f"core recall {recall:.3f}"
    assert precision > 0.90, f"halo precision {precision:.3f}"

    # 2/3) colors and depth agree on the *core* pixels, where a single splat
    # dominates (low-alpha footprint edges legitimately blend neighbours, which
    # the 1px point renderer cannot represent).
    both = out.mask & ref_mask
    core = both & (out.alpha >= 0.5)
    assert core.sum() > 500, f"only {core.sum()} core pixels"
    diff = np.abs(out.rgb[core].astype(int) - ref_img[core].astype(int)).mean()
    assert diff < 12.0, f"mean color diff {diff:.1f}/255"
    ddiff = np.abs(out.depth[core] - ref_depth[core])
    assert np.median(ddiff) < 0.01, f"median depth diff {np.median(ddiff):.3f} m"
    assert np.percentile(ddiff, 95) < 0.05
