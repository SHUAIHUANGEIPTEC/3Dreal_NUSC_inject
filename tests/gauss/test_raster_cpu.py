"""CPU rasterizer: analytic sphere, conventions, culling."""
from __future__ import annotations

import numpy as np

from realinject.gauss.model import GaussianCloud, identity_quat
from realinject.gauss.raster_cpu import quat_to_rotmat, render_gaussians

W, H = 640, 480
FX = FY = 1000.0
CAM = {
    "name": "CAM_FRONT",
    "K": [[FX, 0, W / 2], [0, FY, H / 2], [0, 0, 1]],
    # camera at (0,-12,0) looking along +Y, up +Z (OpenCV: x right, y down)
    "T_cam2global": [
        [1, 0, 0, 0],
        [0, 0, 1, -12],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    "width": W,
    "height": H,
}


def _sphere_cloud(radius=2.0, n=60000, sigma=0.02, opacity=0.95):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    xyz = np.stack([radius * np.cos(theta) * np.sin(phi),
                    radius * np.sin(theta) * np.sin(phi),
                    radius * np.cos(phi)], axis=1).astype(np.float32)
    return GaussianCloud(
        xyz=xyz,
        rot=identity_quat(n),
        scale=np.full((n, 3), sigma, np.float32),
        opacity=np.full(n, opacity, np.float32),
        rgb=np.ones((n, 3), np.float32),
    )


def test_quat_to_rotmat_identity_and_axis():
    q = np.array([[1, 0, 0, 0], [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]])
    r = quat_to_rotmat(q)
    assert np.allclose(r[0], np.eye(3), atol=1e-9)
    # 90 deg about z maps +X to +Y
    assert np.allclose(r[1] @ [1, 0, 0], [0, 1, 0], atol=1e-9)


def test_sphere_silhouette_matches_analytic_area():
    # 60k samples: front-hemisphere coverage saturates alpha inside the disk,
    # so alpha.sum() tracks the projected 3-sigma silhouette area
    cloud = _sphere_cloud()
    out = render_gaussians(CAM, cloud)
    # projected 3-sigma silhouette: pi * (f * (R + 3*sigma) / D)^2
    expected = np.pi * (FX * (2.0 + 3 * 0.02) / 12.0) ** 2
    assert abs(out.alpha.sum() - expected) / expected < 0.15
    assert out.info["n_px"] > 0.5 * expected  # not a sparse sprinkle


def test_sphere_depth_monotone_outward():
    cloud = _sphere_cloud()
    out = render_gaussians(CAM, cloud)
    cy, cx = H // 2, W // 2
    assert np.isfinite(out.depth[cy, cx])
    # near side of the sphere: z = D - R = 10 m
    assert abs(float(out.depth[cy, cx]) - 10.0) < 0.3
    # walking out from the centre the surface recedes
    prev = out.depth[cy, cx]
    for r in (40, 80, 120, 150):
        ring = [out.depth[cy, cx + r], out.depth[cy, cx - r],
                out.depth[cy + r, cx], out.depth[cy - r, cx]]
        finite = [d for d in ring if np.isfinite(d)]
        assert finite, f"no depth at radius {r}"
        now = float(np.mean(finite))
        assert now > prev - 1e-3, f"depth not monotone: {prev} -> {now} at r={r}"
        prev = now


def test_empty_and_offframe_conventions():
    cloud = _sphere_cloud(n=500)
    # move the sphere far off-frame
    off = GaussianCloud(
        xyz=cloud.xyz + np.array([0, 0, 500], np.float32),
        rot=cloud.rot, scale=cloud.scale, opacity=cloud.opacity, rgb=cloud.rgb)
    out = render_gaussians(CAM, off)
    assert out.info["n_kept"] == 0
    assert out.rgb.sum() == 0
    assert np.all(np.isinf(out.depth))
    assert not out.mask.any()


def test_alpha_in_unit_range_and_rgb_shape():
    out = render_gaussians(CAM, _sphere_cloud(n=3000))
    assert out.rgb.shape == (H, W, 3) and out.rgb.dtype == np.uint8
    assert out.alpha.dtype == np.float32
    assert out.alpha.min() >= 0.0 and out.alpha.max() <= 1.0
    # unpremultiplied foreground: white spheres stay white where opaque
    interior = out.alpha > 0.9
    assert interior.sum() > 1000
    assert np.allclose(out.rgb[interior], 255, atol=2)


def test_depth_is_inf_outside_mask():
    out = render_gaussians(CAM, _sphere_cloud(n=3000))
    assert np.all(np.isinf(out.depth[~out.mask]))
    assert np.all(np.isfinite(out.depth[out.mask]))
