"""Render quality regression: no holes, no grain, smooth edges (P3 acceptance).

Quantifies the properties the design doc promises for the foreground render:

- **solid surface** — almost no alpha holes inside the silhouette
  (binary-closed, filled region minus the mask);
- **no grain** — low local luminance variance on the opaque interior;
- **smooth silhouette** — alpha transition across the boundary is gradual
  (supersampled), not a single-pixel hard step on every edge pixel;
- **backend agreement** — when gsplat/CUDA is available, the GPU backend
  must agree with the CPU backend on mask and depth.
"""
from __future__ import annotations

import numpy as np
import pytest

from realinject.gauss.model import GaussianCloud, identity_quat
from realinject.gauss.raster_cpu import render_gaussians

W, H = 800, 600
FX = FY = 900.0
CAM = {
    "name": "CAM_Q",
    "K": [[FX, 0, W / 2], [0, FY, H / 2], [0, 0, 1]],
    "T_cam2global": [
        [1, 0, 0, 0],
        [0, 0, 1, -12],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    "width": W,
    "height": H,
}


def _sphere_cloud(n=60000, radius=2.0, sigma=0.02, seed_colored=True):
    """Dense Fibonacci sphere; optionally per-point color noise (grain source)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    xyz = np.stack([radius * np.cos(theta) * np.sin(phi),
                    radius * np.sin(theta) * np.sin(phi),
                    radius * np.cos(phi)], axis=1).astype(np.float32)
    if seed_colored:
        rng = np.random.default_rng(7)
        # ±25/255 reflectance jitter: grain the smoothing has to suppress
        rgb = np.clip(0.8 + rng.uniform(-0.1, 0.1, (n, 3)), 0, 1).astype(np.float32)
    else:
        rgb = np.ones((n, 3), np.float32)
    return GaussianCloud(
        xyz=xyz,
        rot=identity_quat(n),
        scale=np.full((n, 3), sigma, np.float32),
        opacity=np.full(n, 0.95, np.float32),
        rgb=rgb,
    )


def _interior_hole_rate(alpha: np.ndarray, mask: np.ndarray) -> float:
    from scipy import ndimage

    closed = ndimage.binary_closing(mask, structure=np.ones((5, 5), bool))
    filled = ndimage.binary_fill_holes(closed)
    holes = filled & ~mask
    return float(holes.sum() / max(int(mask.sum()), 1))


def _surface_noise(rgb: np.ndarray, alpha: np.ndarray) -> float:
    from scipy import ndimage

    gray = rgb.astype(np.float32).mean(axis=2)
    mu = ndimage.uniform_filter(gray, 7)
    mu2 = ndimage.uniform_filter(gray * gray, 7)
    local_std = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    interior = ndimage.binary_erosion(alpha > 0.8, np.ones((5, 5), bool))
    if not interior.any():
        return float("nan")
    return float(local_std[interior].mean())


def test_surface_is_solid_without_holes():
    out = render_gaussians(CAM, _sphere_cloud(seed_colored=False))
    assert out.mask.sum() > 20000
    holes = _interior_hole_rate(out.alpha, out.mask)
    assert holes < 0.01, f"interior hole rate {holes:.4f} >= 1%"


def test_interior_alpha_is_saturated():
    # supersampled coverage: the dense interior must read near-opaque
    out = render_gaussians(CAM, _sphere_cloud(seed_colored=False))
    interior = out.alpha > 0.9
    assert interior.sum() > 15000, f"only {interior.sum()} near-opaque px"
    # unpremultiplied: white asset stays white where opaque
    assert np.allclose(out.rgb[interior], 255, atol=2)


def test_surface_has_no_grain():
    out = render_gaussians(CAM, _sphere_cloud(seed_colored=True))
    noise = _surface_noise(out.rgb, out.alpha)
    # raw per-point +/-25/255 jitter would read ~13+ local std; smoothing
    # plus supersampled attribution must cut that down substantially
    assert noise < 6.0, f"surface grain {noise:.2f}/255 too high"


def test_silhouette_edge_is_antialiased():
    out = render_gaussians(CAM, _sphere_cloud(seed_colored=False))
    from scipy import ndimage

    m = out.mask
    # 2-ring boundary: the graded band a supersampled edge produces
    b2 = m & ~ndimage.binary_erosion(m, np.ones((3, 3), bool), iterations=2)
    a = out.alpha
    graded = ((a > 0.05) & (a < 0.95))[b2].mean()
    assert graded > 0.6, f"only {graded:.2f} of the edge band is graded"
    # and no hard step: a fully-opaque pixel directly inside the mask edge
    # would mean single-sample (aliased) coverage
    b1 = m & ~ndimage.binary_erosion(m, np.ones((3, 3), bool))
    hard = (a[b1] > 0.95).mean()
    assert hard < 0.05, f"{hard:.2f} of boundary pixels are hard-jump"


def test_gpu_backend_matches_cpu():
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pytest.importorskip("gsplat")

    from realinject.gauss.render import get_renderer

    cloud = _sphere_cloud(n=60000, seed_colored=False)
    cfg = {"gaussian": {"render": {"backend": "gsplat"}}}
    gpu_out = get_renderer(cfg)(CAM, cloud, None, cfg)
    cpu_out = render_gaussians(CAM, cloud)

    inter = (gpu_out.mask & cpu_out.mask).sum()
    union = (gpu_out.mask | cpu_out.mask).sum()
    iou = inter / max(union, 1)
    assert iou > 0.95, f"mask IoU {iou:.3f} between gsplat and cpu"

    core = (gpu_out.mask & cpu_out.mask
            & (gpu_out.alpha > 0.5) & (cpu_out.alpha > 0.5))
    ddiff = np.abs(gpu_out.depth[core] - cpu_out.depth[core])
    # gsplat blends expected depth front-to-back, the CPU path a
    # weight-normalized mean: a ~1% relative offset is the known contract gap
    rel = ddiff / np.maximum(cpu_out.depth[core], 1e-6)
    assert np.median(rel) < 0.015, f"median relative depth diff {np.median(rel):.4f}"
    # both must be solid where they agree
    assert gpu_out.alpha.max() > 0.98 and cpu_out.alpha.max() > 0.98
