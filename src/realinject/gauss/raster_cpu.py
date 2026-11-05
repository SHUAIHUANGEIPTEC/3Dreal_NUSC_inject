"""CPU Gaussian splat rasterizer.

EWA footprint projection (2D covariance of each anisotropic Gaussian) with an
**order-independent soft-z accumulation** instead of front-to-back transmittance
blending:

- per-pixel z-buffer minimum is taken over footprints with weight > ``w_min``
  (``zmin``);
- every footprint pair contributes ``w = opacity * exp(-0.5 d^T Σ2d^-1 d)``
  scaled by ``exp(-max(0, z - zmin) / tau_m)`` — surfaces further than ``tau_m``
  behind the nearest one are suppressed, which approximates occlusion for
  opaque assets (our assets are sampled surfaces, not volumetric media);
- color is the weight-normalized mean ``Σ w c / Σ w``, alpha is the
  independent transmittance product ``1 - exp(Σ log1p(-w))``, depth the
  weight-normalized mean z.

Exact transmittance blending is order-dependent and needs a Python-loop over
every Gaussian (seconds per frame at 120k splats); the soft-z form is fully
vectorized (bincount / lexsort over the footprint pairs) and stays
deterministic. ``tau_m`` (0.1 m) is the only approximation knob.

Output contract (matches ``nusc_inject.composite.composite_onto``):
``rgb`` is the **unpremultiplied** foreground color, ``alpha`` in [0,1],
``depth`` float32 with ``inf`` where nothing was drawn.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..gauss.model import GaussianCloud

DEFAULTS: dict = {
    "z_near": 0.1,  # meters, camera-space
    "z_far": 200.0,
    "cov2d_floor": 0.3,  # px^2 dilation of every footprint (official 3DGS)
    "radius_scale": 3.0,  # 3-sigma footprint radius
    "r_min_px": 0.5,  # skip footprints narrower than this (sub-pixel)
    "r_max_px": 32.0,  # cap huge close-up footprints (memory guard)
    "w_min": 0.05,  # pairs above this set the per-pixel zmin
    "w_acc": 1.0e-3,  # pairs above this contribute to color/alpha/depth
    "tau_m": 0.1,  # soft-z occlusion falloff, meters
    "alpha_max": 0.999,  # clamp so log1p(-w) stays finite
    "depth_eps": 1.0e-4,  # alpha below this stays inf-depth
}


@dataclass
class RenderOut:
    """Renderer output; see module docstring for the contract."""

    rgb: np.ndarray  # (H, W, 3) uint8, unpremultiplied foreground
    alpha: np.ndarray  # (H, W) float32 in [0,1]
    depth: np.ndarray  # (H, W) float32, inf where empty
    mask: np.ndarray  # (H, W) bool
    info: dict = field(default_factory=dict)


def render_cfg(cfg: dict | None) -> dict:
    out = dict(DEFAULTS)
    out.update((cfg or {}).get("gaussian", {}).get("render") or {})
    return out


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(N,4) wxyz quaternions -> (N,3,3) rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    n[n < 1e-12] = 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    r = np.empty((len(q), 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def _invert_rigid(t: np.ndarray) -> np.ndarray:
    rot, trans = t[:3, :3], t[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


def render_gaussians(cam: dict, cloud: GaussianCloud,
                     T_model2global: np.ndarray | None = None,
                     cfg: dict | None = None) -> RenderOut:
    """Rasterize one GaussianCloud into one camera.

    cam: CameraSpec dict (name/K/T_cam2global/width/height), OpenCV camera.
    T_model2global: 4x4; None means the cloud is already in the global frame.
    """
    t0 = time.perf_counter()
    p = render_cfg(cfg)
    k = np.asarray(cam["K"], dtype=np.float64)
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    width, height = int(cam["width"]), int(cam["height"])

    t_g2c = _invert_rigid(np.asarray(cam["T_cam2global"], dtype=np.float64).reshape(4, 4))
    t_m2c = t_g2c @ (np.eye(4) if T_model2global is None
                     else np.asarray(T_model2global, dtype=np.float64).reshape(4, 4))
    r_m2c, t_vec = t_m2c[:3, :3], t_m2c[:3, 3]

    xyz = np.asarray(cloud.xyz, dtype=np.float64)
    mu = xyz @ r_m2c.T + t_vec
    z = mu[:, 2]
    valid = (z > p["z_near"]) & (z < p["z_far"])

    # 3D covariance in camera frame: R_m2c @ R_q @ diag(s^2) @ (R_m2c @ R_q)^T
    r_total = np.einsum("ij,njk->nik", r_m2c, quat_to_rotmat(cloud.rot))
    s2 = np.asarray(cloud.scale, dtype=np.float64) ** 2
    sigma = (r_total * s2[:, None, :]) @ np.transpose(r_total, (0, 2, 1))

    # EWA: project the 3D covariance through the Jacobian of the projection
    inv_z = 1.0 / np.maximum(z, 1e-6)
    jac = np.zeros((len(z), 2, 3), dtype=np.float64)
    jac[:, 0, 0] = fx * inv_z
    jac[:, 0, 2] = -fx * mu[:, 0] * inv_z * inv_z
    jac[:, 1, 1] = fy * inv_z
    jac[:, 1, 2] = -fy * mu[:, 1] * inv_z * inv_z
    cov2d = np.einsum("nij,njk,nlk->nil", jac, sigma, jac)
    cov2d[:, 0, 0] += p["cov2d_floor"]
    cov2d[:, 1, 1] += p["cov2d_floor"]

    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    tr = a + c
    det = np.maximum(a * c - b * b, 1e-12)
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    lmax = tr / 2.0 + disc
    radius = p["radius_scale"] * np.sqrt(np.maximum(lmax, 0.0))
    radius = np.clip(radius, p["r_min_px"], p["r_max_px"])

    u = fx * mu[:, 0] * inv_z + cx
    v = fy * mu[:, 1] * inv_z + cy

    on_frame = valid & (u + radius > 0) & (u - radius < width) \
        & (v + radius > 0) & (v - radius < height)
    keep = np.flatnonzero(on_frame)
    if len(keep) == 0:
        return RenderOut(
            rgb=np.zeros((height, width, 3), np.uint8),
            alpha=np.zeros((height, width), np.float32),
            depth=np.full((height, width), np.inf, np.float32),
            mask=np.zeros((height, width), bool),
            info={"n_gauss": int(len(cloud)), "n_kept": 0, "n_pairs": 0, "n_px": 0,
                  "t_sec": time.perf_counter() - t0},
        )

    u, v, z = u[keep], v[keep], z[keep]
    radius = radius[keep]
    a, b, c, det = a[keep], b[keep], c[keep], det[keep]
    opacity = np.clip(np.asarray(cloud.opacity, dtype=np.float64)[keep], 0.0, p["alpha_max"])
    colors = np.asarray(cloud.rgb, dtype=np.float64)[keep]

    x0 = np.clip(np.ceil(u - radius).astype(np.int64), 0, width)
    x1 = np.clip(np.floor(u + radius).astype(np.int64) + 1, 0, width)
    y0 = np.clip(np.ceil(v - radius).astype(np.int64), 0, height)
    y1 = np.clip(np.floor(v + radius).astype(np.int64) + 1, 0, height)
    counts = np.maximum((x1 - x0) * (y1 - y0), 0)

    total = int(counts.sum())
    idx = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    starts = np.cumsum(counts) - counts
    local = np.arange(total, dtype=np.int64) - starts[idx]
    width_i = (x1 - x0)[idx]
    gx = x0[idx] + (local % width_i)
    gy = y0[idx] + (local // width_i)
    flat = gy * width + gx

    dx = gx - u[idx]
    dy = gy - v[idx]
    # quadratic form d^T Σ2d^-1 d with Σ2d = [[a,b],[b,c]]
    quad = (c[idx] * dx * dx - 2.0 * b[idx] * dx * dy + a[idx] * dy * dy) / det[idx]
    w = opacity[idx] * np.exp(-0.5 * quad)

    # pass 1: per-pixel nearest depth among significant footprints
    # (sort by (flat, z) then take the first row of each flat group)
    sig1 = w > p["w_min"]
    zmin = np.full(height * width, np.inf, dtype=np.float64)
    if np.any(sig1):
        f1, z1 = flat[sig1], z[idx][sig1]
        order = np.lexsort((z1, f1))
        f1s, z1s = f1[order], z1[order]
        uniq, first = np.unique(f1s, return_index=True)
        zmin[uniq] = z1s[first]

    # pass 2: soft-z visibility + normalized color / alpha / depth
    sig2 = w > p["w_acc"]
    f2, w2, z2, i2 = flat[sig2], w[sig2], z[idx][sig2], idx[sig2]
    vis = np.exp(-np.maximum(0.0, z2 - zmin[f2]) / p["tau_m"])
    wv = w2 * vis
    npx = height * width
    log_term = np.bincount(f2, weights=np.log1p(-wv), minlength=npx)
    wsum = np.bincount(f2, weights=wv, minlength=npx)
    zsum = np.bincount(f2, weights=wv * z2, minlength=npx)
    rgb_acc = np.stack([
        np.bincount(f2, weights=wv * colors[i2, k], minlength=npx)
        for k in range(3)
    ], axis=1)

    alpha = np.maximum(-np.expm1(log_term), 0.0).reshape(height, width)
    wsum = wsum.reshape(height, width)
    with np.errstate(invalid="ignore", divide="ignore"):
        col = rgb_acc.reshape(height, width, 3) / np.maximum(wsum, 1e-9)[:, :, None]
        depth = zsum.reshape(height, width) / np.maximum(wsum, 1e-9)
    mask = alpha > p["depth_eps"]
    col = np.where(mask[:, :, None], col, 0.0)
    depth = np.where(mask, depth, np.inf)
    col = np.clip(col, 0.0, 1.0)

    return RenderOut(
        rgb=(col * 255.0 + 0.5).astype(np.uint8),
        alpha=alpha.astype(np.float32),
        depth=depth.astype(np.float32),
        mask=mask,
        info={"n_gauss": int(len(cloud)), "n_kept": int(len(keep)),
              "n_pairs": total, "n_px": int(mask.sum()),
              "t_sec": time.perf_counter() - t0},
    )
