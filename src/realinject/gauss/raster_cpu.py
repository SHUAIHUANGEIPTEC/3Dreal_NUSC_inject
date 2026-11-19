"""CPU Gaussian splat rasterizer.

EWA footprint projection (2D covariance of each anisotropic Gaussian) with an
**order-independent soft-z accumulation** instead of front-to-back transmittance
blending:

- per-pixel nearest depth is taken over footprints with weight > ``w_min``
  (``zmin``), using each Gaussian's **front surface** ``z - thickness`` so
  neighbouring samples of the same curved surface do not occlude each other;
- every footprint pair contributes ``w = opacity * exp(-0.5 d^T Σ2d^-1 d)``
  scaled by ``exp(-max(0, z_back - zmin) / tau_m)`` — surfaces further than
  ``tau_m`` behind the nearest one are suppressed, which approximates occlusion
  for opaque assets (our assets are sampled surfaces, not volumetric media);
- color is the weight-normalized mean ``Σ w c / Σ w``, alpha is the
  independent transmittance product ``1 - exp(Σ log1p(-w))``, depth the
  weight-normalized mean z.

Quality stages (all configurable under ``gaussian.render``):

1. **pixel-centre sampling** — footprint samples sit at ``(gx+0.5, gy+0.5)``,
   the true pixel centre (integer sampling shifts every footprint by half a
   pixel and produces systematic aliasing);
2. **supersampling** (``ssaa``) — the scene is rasterized at ``ssaa ×``
   resolution and alpha-weighted downsampled (alpha / unpremultiplied rgb /
   depth are all reconstructed from the sub-samples), which removes edge
   stair-stepping and per-pixel coverage noise;
3. **depth-thickness visibility** (``z_thickness_k``) — occlusion compares each
   Gaussian's back surface against the nearest front surface, so surface
   sampling gaps do not punch holes through the object;
4. **edge-preserving cleanup** (``smooth_rgb`` / ``min_blob_px``) — a
   bilateral filter smooths residual color grain on the surface while keeping
   texture edges sharp, and isolated specks below ``min_blob_px`` are
   dropped. Alpha itself is never filtered (supersampling already
   reconstructs coverage; an aggressive filter would dim true opacity). The
   cleanup never changes the output contract: ``rgb`` stays unpremultiplied
   and ``mask == alpha > depth_eps``.

Exact transmittance blending is order-dependent and needs a Python-loop over
every Gaussian (seconds per frame at 120k splats); the soft-z form is fully
vectorized (bincount / lexsort over the footprint pairs) and stays
deterministic. ``tau_m`` (0.1 m) is the only approximation knob of the
visibility term.

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
    # --- quality knobs -------------------------------------------------
    "ssaa": 2,  # supersampling factor (1 = off)
    "z_thickness_k": 1.0,  # sigma multiplier for the front-surface depth
    # alpha is never smoothed: supersampling already reconstructs coverage,
    # and any filter aggressive enough to matter would dim true opacity.
    "smooth_rgb": True,  # bilateral smoothing of rgb on the surface
    "smooth_rgb_sigma": 28.0,  # bilateral color sigma for rgb in [0,255]
    "smooth_space_sigma": 4.0,  # bilateral spatial sigma, pixels
    "min_blob_px": 8,  # drop connected components smaller than this
    # memory guard: a close-up target can grow footprints to the r_max cap,
    # making the COO expansion (≈19 live int64/float64 arrays per pair)
    # reach tens of GB. Budget the pair count, then hard-fail with an
    # actionable message instead of letting the OOM killer take the process.
    "max_pairs": 200_000_000,
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


def _rasterize(cam: dict, cloud: GaussianCloud,
               T_model2global: np.ndarray | None, p: dict
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Raw soft-z accumulation at ``cam`` resolution.

    Returns ``(rgb float[0,1], alpha float, depth float[inf=empty], info)``;
    the caller runs supersample reconstruction and post-processing.
    """
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
    # view-direction (camera z) thickness of each splat: front/back surfaces
    sigma_z = np.sqrt(np.maximum(sigma[:, 2, 2], 0.0))

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
    # the pixel caps are output-resolution quantities: at ssaa>1 a given
    # physical footprint is s x larger in sub-pixel units, so both caps scale
    # with ssaa (otherwise close-up splats get culled by r_max_px)
    s_ssaa = max(1, int(p.get("ssaa", 1)))
    radius = np.clip(radius, p["r_min_px"], p["r_max_px"] * s_ssaa)

    u = fx * mu[:, 0] * inv_z + cx
    v = fy * mu[:, 1] * inv_z + cy

    # 1 output-pixel cull margin: at ssaa>1 the sub-pixel floor term is not
    # scaled with the intrinsics, so r'/s can be up to ~0.8 px tighter than
    # the output-resolution radius and would cull edge-touching splats
    margin = max(1, int(p.get("ssaa", 1)))
    on_frame = valid & (u + radius + margin > 0) & (u - radius - margin < width) \
        & (v + radius + margin > 0) & (v - radius - margin < height)
    keep = np.flatnonzero(on_frame)
    if len(keep) == 0:
        npx = height * width
        return (np.zeros((height, width, 3), np.float64),
                np.zeros((height, width), np.float64),
                np.full((height, width), np.inf, np.float64),
                {"n_gauss": int(len(cloud)), "n_kept": 0, "n_pairs": 0})

    u, v, z = u[keep], v[keep], z[keep]
    radius = radius[keep]
    a, b, c, det = a[keep], b[keep], c[keep], det[keep]
    sigma_z = sigma_z[keep]
    opacity = np.clip(np.asarray(cloud.opacity, dtype=np.float64)[keep], 0.0, p["alpha_max"])
    colors = np.asarray(cloud.rgb, dtype=np.float64)[keep]

    x0 = np.clip(np.ceil(u - radius).astype(np.int64), 0, width)
    x1 = np.clip(np.floor(u + radius).astype(np.int64) + 1, 0, width)
    y0 = np.clip(np.ceil(v - radius).astype(np.int64), 0, height)
    y1 = np.clip(np.floor(v + radius).astype(np.int64) + 1, 0, height)
    counts = np.maximum((x1 - x0) * (y1 - y0), 0)

    total = int(counts.sum())
    budget = int(p.get("max_pairs", 200_000_000))
    if total > budget:
        # one shrink pass: scale every radius by sqrt(budget/total) so the
        # pair count lands at the budget (3-sigma edge weight is negligible)
        shrink = float(np.sqrt(budget / total))
        radius = radius * shrink
        x0 = np.clip(np.ceil(u - radius).astype(np.int64), 0, width)
        x1 = np.clip(np.floor(u + radius).astype(np.int64) + 1, 0, width)
        y0 = np.clip(np.ceil(v - radius).astype(np.int64), 0, height)
        y1 = np.clip(np.floor(v + radius).astype(np.int64) + 1, 0, height)
        counts = np.maximum((x1 - x0) * (y1 - y0), 0)
        total = int(counts.sum())
        if total > budget:
            raise RuntimeError(
                f"CPU rasterizer footprint budget exceeded: {total} > {budget} "
                f"pairs at {width}x{height} (close-up target). Set "
                f"gaussian.render.backend: gsplat (GPU) or lower ssaa.")
    idx = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    starts = np.cumsum(counts) - counts
    local = np.arange(total, dtype=np.int64) - starts[idx]
    width_i = (x1 - x0)[idx]
    gx = x0[idx] + (local % width_i)
    gy = y0[idx] + (local // width_i)
    flat = gy * width + gx

    # sample at the true pixel centre (gx + 0.5); integer sampling shifts
    # every footprint by half a pixel and aliases the silhouette
    dx = gx + 0.5 - u[idx]
    dy = gy + 0.5 - v[idx]
    # quadratic form d^T Σ2d^-1 d with Σ2d = [[a,b],[b,c]]
    quad = (c[idx] * dx * dx - 2.0 * b[idx] * dx * dy + a[idx] * dy * dy) / det[idx]
    w = opacity[idx] * np.exp(-0.5 * quad)

    # pass 1: per-pixel nearest *front surface* (z - k*sigma_z) among
    # significant footprints — same-surface neighbours must not occlude each
    # other (that is what punched holes through curved bodies).
    z_front = z - p["z_thickness_k"] * sigma_z
    sig1 = w > p["w_min"]
    zmin = np.full(height * width, np.inf, dtype=np.float64)
    if np.any(sig1):
        f1, z1 = flat[sig1], z_front[idx][sig1]
        order = np.lexsort((z1, f1))
        f1s, z1s = f1[order], z1[order]
        uniq, first = np.unique(f1s, return_index=True)
        zmin[uniq] = z1s[first]

    # pass 2: soft-z visibility + normalized color / alpha / depth.
    # Front-surface vs front-surface: the splat that *defines* zmin has
    # gap 0 (no self-suppression), while a splat sitting behind a thick
    # neighbour is still pushed back by that neighbour's front surface —
    # this is what closes sampling holes without dimming the whole body.
    sig2 = w > p["w_acc"]
    f2, w2 = flat[sig2], w[sig2]
    i2 = idx[sig2]
    vis = np.exp(-np.maximum(0.0, z_front[i2] - zmin[f2]) / p["tau_m"])
    wv = w2 * vis
    npx = height * width
    log_term = np.bincount(f2, weights=np.log1p(-wv), minlength=npx)
    wsum = np.bincount(f2, weights=wv, minlength=npx)
    zsum = np.bincount(f2, weights=wv * z[i2], minlength=npx)
    rgb_acc = np.stack([
        np.bincount(f2, weights=wv * colors[i2, k], minlength=npx)
        for k in range(3)
    ], axis=1)

    alpha = np.maximum(-np.expm1(log_term), 0.0).reshape(height, width)
    wsum = wsum.reshape(height, width)
    with np.errstate(invalid="ignore", divide="ignore"):
        col = rgb_acc.reshape(height, width, 3) / np.maximum(wsum, 1e-9)[:, :, None]
        depth = zsum.reshape(height, width) / np.maximum(wsum, 1e-9)
    depth = np.where(alpha > p["depth_eps"], depth, np.inf)

    info = {"n_gauss": int(len(cloud)), "n_kept": int(len(keep)),
            "n_pairs": total}
    return col, alpha, depth, info


def _downsample(col: np.ndarray, alpha: np.ndarray, depth: np.ndarray,
                s: int, p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the output-resolution buffers from an ``s x s`` sub-grid.

    alpha: sub-sample mean; rgb / depth: alpha-weighted mean so the
    unpremultiplied colour contract survives the reconstruction.
    """
    if s <= 1:
        return col, alpha, depth
    h, w = alpha.shape
    a = alpha.reshape(h // s, s, w // s, s)
    a_sum = a.sum(axis=(1, 3))
    alpha_o = a_sum / (s * s)
    c = col.reshape(h // s, s, w // s, s, 3)
    col_o = (c * a[..., None]).sum(axis=(1, 3)) / np.maximum(a_sum, 1e-12)[..., None]
    d = depth.reshape(h // s, s, w // s, s)
    finite = np.isfinite(d)
    wf = np.where(finite, a, 0.0)
    wf_sum = wf.sum(axis=(1, 3))
    d_o = np.where(wf_sum > 1e-12,
                   (np.where(finite, d, 0.0) * wf).sum(axis=(1, 3))
                   / np.maximum(wf_sum, 1e-12),
                   np.inf)
    return col_o, alpha_o, d_o


def _postprocess(col: np.ndarray, alpha: np.ndarray, depth: np.ndarray,
                 p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop isolated specks and bilateral-smooth residual grain.

    Edges are preserved by the bilateral filter; depth is left untouched so
    the depth contract (and any parity check) stays exact.
    """
    if int(p["min_blob_px"]) > 0:
        from scipy import ndimage

        m0 = alpha > p["depth_eps"]
        if m0.any():
            lab, n = ndimage.label(m0, structure=np.ones((3, 3), bool))
            sizes = np.bincount(lab.ravel())
            too_small = sizes < int(p["min_blob_px"])
            too_small[0] = False  # background label
            bad = too_small[lab]
            alpha = np.where(bad, 0.0, alpha)
            col = np.where(bad[..., None], 0.0, col)
            depth = np.where(bad, np.inf, depth)

    if bool(p["smooth_rgb"]):
        import cv2

        m = alpha > p["depth_eps"]
        if m.any():
            col = np.nan_to_num(col, nan=0.0, posinf=1.0, neginf=0.0)
            rgb8 = (np.clip(col, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            sm = cv2.bilateralFilter(rgb8, d=7,
                                     sigmaColor=float(p["smooth_rgb_sigma"]),
                                     sigmaSpace=float(p["smooth_space_sigma"]))
            col = np.where(m[..., None], sm.astype(np.float64) / 255.0, col)

    return col, alpha, depth


def _finalize(col: np.ndarray, alpha: np.ndarray, depth: np.ndarray,
              p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enforce the output contract: mask == alpha > depth_eps, inf outside."""
    mask = (alpha > p["depth_eps"]) & np.isfinite(depth)
    col = np.where(mask[..., None], np.clip(col, 0.0, 1.0), 0.0)
    depth = np.where(mask, depth, np.inf)
    alpha = np.where(mask, alpha, 0.0)
    return ((col * 255.0 + 0.5).astype(np.uint8),
            alpha.astype(np.float32),
            depth.astype(np.float32), mask)


def render_gaussians(cam: dict, cloud: GaussianCloud,
                     T_model2global: np.ndarray | None = None,
                     cfg: dict | None = None) -> RenderOut:
    """Rasterize one GaussianCloud into one camera.

    cam: CameraSpec dict (name/K/T_cam2global/width/height), OpenCV camera.
    T_model2global: 4x4; None means the cloud is already in the global frame.
    """
    t0 = time.perf_counter()
    p = render_cfg(cfg)
    s = max(1, int(p["ssaa"]))
    if s > 1:
        cam_hi = dict(cam)
        k = np.asarray(cam["K"], dtype=np.float64)
        k_hi = k.copy()
        k_hi[0, 0] *= s
        k_hi[1, 1] *= s
        k_hi[0, 2] *= s
        k_hi[1, 2] *= s
        cam_hi["K"] = k_hi
        cam_hi["width"] = int(cam["width"]) * s
        cam_hi["height"] = int(cam["height"]) * s
    else:
        cam_hi = cam

    col, alpha, depth, info = _rasterize(cam_hi, cloud, T_model2global, p)
    col, alpha, depth = _downsample(col, alpha, depth, s, p)
    col, alpha, depth = _postprocess(col, alpha, depth, p)
    rgb, alpha, depth, mask = _finalize(col, alpha, depth, p)

    info.update({"ssaa": s, "n_px": int(mask.sum()),
                 "t_sec": time.perf_counter() - t0})
    return RenderOut(rgb=rgb, alpha=alpha, depth=depth, mask=mask, info=info)


__all__ = ["DEFAULTS", "RenderOut", "quat_to_rotmat", "render_cfg",
           "render_gaussians"]
