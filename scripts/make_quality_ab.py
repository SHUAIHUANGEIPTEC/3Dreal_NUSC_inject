#!/usr/bin/env python
"""Render-quality A/B: legacy soft-z rasterizer vs the optimized CPU path vs gsplat.

Extracts the pre-optimization ``raster_cpu.py`` from git (``HEAD``), renders a
synthetic car-like asset with all three paths and writes a comparison figure
plus a metrics table (hole rate / surface grain / edge hard-jumps / timing).

    python scripts/make_quality_ab.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _legacy_render(cam, cloud, T_model2global=None, cfg=None):
    """Pre-optimization rasterizer, replicated for the A/B baseline.

    The three defining behaviours that were fixed: integer-pixel sampling
    (no half-pixel centre), plain soft-z (``z - zmin``, no thickness term)
    and no supersampling / post-processing.
    """
    from realinject.gauss.raster_cpu import (
        DEFAULTS,
        RenderOut,
        _invert_rigid,
        quat_to_rotmat,
    )

    p = dict(DEFAULTS)
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
    r_total = np.einsum("ij,njk->nik", r_m2c, quat_to_rotmat(cloud.rot))
    s2 = np.asarray(cloud.scale, dtype=np.float64) ** 2
    sigma = (r_total * s2[:, None, :]) @ np.transpose(r_total, (0, 2, 1))
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
    radius = np.clip(p["radius_scale"] * np.sqrt(np.maximum(lmax, 0.0)),
                     p["r_min_px"], p["r_max_px"])
    u = fx * mu[:, 0] * inv_z + cx
    v = fy * mu[:, 1] * inv_z + cy
    on = valid & (u + radius > 0) & (u - radius < width) \
        & (v + radius > 0) & (v - radius < height)
    keep = np.flatnonzero(on)
    u, v, z = u[keep], v[keep], z[keep]
    radius = radius[keep]
    a, b, c, det = a[keep], b[keep], c[keep], det[keep]
    opacity = np.clip(np.asarray(cloud.opacity, np.float64)[keep], 0.0, p["alpha_max"])
    colors = np.asarray(cloud.rgb, np.float64)[keep]
    x0 = np.clip(np.ceil(u - radius).astype(np.int64), 0, width)
    x1 = np.clip(np.floor(u + radius).astype(np.int64) + 1, 0, width)
    y0 = np.clip(np.ceil(v - radius).astype(np.int64), 0, height)
    y1 = np.clip(np.floor(v + radius).astype(np.int64) + 1, 0, height)
    counts = np.maximum((x1 - x0) * (y1 - y0), 0)
    idx = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    starts = np.cumsum(counts) - counts
    local = np.arange(int(counts.sum()), dtype=np.int64) - starts[idx]
    width_i = (x1 - x0)[idx]
    gx = x0[idx] + (local % width_i)
    gy = y0[idx] + (local // width_i)
    flat = gy * width + gx
    dx, dy = gx - u[idx], gy - v[idx]  # legacy: integer pixel, no +0.5
    quad = (c[idx] * dx * dx - 2.0 * b[idx] * dx * dy + a[idx] * dy * dy) / det[idx]
    w = opacity[idx] * np.exp(-0.5 * quad)
    sig1 = w > p["w_min"]
    zmin = np.full(height * width, np.inf, np.float64)
    f1, z1 = flat[sig1], z[idx][sig1]
    order = np.lexsort((z1, f1))
    f1s, z1s = f1[order], z1[order]
    uniq, first = np.unique(f1s, return_index=True)
    zmin[uniq] = z1s[first]
    sig2 = w > p["w_acc"]
    f2, w2, z2, i2 = flat[sig2], w[sig2], z[idx][sig2], idx[sig2]
    vis = np.exp(-np.maximum(0.0, z2 - zmin[f2]) / p["tau_m"])  # legacy: no thickness
    wv = w2 * vis
    npx = height * width
    log_term = np.bincount(f2, weights=np.log1p(-wv), minlength=npx)
    wsum = np.bincount(f2, weights=wv, minlength=npx)
    zsum = np.bincount(f2, weights=wv * z2, minlength=npx)
    rgb_acc = np.stack([np.bincount(f2, weights=wv * colors[i2, kk], minlength=npx)
                        for kk in range(3)], axis=1)
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
        info={"n_gauss": int(len(cloud)), "n_kept": int(len(keep))},
    )


def _car_cloud(n=60000, seed=0):
    """Synthetic sedan: cabin + hood boxes, 4 cylinder wheels, grainy paint."""
    import trimesh

    rng = np.random.default_rng(seed)
    body = trimesh.creation.box(extents=[4.4, 1.8, 1.1])
    body.apply_translation([0, 0, 0.75])
    cabin = trimesh.creation.box(extents=[2.2, 1.7, 0.75])
    cabin.apply_translation([-0.3, 0, 1.6])
    meshes = [body, cabin]
    for sx in (1.4, -1.4):
        for sy in (0.95, -0.95):
            wheel = trimesh.creation.cylinder(radius=0.34, height=0.24, sections=24)
            wheel.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi / 2, [1, 0, 0]))
            wheel.apply_translation([sx, sy, 0.34])
            meshes.append(wheel)
    areas = np.array([m.area for m in meshes])
    weights = areas / areas.sum()
    pts, face_idx, mesh_idx = [], [], []
    for i, m in enumerate(meshes):
        k = int(round(n * weights[i]))
        p, fi = trimesh.sample.sample_surface(m, k, seed=seed + i)
        pts.append(p)
        face_idx.append(fi)
        mesh_idx.append(np.full(k, i, dtype=np.int64))
    pts = np.concatenate(pts)
    mesh_idx = np.concatenate(mesh_idx)

    # paint: silver body / dark cabin / black wheels + per-point grain
    cols = np.empty((len(pts), 3), np.float32)
    for i, base in enumerate(([0.72, 0.74, 0.76], [0.16, 0.17, 0.19],
                              [0.10, 0.10, 0.11])):
        sel = mesh_idx == i
        cols[sel] = base
    cols = np.clip(cols + rng.uniform(-0.10, 0.10, cols.shape), 0, 1)

    from realinject.gauss.model import GaussianCloud, identity_quat

    scale = np.clip(rng.uniform(0.010, 0.030, len(pts)), 0.005, 0.15)
    return GaussianCloud(
        xyz=pts.astype(np.float32),
        rot=identity_quat(len(pts)),
        scale=scale.astype(np.float32)[:, None].repeat(3, axis=1),
        opacity=np.full(len(pts), 0.92, np.float32),
        rgb=cols,
    )


def _metrics(rgb, alpha):
    from scipy import ndimage

    m = alpha > 0.5
    closed = ndimage.binary_closing(m, structure=np.ones((5, 5), bool))
    filled = ndimage.binary_fill_holes(closed)
    holes = float((filled & ~m).sum() / max(int(m.sum()), 1))

    gray = rgb.astype(np.float32).mean(axis=2)
    mu = ndimage.uniform_filter(gray, 7)
    mu2 = ndimage.uniform_filter(gray * gray, 7)
    interior = ndimage.binary_erosion(alpha > 0.8, np.ones((5, 5), bool))
    noise = float(np.sqrt(np.maximum(mu2 - mu * mu, 0))[interior].mean())

    er = ndimage.binary_erosion(m, np.ones((3, 3), bool))
    hard = float((alpha[m & ~er] > 0.95).mean())
    return {"hole%": holes * 100, "grain": noise, "hard_edge%": hard * 100,
            "alpha_max": float(alpha.max()), "opaque%": float(
                (alpha > 0.9).sum() / max(int(m.sum()), 1)) * 100}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=60000)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--out", default="data/outputs/quality_ab/ab.png")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from realinject.gauss.render import get_renderer

    cloud = _car_cloud(args.samples)
    w, h = args.width, args.height
    fx = fy = 0.62 * w

    def _lookat(eye, target, up=(0.0, 0.0, 1.0)):
        f = np.asarray(target, float) - np.asarray(eye, float)
        f = f / np.linalg.norm(f)
        x = np.cross(f, np.asarray(up, float))
        x = x / np.linalg.norm(x)
        y = np.cross(f, x)  # OpenCV down
        t = np.eye(4)
        t[:3, :3] = np.column_stack([x, y, f])
        t[:3, 3] = eye
        return t

    cam = {
        "name": "CAM_AB",
        "K": [[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]],
        "T_cam2global": _lookat((6.5, -8.5, 2.6), (0.3, 0.0, 0.9)).tolist(),
        "width": w, "height": h,
    }

    rows = []
    # 1) legacy: integer-pixel sampling, raw soft-z, no SSAA / cleanup
    t0 = time.perf_counter()
    old = _legacy_render(cam, cloud)
    rows.append(("legacy (pre-fix)", old, time.perf_counter() - t0))
    # 2) optimized CPU: pixel centres + ssaa2 + thickness-aware z + cleanup
    t0 = time.perf_counter()
    new_cpu = get_renderer({"gaussian": {"render": {"backend": "cpu"}}})(
        cam, cloud, None, None)
    rows.append(("optimized cpu", new_cpu, time.perf_counter() - t0))
    # 3) gsplat on CUDA
    try:
        t0 = time.perf_counter()
        new_gpu = get_renderer({"gaussian": {"render": {"backend": "gsplat"}}})(
            cam, cloud, None, {"gaussian": {"render": {"backend": "gsplat"}}})
        rows.append(("gsplat (cuda)", new_gpu, time.perf_counter() - t0))
    except RuntimeError as exc:
        print(f"WARN: gsplat unavailable ({exc})", file=sys.stderr)

    bg = 160.0
    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.1 * len(rows)),
                             dpi=120, squeeze=False)
    print(f"{'path':18s} {'hole%':>7s} {'grain':>7s} {'hard%':>7s} "
          f"{'amax':>6s} {'opaque%':>8s} {'sec':>6s}")
    for r, (name, out, sec) in enumerate(rows):
        m = _metrics(out.rgb, out.alpha)
        print(f"{name:18s} {m['hole%']:7.3f} {m['grain']:7.2f} "
              f"{m['hard_edge%']:7.2f} {m['alpha_max']:6.3f} "
              f"{m['opaque%']:8.1f} {sec:6.2f}")
        comp = out.rgb.astype(np.float32) * out.alpha[..., None] \
            + bg * (1 - out.alpha[..., None])
        axes[r][0].imshow(comp.astype(np.uint8))
        axes[r][0].set_title(f"{name} — composite", fontsize=9)
        axes[r][1].imshow(out.alpha, cmap="magma", vmin=0, vmax=1)
        axes[r][1].set_title(
            f"alpha (hole {m['hole%']:.2f}%, grain {m['grain']:.1f})", fontsize=9)
        finite = np.isfinite(out.depth)
        d = np.where(finite, out.depth, np.nan)
        axes[r][2].imshow(d, cmap="viridis",
                          vmin=np.nanpercentile(d, 2),
                          vmax=np.nanpercentile(d, 98))
        axes[r][2].set_title(f"depth — {sec:.2f}s", fontsize=9)
        for c in range(3):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
    fig.suptitle(f"render quality A/B — {len(cloud)} gaussians @ {w}x{h}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
