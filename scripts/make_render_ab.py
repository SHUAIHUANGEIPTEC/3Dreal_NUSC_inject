#!/usr/bin/env python
"""Render one asset twice — Gaussian splat vs stag3d point renderer — for docs.

Produces ``docs/figures/fig_gauss_vs_point.png``: rows = {gaussian, point},
columns = {composited RGB, alpha, depth}. The two rows must agree on
silhouette, color and depth (see docs/GAUSSIAN_RENDERING.md for the numbers).

    python scripts/make_render_ab.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


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


def _stag3d():
    src = ROOT.parent / "stag1_inject" / "src"
    if not (src / "stag3d").is_dir():
        return None
    sys.path.insert(0, str(src))
    try:
        from stag3d import geometry, renderer  # noqa: PLC0415
    finally:
        sys.path.pop(0)
    return geometry, renderer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--library", default="data/library")
    ap.add_argument("--manifest", default="data/library/assets_manifest.json")
    ap.add_argument("--asset", default=None, help="asset_id (default: first passed realcar)")
    ap.add_argument("--samples", type=int, default=60000)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--out", default="docs/figures/fig_gauss_vs_point.png")
    args = ap.parse_args(argv)

    from realinject.assets.convert import record_to_cloud
    from realinject.assets.manifest import AssetLibrary
    from realinject.gauss.raster_cpu import render_gaussians

    lib_path = Path(args.manifest)
    if not lib_path.is_absolute():
        lib_path = ROOT / lib_path
    lib = AssetLibrary.load(lib_path)
    pool = lib.select(kind="realcar_mesh")
    if args.asset:
        pool = [r for r in pool if r.asset_id == args.asset] + pool
    if not pool:
        print("FATAL: no passed realcar assets", file=sys.stderr)
        return 2
    record = pool[0]
    library = Path(args.library)
    if not library.is_absolute():
        library = ROOT / library
    cloud = record_to_cloud(record, library,
                            {"gaussian_init": {"mesh_samples": args.samples}})

    w, h = args.width, args.height
    fx = fy = 0.62 * w  # ~78 deg hfov, nuScenes-like
    eye = np.array([0.0, -12.0, 1.6])
    target = np.array([0.0, 0.0, 0.9])
    t_c2w = _lookat(eye, target)
    cam = {
        "name": "CAM_A", "K": [[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]],
        "T_cam2w": None, "T_cam2global": t_c2w.tolist(),
        "width": w, "height": h,
    }
    cam.pop("T_cam2w")

    out = render_gaussians(cam, cloud, np.eye(4))
    rgb_g = out.rgb.astype(np.float32)
    alpha_g = out.alpha
    depth_g = out.depth

    # reference: same points through the stag3d point renderer
    stag = _stag3d()
    if stag is None:
        print("WARN: stag3d unavailable; point row omitted", file=sys.stderr)
        rgb_p = alpha_p = depth_p = None
    else:
        geometry, renderer = stag
        intr = geometry.CameraIntrinsics.from_matrix(np.asarray(cam["K"], float), w, h)
        colors = (np.asarray(cloud.rgb) * 255).astype(np.uint8)
        img_p, depth_p = renderer.render_rgbd(
            np.asarray(cloud.xyz, dtype=float), colors, t_c2w, intr,
            point_size=3, background=0)
        rgb_p = img_p.astype(np.float32)
        alpha_p = (depth_p > 0).astype(np.float32)
        depth_p = np.where(depth_p > 0, depth_p, np.nan)  # stag3d uses 0 = empty

    bg = 255.0  # white background for the composite
    rows = [("gaussian", rgb_g, alpha_g, depth_g)]
    if rgb_p is not None:
        rows.append(("stag3d point", rgb_p, alpha_p, depth_p))

    fig, axes = plt.subplots(len(rows), 3, figsize=(9.5, 3.2 * len(rows)),
                             dpi=130, squeeze=False)
    for r, (name, rgb, alpha, depth) in enumerate(rows):
        comp = rgb * alpha[..., None] + bg * (1 - alpha[..., None])
        axes[r][0].imshow(comp.astype(np.uint8))
        axes[r][0].set_title(f"{name} — composite", fontsize=9)
        axes[r][1].imshow(alpha, cmap="magma", vmin=0, vmax=1)
        axes[r][1].set_title("alpha", fontsize=9)
        finite = np.isfinite(depth)
        if finite.any():
            d = np.where(finite, depth, np.nan)
            im = axes[r][2].imshow(d, cmap="viridis",
                                   vmin=np.nanpercentile(d, 2),
                                   vmax=np.nanpercentile(d, 98))
            fig.colorbar(im, ax=axes[r][2], fraction=0.046)
        axes[r][2].set_title("depth (m)", fontsize=9)
        for c in range(3):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
    fig.suptitle(f"asset {record.asset_id} — {len(cloud)} gaussians, "
                 f"{out.info['t_sec']:.2f}s @ {w}x{h}", fontsize=10)
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
