#!/usr/bin/env python
"""Render the asset-library contact sheet (docs/figures/fig_asset_grid.png).

Each row is one normalized asset shown in three orthographic views
(front XZ / side YZ / top XY), colored by texture or reflectance, so the docs
can be re-generated after any ingest change:

    python scripts/make_asset_sheet.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# only 'Noto Sans CJK JP' resolves on this machine; fall back to English labels
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

VIEWS = (("front", 0, 2), ("side", 1, 2), ("top", 0, 1))


def _load_points(npz_path: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        canon = {k: data[k] for k in data.files}
    if "xyz" in canon:  # av2 cloud
        xyz = np.asarray(canon["xyz"], dtype=float)
        inten = np.asarray(canon.get("intensity", np.zeros(len(xyz))), dtype=float) / 255.0
        colors = np.repeat(inten[:, None], 3, axis=1)
    else:  # canonical mesh
        verts = np.asarray(canon["verts"], dtype=float)
        faces = np.asarray(canon["faces"], dtype=np.int64)
        vcols = np.asarray(canon["colors"], dtype=float)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        pts, fi = trimesh.sample.sample_surface(mesh, min(n, max(200, len(faces))), seed=0)
        tri = verts[faces[fi]]
        bary = trimesh.triangles.points_to_barycentric(tri, pts)
        colors = (bary[:, :, None] * vcols[faces[fi]]).sum(axis=1)
        xyz = pts
    if len(xyz) > n:
        idx = np.random.default_rng(0).choice(len(xyz), n, replace=False)
        xyz, colors = xyz[idx], colors[idx]
    return xyz, colors


def _pick(records: list[dict], kind: str, max_n: int) -> list[dict]:
    pool = [r for r in records if r["kind"] == kind and r["quality"].get("passed")]
    if kind == "av2_cloud":  # stratify: one per category first
        seen, ordered = set(), []
        for r in pool:
            if r["category"] not in seen:
                seen.add(r["category"])
                ordered.append(r)
        ordered += [r for r in pool if r not in ordered]
        pool = ordered
    return pool[:max_n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="data/library/assets_manifest.json")
    ap.add_argument("--library", default="data/library")
    ap.add_argument("--out", default="docs/figures/fig_asset_grid.png")
    ap.add_argument("--max-realcar", type=int, default=8)
    ap.add_argument("--max-av2", type=int, default=6)
    ap.add_argument("--points", type=int, default=1200)
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not manifest_path.exists():
        print(f"FATAL: manifest not found: {manifest_path} "
              f"(run scripts/build_asset_library.py first)", file=sys.stderr)
        return 2
    data = json.loads(manifest_path.read_text())
    library = Path(args.library)
    if not library.is_absolute():
        library = ROOT / library

    rows = _pick(data["assets"], "realcar_mesh", args.max_realcar)
    rows += _pick(data["assets"], "av2_cloud", args.max_av2)
    if not rows:
        print("FATAL: no passed assets to plot", file=sys.stderr)
        return 2

    fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 1.35 * len(rows)), dpi=140,
                             squeeze=False)
    for row, record in enumerate(rows):
        xyz, colors = _load_points(library / record["canonical_path"], args.points)
        dims = record["dims_lwh_m"]
        label = (f"{record['asset_id']}  —  "
                 f"L/W/H {dims[0]:.2f}/{dims[1]:.2f}/{dims[2]:.2f} m")
        for col, (name, ia, ib) in enumerate(VIEWS):
            ax = axes[row][col]
            ax.scatter(xyz[:, ia], xyz[:, ib], s=1.2, c=colors, linewidths=0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0 and col > 0:
                ax.set_title(name, fontsize=9)
            if col == 0:
                ax.set_title(label, fontsize=6.5, loc="left")
    fig.suptitle("3Dreal asset library — canonical frame (Z-up, ground z=0), "
                 "colored by texture / reflectance", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out} ({len(rows)} assets x 3 views)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
