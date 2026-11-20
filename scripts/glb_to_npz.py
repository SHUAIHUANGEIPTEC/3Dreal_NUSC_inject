#!/usr/bin/env python
"""Convert textured GLB models (Objaverse / road_ood) to the canonical mesh npz.

Same contract and transform chain as ``normalize.canonicalize_realcar``
(Y-up yaw align -> Blender frame -> bbox-min at origin -> UV-sampled
per-vertex colors), but the input is ``.glb`` instead of ``mesh.obj`` + ``mesh.jpg``:

    out.npz: verts (N,3) f32 | faces (F,3) i32 | uv (N,2) f32 |
             normals (N,3) f32 | colors (N,3) f32 in [0,1] |
             raw_bbox_min (3,) | raw_bbox_size (3,) | dims_lwh (3,) | yaw_rad ()

Usage:
    python scripts/glb_to_npz.py model.glb -o model.npz
    python scripts/glb_to_npz.py models_dir/ -o out_dir/            # batch *.glb
    python scripts/glb_to_npz.py models_dir/ -o out_dir/ --glob "fireplug_*.glb"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realinject.assets.normalize import (  # noqa: E402
    _face_normals_safe,
    boundary_edge_count,
    sample_texture,
    yaw_align,
    yup_to_blender,
)


def load_glb(path: str | Path):
    """Scene -> merged arrays: (verts_yup, faces, uv, tex_rgb | None, vcolors | None).

    Multi-geometry scenes are concatenated with index offsets; every part must
    carry per-vertex UV (road_ood assets are texture-selected, so this holds).
    """
    import trimesh

    path = Path(path)
    scene = trimesh.load(str(path), force="scene")
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    geoms = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh)
             and len(g.faces)]
    if not geoms:
        raise ValueError(f"no triangle mesh in {path}")

    vs, fs, uvs, has_uv = [], [], [], True
    off = 0
    for g in geoms:
        uv = getattr(g.visual, "uv", None)
        if uv is None or len(uv) != len(g.vertices):
            has_uv = False
            uv = np.zeros((len(g.vertices), 2), dtype=float)
        vs.append(np.asarray(g.vertices, dtype=float))
        uvs.append(np.asarray(uv, dtype=float))
        fs.append(np.asarray(g.faces, dtype=np.int32) + off)
        off += len(g.vertices)
    verts = np.vstack(vs)
    faces = np.vstack(fs)
    uv = np.vstack(uvs)

    # base-color texture among the parts (same pick as fetch_net_assets)
    tex = None
    for g in geoms:
        mat = getattr(g.visual, "material", None)
        img = (getattr(mat, "baseColorTexture", None)
               or getattr(mat, "image", None))
        if img is not None:
            tex = np.asarray(img.convert("RGB"), dtype=np.uint8)
            break

    # vertex-color fallback (untextured parts)
    vcolors = None
    if tex is None:
        for g in geoms:
            vis = g.visual
            if getattr(vis, "kind", None) == "vertex" and len(
                    getattr(vis, "vertex_colors", []) or []) == len(g.vertices):
                vcolors = np.asarray(vis.vertex_colors, dtype=np.uint8)[:, :3]
                break

    if tex is None and vcolors is None:
        raise ValueError(f"no texture and no vertex colors: {path}")
    if tex is not None and not has_uv:
        raise ValueError(f"texture present but mesh has no usable uv: {path}")
    return verts, faces, uv, tex, vcolors


def canon_from_arrays(verts_yup: np.ndarray, faces: np.ndarray,
                      uv: np.ndarray, tex_rgb: np.ndarray | None = None,
                      vcolors: np.ndarray | None = None) -> tuple[dict, dict]:
    """Run the canonical transform chain; returns (npz payload, stats)."""
    import trimesh

    verts_yup = np.asarray(verts_yup, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)
    uv = np.asarray(uv, dtype=float)

    # GLB has split verts at UV seams already — no seam merge (that would
    # average differing uvs apart); normals follow the yaw+frame transform
    tm = trimesh.Trimesh(vertices=verts_yup, faces=faces, process=False)
    vnormals = _face_normals_safe(tm)

    t_yaw, yaw = yaw_align(verts_yup)
    verts_b = yup_to_blender((t_yaw[:3, :3] @ verts_yup.T).T)
    norm_b = yup_to_blender((t_yaw[:3, :3] @ vnormals.T).T)

    bmin = verts_b.min(axis=0)
    verts_b = verts_b - bmin
    bsize = verts_b.max(axis=0)

    if tex_rgb is not None:
        colors = sample_texture(tex_rgb, uv)
    else:
        colors = np.asarray(vcolors, dtype=np.uint8)[:, :3] / 255.0

    canon = {
        "verts": verts_b.astype(np.float32),
        "faces": faces,
        "uv": uv.astype(np.float32),
        "normals": norm_b.astype(np.float32),
        "colors": colors.astype(np.float32),
        "raw_bbox_min": bmin.astype(np.float32),
        "raw_bbox_size": bsize.astype(np.float32),
        "dims_lwh": bsize.astype(np.float32),
        "yaw_rad": np.float32(yaw),
    }
    stats = {
        "dims_lwh_m": tuple(float(v) for v in bsize),
        "n_verts": int(len(verts_b)),
        "n_faces": int(len(faces)),
        "yaw_deg": float(np.degrees(yaw)),
        "texture_std": float(colors.std() * 255.0),
        "texture_black_frac": float(
            (colors.max(axis=1) < 20.0 / 255.0).mean()),
        "boundary_edges": int(boundary_edge_count(tm)),
        "watertight": bool(tm.is_watertight),
    }
    return canon, stats


def glb_to_npz(src: str | Path, out_npz: str | Path) -> dict:
    """Convert one GLB file; returns stats (canonical_path included)."""
    verts, faces, uv, tex, vcolors = load_glb(src)
    canon, stats = canon_from_arrays(verts, faces, uv, tex, vcolors)
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **canon)
    stats["canonical_path"] = str(out_npz)
    stats["source_glb"] = str(src)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="one .glb file, or a directory of .glb files")
    ap.add_argument("-o", "--out", required=True,
                    help="output .npz (file mode) or output dir (dir mode)")
    ap.add_argument("--glob", default="*.glb",
                    help="filename pattern in dir mode (default *.glb)")
    args = ap.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    if src.is_file():
        stats = glb_to_npz(src, out)
        print(f"[glb2npz] {src.name}: {stats['n_verts']} verts "
              f"dims={tuple(round(v, 2) for v in stats['dims_lwh_m'])} "
              f"yaw={stats['yaw_deg']:.1f}deg -> {out}")
        return 0

    files = sorted(src.glob(args.glob))
    if not files:
        print(f"FATAL: no files match {args.glob} under {src}", file=sys.stderr)
        return 2
    ok = failed = 0
    for f in files:
        dst = out / f"{f.stem}.npz"
        try:
            st = glb_to_npz(f, dst)
            ok += 1
            print(f"[glb2npz] ok   {f.name}: {st['n_verts']} verts "
                  f"dims={tuple(round(v, 2) for v in st['dims_lwh_m'])}")
        except Exception as exc:  # noqa: BLE001 - batch keeps going
            failed += 1
            print(f"[glb2npz] FAIL {f.name}: {type(exc).__name__}: {exc}")
    print(f"[glb2npz] done: {ok} converted, {failed} failed -> {out}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
