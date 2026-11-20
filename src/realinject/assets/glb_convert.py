"""GLB -> canonical mesh npz conversion (shared by the CLI and ingest_glb).

Same contract and transform chain as ``normalize.canonicalize_realcar``
(Y-up yaw align -> Blender frame -> bbox-min at origin -> UV-sampled
per-vertex colors), but the input is ``.glb`` instead of ``mesh.obj`` + ``mesh.jpg``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .normalize import (
    _face_normals_safe,
    boundary_edge_count,
    sample_texture,
    yaw_align,
    yup_to_blender,
)


def load_glb(path: str | Path):
    """Scene -> merged arrays: (verts_yup, faces, uv, tex_rgb | None, vcolors | None).

    Multi-geometry scenes are concatenated with index offsets; textured parts
    must carry per-vertex UV (road_ood assets are texture-selected).
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
                      vcolors: np.ndarray | None = None,
                      size_range: tuple[float, float] | None = None,
                      ) -> tuple[dict, dict]:
    """Run the canonical transform chain; returns (npz payload, stats).

    ``size_range`` (min, max) longest-extent prior: if the transformed mesh
    falls outside it, the whole model is uniformly scaled onto the nearest
    bound (Objaverse unit chaos — a 1.68-m fire_extinguisher becomes 0.7 m).
    """
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

    scale = 1.0
    if size_range is not None:
        longest = float(bsize.max())
        lo, hi = float(size_range[0]), float(size_range[1])
        if longest > hi:
            scale = hi / longest
        elif longest < lo:
            scale = lo / longest
        if scale != 1.0:
            verts_b = verts_b * scale
            bmin = bmin * scale
            bsize = bsize * scale

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
        # cache marker: ingest re-converts npz written before size priors
        "size_scale": np.float32(scale),
    }
    stats = {
        "dims_lwh_m": tuple(float(v) for v in bsize),
        "n_verts": int(len(verts_b)),
        "n_faces": int(len(faces)),
        "yaw_deg": float(np.degrees(yaw)),
        "size_scale": float(scale),
        "texture_std": float(colors.std() * 255.0),
        "texture_black_frac": float(
            (colors.max(axis=1) < 20.0 / 255.0).mean()),
        "boundary_edges": int(boundary_edge_count(tm)),
        "watertight": bool(tm.is_watertight),
    }
    return canon, stats


def glb_to_npz(src: str | Path, out_npz: str | Path) -> dict:
    """Convert one GLB file; returns stats (canonical_path included).

    Applies the class size prior (``quality.size_range_for_class``) so the
    cached npz already carries the corrected scale.
    """
    from .quality import size_range_for_class

    src = Path(src)
    verts, faces, uv, tex, vcolors = load_glb(src)
    canon, stats = canon_from_arrays(
        verts, faces, uv, tex, vcolors,
        size_range=size_range_for_class(src.stem))
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **canon)
    stats["canonical_path"] = str(out_npz)
    stats["source_glb"] = str(src)
    return stats
