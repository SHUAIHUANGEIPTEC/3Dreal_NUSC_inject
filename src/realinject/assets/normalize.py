"""Asset normalization into the canonical frame.

Canonical frame contract (identical to ``nusc_inject.mesh_points``'s Blender
import frame so meshes and raw point clouds stay comparable):

- Z-up, ground contact at ``z = 0``
- bbox min translated to the origin
- axes: ``X`` = length (long horizontal axis), ``Y`` = width, ``Z`` = height
- texture/uv/normals carried alongside; texture files are referenced, not copied

Realcar crops arrive Y-up with the car rotated ~45 deg in the XZ plane (the
exported AABB is inflated: 3.82 x 4.60 m for a 4.945 x 1.945 m car), so the
yaw must be recovered from the horizontal vertex distribution before the
Y-up -> Z-up conversion. AV2 track clouds are already Z-up and bbox-centred.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import trimesh

#: Y-up -> Blender Z-up: (x, y, z) -> (x, -z, y)
YUP_TO_BLENDER = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float
)


def rot_y(angle_rad: float) -> np.ndarray:
    """4x4 rotation about +Y (Y-up frame)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )


def yaw_align(verts_yup: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate Y-up vertices about Y so the long horizontal axis lies on +X.

    Returns (4x4 T_yaw, angle_rad). Deterministic: the principal axis sign is
    fixed so the rotated axis has a non-negative X component.
    """
    xz = np.asarray(verts_yup, dtype=float)[:, [0, 2]]
    xz = xz - xz.mean(axis=0)
    # eigen-decomposition of the 2x2 covariance: largest eigenvalue = length axis
    cov = xz.T @ xz / max(len(xz) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]
    if axis[0] < 0:  # deterministic sign: long axis points +X
        axis = -axis
    angle = float(np.arctan2(axis[1], axis[0]))
    return rot_y(angle), angle


def yup_to_blender(pts: np.ndarray) -> np.ndarray:
    return np.asarray(pts, dtype=float) @ YUP_TO_BLENDER.T


def load_texture(path: str | Path) -> np.ndarray:
    """Load a texture image as RGB uint8 (BGR file order fixed)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read texture: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def sample_texture(tex_rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear texture lookup for (V,2) uv -> (V,3) float32 in [0,1].

    OBJ/glTF uv origin is bottom-left, image rows start at the top.
    """
    h, w = tex_rgb.shape[:2]
    uv = np.asarray(uv, dtype=np.float64)
    x = np.clip(uv[:, 0] * (w - 1), 0, w - 1)
    y = np.clip((1.0 - uv[:, 1]) * (h - 1), 0, h - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    img = tex_rgb.astype(np.float32)
    c00, c10 = img[y0, x0], img[y0, x1]
    c01, c11 = img[y1, x0], img[y1, x1]
    top = c00 * (1 - fx) + c10 * fx
    bot = c01 * (1 - fx) + c11 * fx
    return (top * (1 - fy) + bot * fy) / 255.0


def load_mesh_raw(path: str | Path) -> trimesh.Trimesh:
    """Load a triangle mesh without trimesh's processing (keeps uv)."""
    obj = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(obj, trimesh.Scene):
        obj = obj.dump(concatenate=True)
    if not isinstance(obj, trimesh.Trimesh) or len(obj.faces) == 0:
        raise ValueError(f"no triangle mesh in {path}")
    return obj


def merge_uv_seam_vertices(mesh: trimesh.Trimesh, digits: int = 3) -> trimesh.Trimesh:
    """Merge vertices duplicated at UV seams (1 mm quantization).

    The 3DRealCar exporter duplicates vertices along UV seams, which inflates
    connectivity checks (34,754 -> 31,879 vertices on the reference asset).
    ``merge_tex``/``merge_norm`` keep genuinely distinct uv/normal pairs apart.
    """
    rnd = np.round(np.asarray(mesh.vertices, dtype=np.float64), digits)
    _, inverse = np.unique(rnd, axis=0, return_inverse=True)
    if len(np.unique(inverse)) == len(mesh.vertices):
        return mesh
    mesh = mesh.copy()
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh


def _face_normals_safe(mesh: trimesh.Trimesh) -> np.ndarray:
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    bad = ~np.isfinite(normals).all(axis=1) | (np.linalg.norm(normals, axis=1) < 1e-9)
    if np.any(bad):
        normals = normals.copy()
        normals[bad] = np.array([0.0, 0.0, 1.0])
    return normals


def boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    """Number of edges used by exactly one face (0 = closed)."""
    edges = np.sort(np.asarray(mesh.edges, dtype=np.int64), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts == 1))


def canonicalize_realcar(
    asset_dir: str | Path,
    meshes_root: str | Path,
    out_npz: str | Path,
) -> dict:
    """Normalize one 3DRealCar crop into ``out_npz``; return record stats.

    ``meshes_root`` is the assets root used to store the texture as a
    repository-relative path.
    """
    asset_dir = Path(asset_dir)
    obj_path = asset_dir / "mesh.obj"
    if not obj_path.exists():
        raise FileNotFoundError(f"missing mesh.obj under {asset_dir}")
    mesh = merge_uv_seam_vertices(load_mesh_raw(obj_path))

    verts_yup = np.asarray(mesh.vertices, dtype=float)
    uv = np.asarray(mesh.visual.uv, dtype=float) if hasattr(mesh.visual, "uv") else None
    if uv is None or len(uv) != len(verts_yup):
        raise ValueError(f"mesh has no usable uv: {obj_path}")
    vnormals = _face_normals_safe(mesh)

    t_yaw, yaw = yaw_align(verts_yup)
    # Y-up yaw alignment, then the Blender-frame conversion used by nusc-inject
    verts_b = yup_to_blender((t_yaw[:3, :3] @ verts_yup.T).T)
    norm_b = yup_to_blender((t_yaw[:3, :3] @ vnormals.T).T)

    bmin = verts_b.min(axis=0)
    verts_b = verts_b - bmin
    bsize = verts_b.max(axis=0)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    tex_path = asset_dir / "mesh.jpg"
    tex = load_texture(tex_path)
    colors = sample_texture(tex, uv)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        verts=verts_b.astype(np.float32),
        faces=faces,
        uv=uv.astype(np.float32),
        normals=norm_b.astype(np.float32),
        colors=colors.astype(np.float32),
        raw_bbox_min=bmin.astype(np.float32),
        raw_bbox_size=bsize.astype(np.float32),
        dims_lwh=bsize.astype(np.float32),
        yaw_rad=np.float32(yaw),
    )

    return {
        "canonical_path": str(out_npz),
        "dims_lwh_m": tuple(float(v) for v in bsize),
        "raw_bbox_min": tuple(float(v) for v in bmin),
        "raw_bbox_size": tuple(float(v) for v in bsize),
        "n_verts": int(len(verts_b)),
        "n_faces": int(len(faces)),
        "boundary_edges": boundary_edge_count(mesh),
        "watertight": bool(mesh.is_watertight),
        "yaw_deg": float(np.degrees(yaw)),
        "texture_path": str(tex_path.relative_to(Path(meshes_root).resolve())),
        "texture_std": float(colors.std() * 255.0),
        "aabb_extent": tuple(
            float(v) for v in (verts_yup.max(axis=0) - verts_yup.min(axis=0))
        ),
    }


def canonicalize_av2(npz_path: str | Path, out_npz: str | Path) -> dict:
    """Normalize one AV2 track cloud (Z-up, bbox-centred) into ``out_npz``."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as data:
        pts = np.asarray(data["pts"], dtype=float)
        intensity = np.asarray(data["intensity"])
        dims = np.asarray(data["dims"], dtype=float)
        ego = np.asarray(data["ego"], dtype=float) if "ego" in data else np.zeros((0, 3))
        cat = str(data["cat"]) if "cat" in data else ""
    if len(pts) < 3:
        raise ValueError(f"too few points in {npz_path}: {len(pts)}")

    bmin = pts.min(axis=0)
    xyz = pts - bmin
    bsize = xyz.max(axis=0)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        xyz=xyz.astype(np.float32),
        intensity=intensity.astype(np.uint8),
        raw_bbox_min=bmin.astype(np.float32),
        raw_bbox_size=bsize.astype(np.float32),
        box_lwh=dims.astype(np.float32),
        cloud_extent=bsize.astype(np.float32),
        ego=ego.astype(np.float32),
        cat=np.array(cat),
    )
    return {
        "canonical_path": str(out_npz),
        "dims_lwh_m": tuple(float(v) for v in dims),
        "cloud_extent": tuple(float(v) for v in bsize),
        "raw_bbox_min": tuple(float(v) for v in bmin),
        "raw_bbox_size": tuple(float(v) for v in bsize),
        "n_points": int(len(xyz)),
        "cat": cat,
        "n_frames": int(len(ego)),
    }


def load_canonical(path: str | Path) -> dict:
    """Load a canonical npz as a plain dict."""
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}
