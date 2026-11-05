"""Canonical asset -> GaussianCloud conversion.

Two initialization paths (the renderer consumes only GaussianCloud):

- mesh    area-weighted surface samples; texture color via barycentric
          interpolation of per-vertex colors; anisotropic scale oriented along
          the face normal; opacity from the mesh init config.
- points  raw track points; isotropic scale from kNN spacing; color from
          reflectance (``intensity``) or local density.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..gauss.model import GaussianCloud, identity_quat, quat_from_z_axis

DEFAULT_INIT: dict = {
    "mesh_samples": 120_000,
    "mesh_scale_mul": 0.8,
    "point_scale_mul": 1.2,
    "knn": 8,
    "opacity_mesh": 0.9,
    "opacity_points": 0.85,
    "scale_range": [0.003, 0.15],  # meters
    "color_mode": "intensity",  # intensity | density
}


def init_from_cfg(cfg: dict | None) -> dict:
    out = dict(DEFAULT_INIT)
    out.update((cfg or {}).get("gaussian_init") or {})
    return out


def knn_scale(xyz: np.ndarray, k: int, mul: float,
              scale_range: list[float]) -> np.ndarray:
    """Per-point isotropic sigma from mean kNN distance, clipped."""
    xyz = np.asarray(xyz, dtype=np.float64)
    k = int(max(2, min(k, len(xyz))))
    dist, _ = cKDTree(xyz).query(xyz, k=k)
    mean = dist[:, 1:].mean(axis=1) * mul
    lo, hi = float(scale_range[0]), float(scale_range[1])
    return np.clip(mean, lo, hi).astype(np.float32)


def mesh_to_gaussians(canon: dict, cfg: dict | None = None) -> GaussianCloud:
    """Sample a canonical mesh into anisotropic Gaussians."""
    init = init_from_cfg(cfg)
    verts = np.asarray(canon["verts"], dtype=np.float64)
    faces = np.asarray(canon["faces"], dtype=np.int64)
    colors = np.asarray(canon["colors"], dtype=np.float32)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    n = int(init["mesh_samples"])
    pts, face_idx = trimesh.sample.sample_surface(mesh, n, seed=0)
    tri = verts[faces[face_idx]]
    bary = trimesh.triangles.points_to_barycentric(tri, pts)
    tri_colors = colors[faces[face_idx]]
    rgb = (bary[:, :, None] * tri_colors).sum(axis=1)

    normals = mesh.face_normals[face_idx]
    scale = knn_scale(pts, int(init["knn"]), float(init["mesh_scale_mul"]),
                      init["scale_range"])
    scale = np.repeat(scale[:, None], 3, axis=1)
    return GaussianCloud(
        xyz=pts.astype(np.float32),
        rot=quat_from_z_axis(normals),
        scale=scale,
        opacity=np.full(n, float(init["opacity_mesh"]), dtype=np.float32),
        rgb=rgb.astype(np.float32),
        extra={"normal": np.asarray(normals, dtype=np.float32)},
    )


def _intensity_colors(intensity: np.ndarray) -> np.ndarray:
    gray = (np.asarray(intensity, dtype=np.float32) / 255.0)
    return np.repeat(gray[:, None], 3, axis=1)


def _density_colors(xyz: np.ndarray) -> np.ndarray:
    """turbo colormap on local point density (for debugging / demos)."""
    dist, _ = cKDTree(xyz).query(xyz, k=min(8, max(2, len(xyz))))
    dens = 1.0 / np.maximum(dist[:, 1:].mean(axis=1), 1e-4)
    lo, hi = np.percentile(dens, [2, 98])
    norm = np.clip((dens - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    lut = cv2.applyColorMap((norm * 255).astype(np.uint8).reshape(-1, 1),
                            cv2.COLORMAP_TURBO)[:, 0, :]
    return cv2.cvtColor(lut.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0


def points_to_gaussians(canon: dict, cfg: dict | None = None) -> GaussianCloud:
    """Lift a canonical point cloud to isotropic Gaussians."""
    init = init_from_cfg(cfg)
    xyz = np.asarray(canon["xyz"], dtype=np.float64)
    intensity = np.asarray(canon["intensity"] if "intensity" in canon
                           else np.zeros(len(xyz)), dtype=np.float32)
    mode = str(init["color_mode"])
    rgb = _density_colors(xyz) if mode == "density" else _intensity_colors(intensity)

    scale = knn_scale(xyz, int(init["knn"]), float(init["point_scale_mul"]),
                      init["scale_range"])
    scale = np.repeat(scale[:, None], 3, axis=1)
    n = len(xyz)
    return GaussianCloud(
        xyz=xyz.astype(np.float32),
        rot=identity_quat(n),
        scale=scale,
        opacity=np.full(n, float(init["opacity_points"]), dtype=np.float32),
        rgb=rgb.astype(np.float32),
        extra={"intensity": intensity},
    )


def record_to_cloud(record, library_root: str | Path, cfg: dict | None = None) -> GaussianCloud:
    """Load an AssetRecord's canonical npz and build its GaussianCloud."""
    npz = Path(library_root) / record.canonical_path
    with np.load(npz, allow_pickle=True) as data:
        canon = {k: data[k] for k in data.files}
    if record.kind == "realcar_mesh":
        cloud = mesh_to_gaussians(canon, cfg)
    elif record.kind == "av2_cloud":
        cloud = points_to_gaussians(canon, cfg)
    else:
        raise ValueError(f"unknown asset kind: {record.kind}")
    cloud.extra["asset_id"] = record.asset_id
    cloud.extra["provenance"] = str(npz)
    return cloud
