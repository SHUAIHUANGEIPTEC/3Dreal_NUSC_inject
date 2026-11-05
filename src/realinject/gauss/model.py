"""Gaussian cloud data model shared by every renderer and initializer.

Frame contract: ``xyz`` lives in the canonical asset frame, which is the same
frame ``nusc_inject.mesh_points`` produces (Blender Z-up import frame, ground
at ``z = 0``, bbox min translated to the origin in the asset library).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: per-point extras carried through every stage (not all renderers use them)
EXTRA_KEYS = ("normal", "intensity", "uv", "provenance", "asset_id")


@dataclass
class GaussianCloud:
    """Anisotropic Gaussian splat cloud.

    xyz:     (N,3) float32 positions, canonical frame
    rot:     (N,4) float32 rotation quaternion (w, x, y, z), unit norm
    scale:   (N,3) float32 per-axis std in meters
    opacity: (N,)  float32 in (0,1)
    rgb:     (N,3) float32 in [0,1]
    """

    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    rgb: np.ndarray
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(np.asarray(self.xyz))
        if n == 0:
            raise ValueError("empty GaussianCloud")
        self.xyz = np.asarray(self.xyz, dtype=np.float32).reshape(n, 3)
        self.rot = _normalize_quat(np.asarray(self.rot, dtype=np.float32).reshape(n, 4))
        self.scale = np.asarray(self.scale, dtype=np.float32).reshape(n, 3)
        self.opacity = np.asarray(self.opacity, dtype=np.float32).reshape(n)
        self.rgb = np.asarray(self.rgb, dtype=np.float32).reshape(n, 3)

    def __len__(self) -> int:
        return len(self.xyz)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "xyz": self.xyz,
            "rot": self.rot,
            "scale": self.scale,
            "opacity": self.opacity,
            "rgb": self.rgb,
        }
        for key in ("normal", "intensity", "uv"):
            val = self.extra.get(key)
            if val is not None:
                arrays[key] = np.asarray(val)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> GaussianCloud:
        with np.load(path) as data:
            arrays = {k: np.array(data[k]) for k in data.files}
        provenance = str(path)
        extra = {k: arrays[k] for k in ("normal", "intensity", "uv") if k in arrays}
        extra["provenance"] = provenance
        return cls(
            xyz=arrays["xyz"],
            rot=arrays["rot"],
            scale=arrays["scale"],
            opacity=arrays["opacity"],
            rgb=arrays["rgb"],
            extra=extra,
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.xyz.min(axis=0), self.xyz.max(axis=0)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (q / norms).astype(np.float32)


def identity_quat(n: int) -> np.ndarray:
    """(N,4) identity quaternions (w=1)."""
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 0] = 1.0
    return q


def quat_from_z_axis(normals: np.ndarray) -> np.ndarray:
    """Rotation taking local +Z to each given unit normal (shortest arc)."""
    n = np.asarray(normals, dtype=np.float32)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    z = np.zeros_like(n)
    z[:, 2] = 1.0
    dot = np.clip((n * z).sum(axis=1), -1.0, 1.0)
    axis = np.cross(z, n)
    q = np.concatenate([(1.0 + dot)[:, None], axis], axis=1)
    # antiparallel fallback: rotate 180 deg about X
    bad = dot < -0.9999
    if np.any(bad):
        q[bad] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    return q.astype(np.float32)
