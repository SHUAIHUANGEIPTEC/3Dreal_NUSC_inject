"""Cloud -> LiDAR-channel payload adapter.

``nusc_inject.lidar_channel.inject`` consumes per-object payloads of
``{"pts_raw": (M,3), "normals_raw": (M,3), "gt": dict}`` in the **model frame**
(it applies ``gt["T_model2lidar"]`` itself). Gaussian clouds carry their
canonical-frame centers in ``xyz``; normals come from the mesh sampling
(``extra["normal"]``) or are estimated by kNN PCA for point assets and oriented
away from the centroid (outward), which is what the channel's back-face drop
expects.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..gauss.model import GaussianCloud

DEFAULTS = {
    "n_dense": 60_000,
    "normal_knn": 16,
    "seed": 0,
}


def estimate_normals(xyz: np.ndarray, k: int = 16) -> np.ndarray:
    """Outward unit normals from local PCA (smallest-eigenvalue direction)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    k = int(max(3, min(k, n)))
    _, idx = cKDTree(xyz).query(xyz, k=k)
    neigh = xyz[idx]  # (N, k, 3)
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    normals = evecs[:, :, 0]
    # orient outward: away from the cloud centroid
    centroid = xyz.mean(axis=0)
    flip = np.sum((xyz - centroid) * normals, axis=1) < 0
    normals[flip] *= -1.0
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return normals / norms


def sample_indices(n: int, n_dense: int, seed: int = 0) -> np.ndarray:
    """Deterministic uniform subset of ``[0, n)``."""
    if n <= n_dense:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=n_dense, replace=False))


def cloud_to_lidar_payload(cloud: GaussianCloud, gt: dict | None = None,
                           cfg: dict | None = None) -> dict:
    """Build the payload ``lidar_channel.inject`` expects for one object."""
    p = dict(DEFAULTS)
    p.update((cfg or {}).get("gaussian", {}).get("lidar") or {})
    sel = sample_indices(len(cloud), int(p["n_dense"]), int(p["seed"]))
    pts = np.asarray(cloud.xyz, dtype=np.float64)[sel]

    normals_src = cloud.extra.get("normal")
    if normals_src is not None and len(normals_src) == len(cloud):
        normals = np.asarray(normals_src, dtype=np.float64)[sel]
    else:
        normals = estimate_normals(pts, int(p["normal_knn"]))

    payload = {"pts_raw": pts, "normals_raw": normals}
    if gt is not None:
        payload["gt"] = gt
    return payload
