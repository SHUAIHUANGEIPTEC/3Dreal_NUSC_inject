"""GPU Gaussian splat backend (gsplat), lazy-imported.

Same contract as ``raster_cpu.render_gaussians``: (cam, cloud, T_model2global,
cfg) -> RenderOut with straight-alpha rgb / alpha / depth(inf=empty).

Local machines have no CUDA; the backend raises an actionable error so the
pipeline can fall back to ``backend: cpu`` (see configs).
"""
from __future__ import annotations

import numpy as np

from ..gauss.model import GaussianCloud
from .raster_cpu import RenderOut

_INSTALL_HINT = (
    "gsplat backend requested but unavailable. Install on a CUDA machine:\n"
    "    pip install gsplat\n"
    "or set gaussian.render.backend: cpu in your config."
)


def render_gaussians_gsplat(cam: dict, cloud: GaussianCloud,
                            T_model2global: np.ndarray | None = None,
                            cfg: dict | None = None) -> RenderOut:
    try:
        import torch
        from gsplat import rasterization
    except ImportError as exc:  # pragma: no cover - exercised only without CUDA
        raise RuntimeError(_INSTALL_HINT) from exc

    k = np.asarray(cam["K"], dtype=np.float64)
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    width, height = int(cam["width"]), int(cam["height"])
    t_g2c = np.linalg.inv(
        np.asarray(cam["T_cam2global"], dtype=np.float64).reshape(4, 4))
    t_m2c = t_g2c @ (np.eye(4) if T_model2global is None
                     else np.asarray(T_model2global, dtype=np.float64).reshape(4, 4))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    means = (np.asarray(cloud.xyz, dtype=np.float64) @ t_m2c[:3, :3].T
             + t_m2c[:3, 3])
    # gsplat expects (N,3) means, (N,4) quats (wxyz) already in camera frame
    means3 = torch.tensor(means, dtype=torch.float32, device=device)
    quats = torch.tensor(cloud.rot, dtype=torch.float32, device=device)
    scales = torch.tensor(cloud.scale, dtype=torch.float32, device=device)
    opacities = torch.tensor(cloud.opacity, dtype=torch.float32, device=device)
    colors = torch.tensor(cloud.rgb, dtype=torch.float32, device=device)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)  # camera == world frame here
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     dtype=torch.float32, device=device)

    renders, alphas, meta = rasterization(
        means3, quats, scales, opacities, colors[None], viewmat, K,
        width, height, render_mode="RGB+ED",
    )
    rgb = renders[0, ..., :3].permute(1, 2, 0).cpu().numpy()
    alpha = alphas[0, 0].cpu().numpy()
    depth = renders[0, ..., 3].cpu().numpy()
    depth = np.where(alpha > 1e-4, depth, np.inf)
    return RenderOut(
        rgb=(np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8),
        alpha=alpha.astype(np.float32),
        depth=depth.astype(np.float32),
        mask=alpha > 1e-4,
        info={"n_gauss": int(len(cloud)), "backend": "gsplat"},
    )
