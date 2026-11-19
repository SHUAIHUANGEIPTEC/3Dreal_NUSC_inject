"""GPU Gaussian splat backend (gsplat), lazy-imported.

Same contract as ``raster_cpu.render_gaussians``: (cam, cloud, T_model2global,
cfg) -> RenderOut with straight-alpha rgb / alpha / depth(inf=empty).

gsplat performs exact front-to-back transmittance blending on the GPU, so it
does not need the CPU renderer's soft-z approximation; the shared quality
stages are reused so both backends agree on output semantics:

- supersampling (``ssaa``): rasterize at ``ssaa x`` resolution and
  alpha-weighted downsample (unpremultiplied rgb / depth reconstructed from
  the sub-samples);
- edge-preserving cleanup (``smooth_rgb`` / ``min_blob_px``) and the final
  contract enforcement (``mask == alpha > depth_eps``).

Install on a CUDA machine with ``pip install gsplat``; set
``gaussian.render.backend: gsplat`` in the config (default is ``cpu``).
"""
from __future__ import annotations

import numpy as np

from ..gauss.model import GaussianCloud
from .raster_cpu import (
    RenderOut,
    _downsample,
    _finalize,
    _postprocess,
    render_cfg,
)

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

    p = render_cfg(cfg)
    s = max(1, int(p["ssaa"]))
    if s > 1:
        cam_hi = dict(cam)
        k = np.asarray(cam["K"], dtype=np.float64)
        k_hi = k.copy()
        k_hi[0, 0] *= s
        k_hi[1, 1] *= s
        k_hi[0, 2] *= s
        k_hi[1, 2] *= s
        cam_hi["K"] = k_hi
        cam_hi["width"] = int(cam["width"]) * s
        cam_hi["height"] = int(cam["height"]) * s
    else:
        cam_hi = cam

    fx = float(np.asarray(cam_hi["K"], dtype=np.float64)[0, 0])
    fy = float(np.asarray(cam_hi["K"], dtype=np.float64)[1, 1])
    cx = float(np.asarray(cam_hi["K"], dtype=np.float64)[0, 2])
    cy = float(np.asarray(cam_hi["K"], dtype=np.float64)[1, 2])
    width, height = int(cam_hi["width"]), int(cam_hi["height"])
    t_g2c = np.linalg.inv(
        np.asarray(cam_hi["T_cam2global"], dtype=np.float64).reshape(4, 4))
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
    # gsplat>=1.x expects batched (C,4,4) viewmats and (C,3,3) intrinsics
    viewmat = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)  # cam == world here
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     dtype=torch.float32, device=device).unsqueeze(0)

    renders, alphas, meta = rasterization(
        means3, quats, scales, opacities, colors[None], viewmat, K,
        width, height, render_mode="RGB+ED",
    )
    # gsplat returns (C,H,W,4) renders and (C,H,W,1) alphas
    rgb = renders[0, ..., :3].cpu().numpy().astype(np.float64)
    alpha = alphas[0, ..., 0].cpu().numpy().astype(np.float64)
    depth = renders[0, ..., 3].cpu().numpy().astype(np.float64)
    # gsplat RGB is premultiplied accumulation; the contract wants straight alpha
    col = rgb / np.maximum(alpha, 1e-9)[..., None]
    # guard float boundary cases (0/0 on empty pixels) before the uint8 cast
    col = np.nan_to_num(col, nan=0.0, posinf=1.0, neginf=0.0)
    depth = np.where(alpha > 1e-9, depth, np.inf)

    col, alpha, depth = _downsample(col, alpha, depth, s, p)
    col, alpha, depth = _postprocess(col, alpha, depth, p)
    rgb8, alpha, depth, mask = _finalize(col, alpha, depth, p)

    return RenderOut(
        rgb=rgb8,
        alpha=alpha,
        depth=depth,
        mask=mask,
        info={"n_gauss": int(len(cloud)), "backend": "gsplat", "ssaa": s,
              "n_px": int(mask.sum())},
    )
