"""Renderer dispatch: CPU EWA splat (default) or GPU gsplat."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..gauss.model import GaussianCloud
from .raster_cpu import RenderOut, render_cfg, render_gaussians


def get_renderer(cfg: dict | None = None) -> Callable[..., RenderOut]:
    """Return the render function selected by ``gaussian.render.backend``."""
    backend = str((cfg or {}).get("gaussian", {}).get("render", {}).get("backend", "cpu"))
    if backend == "cpu":
        return render_gaussians
    if backend == "gsplat":
        from .raster_gpu import render_gaussians_gsplat

        return render_gaussians_gsplat
    raise ValueError(f"unknown gaussian render backend: {backend!r} (cpu | gsplat)")


def render(cam: dict, cloud: GaussianCloud,
           T_model2global: np.ndarray | None = None,
           cfg: dict | None = None) -> RenderOut:
    """Rasterize ``cloud`` into ``cam`` using the configured backend."""
    return get_renderer(cfg)(cam, cloud, T_model2global, cfg)


def save_rgba_png(path, out: RenderOut) -> None:
    """Write a straight-alpha RGBA png the way nusc-inject's blender channel does."""
    import cv2

    rgba = np.dstack([
        out.rgb,
        (np.clip(out.alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
    ])
    # cv2 wants BGR(A)
    import pathlib

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


__all__ = ["RenderOut", "get_renderer", "render", "render_cfg", "save_rgba_png",
           "render_gaussians"]
