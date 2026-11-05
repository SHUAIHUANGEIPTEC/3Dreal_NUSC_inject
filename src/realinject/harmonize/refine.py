"""Pluggable refinement backends.

Local (CPU) is the default; GPU backends (sd_inpaint / magicdrive) are
interface stubs — importing torch only happens inside their run(), with an
actionable error message when dependencies are missing. See
docs/GPU_DIFFUSION.md for the GPU-server workflow.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Refiner(Protocol):
    name: str

    def refine_batch(self, frames: list[dict], session) -> list[np.ndarray]:
        """frames: masks.load_frame_cam dicts (+ params filled by session).
        Returns refined RGB images aligned with frames."""
        ...


class NoneRefiner:
    """Identity: keep the raw composite (for A/B baselines)."""
    name = "none"

    def refine_batch(self, frames, session):
        return [f["comp_rgb"].copy() for f in frames]


class CPUHarmonizer:
    """LAB match -> Poisson -> noise/sharpness, with session-smoothed params."""
    name = "cpu"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def refine_batch(self, frames, session=None):
        from . import lab_match, poisson, shadow, texture
        cfg = self.cfg
        out = []
        for f in frames:
            raw = f["comp_rgb"]
            target = f["target"]
            params = f.get("params")            # smoothed, may be None
            feather = f["feather"]

            # 0) contact shadow grounds the object (2D approximation)
            if cfg.get("shadow_enabled", True):
                smap = shadow.build_footprint(
                    target, blur=int(cfg.get("shadow_blur", 15)),
                    dx_frac=float(cfg.get("shadow_dx_frac", 0.10)),
                    dy_frac=float(cfg.get("shadow_dy_frac", 0.13)))
                od, sd = f.get("obj_depth"), f.get("scene_depth")
                if od is not None and sd is not None and target.any():
                    d_obj = float(np.median(od[target]))
                    fg_mask = np.isfinite(sd) & (sd < d_obj - 0.5)
                    smap = np.where(fg_mask, 0.0, smap).astype(np.float32)
                f["shadow_map"] = smap
                raw = shadow.add_contact_shadow(
                    raw, target,
                    strength=float(cfg.get("shadow_strength", 0.55)),
                    blur=int(cfg.get("shadow_blur", 15)),
                    dx_frac=float(cfg.get("shadow_dx_frac", 0.10)),
                    dy_frac=float(cfg.get("shadow_dy_frac", 0.13)))

            # 1) exposure/lightness match of the object onto its surroundings
            #    (identity-preserving: default L channel only)
            img = (lab_match.apply_params(raw, params, feather)
                   if params is not None else raw.copy())

            # 2a) optional full-silhouette Poisson clone — OFF by default:
            #     re-anchoring to the raw object edge cancels the exposure
            #     correction (measured: dL went +26.5 -> +28.6 with it on)
            if cfg.get("poisson_enabled", False):
                img = poisson.blend(img, raw, target,
                                    pad=int(cfg.get("poisson_pad", 32)),
                                    enabled=True)
            # 2b) edge-band fringe smoothing (default on)
            if cfg.get("edge_band_enabled", True):
                img = texture.smooth_edge_band(
                    img, target, band_px=int(cfg.get("edge_band_px", 4)))

            if cfg.get("noise_enabled", True):
                sigma = f.get("noise_sigma", 0.0)
                img = texture.add_noise(img, target, sigma,
                                        seed=int(f.get("seed", 0)))
            if cfg.get("unsharp_amount", 0) > 0:
                img = texture.unsharp(img, target,
                                      amount=float(cfg["unsharp_amount"]))
            out.append(img)
        return out


class SDInpaintRefiner:
    """GPU: Stable-Diffusion inpaint over the dilated target mask."""
    name = "sd_inpaint"

    def __init__(self, *a, **kw):
        import importlib

        try:
            importlib.import_module("diffusers")
            importlib.import_module("torch")
        except ImportError as e:
            raise RuntimeError(
                "sd_inpaint requires a GPU environment: pip install torch "
                "diffusers transformers accelerate — see docs/GPU_DIFFUSION.md"
            ) from e
        raise NotImplementedError("wire up on the GPU server (Phase 4 stub)")

    def refine_batch(self, frames, session):  # pragma: no cover
        raise NotImplementedError


class MagicDriveRefiner:
    """GPU: box-conditioned generation (bbox_add mechanism, Stag-main)."""
    name = "magicdrive"

    def __init__(self, *a, **kw):
        raise RuntimeError(
            "magicdrive requires the Stag-main/MagicDrive stack on a GPU "
            "server (missing lvdm / magicdrive.networks) — see "
            "docs/GPU_DIFFUSION.md")


def get_refiner(name: str, cfg: dict | None = None, **kw) -> Refiner:
    name = (name or "cpu").lower()
    if name == "none":
        return NoneRefiner()
    if name == "cpu":
        return CPUHarmonizer(cfg=cfg)
    if name == "sd_inpaint":
        return SDInpaintRefiner(cfg=cfg, **kw)
    if name == "magicdrive":
        return MagicDriveRefiner(cfg=cfg, **kw)
    raise ValueError(f"unknown refiner {name!r} "
                     f"(choose none|cpu|sd_inpaint|magicdrive)")
