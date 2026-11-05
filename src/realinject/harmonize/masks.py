"""Reconstruct composite inputs (alpha / visible / regions) from batch artifacts.

Mirrors ``nusc_inject.composite`` exactly (same thresholds), without modifying
the upstream repo: alpha comes from the **Gaussian renderer's** RGBA png
(``gauss/<cam>.png`` — Blender's ``blender/`` is accepted as a fallback for
A/B batches), object/scene depth from the ``depth/*.npy`` files
``pipeline_gauss.run_sample`` persists.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_frame_cam(frame_dir: Path, cam: str,
                   alpha_thr: float = 0.05,
                   depth_margin: float = 0.5) -> dict | None:
    """Load one camera's artifacts. Returns None when not composited."""
    # prefer the occlusion-corrected composite/depth when available
    comp = frame_dir / "composite_fixed" / f"{cam}.jpg"
    if not comp.exists():
        comp = frame_dir / "composite" / f"{cam}.jpg"
    if not comp.exists():
        return None
    # gaussian frames first (this project), blender RGBA as A/B fallback
    png = frame_dir / "gauss" / f"{cam}.png"
    if not png.exists():
        png = frame_dir / "blender" / f"{cam}.png"
    obj_d = frame_dir / "depth_fixed" / f"{cam}_obj_depth.npy"
    scn_d = frame_dir / "depth_fixed" / f"{cam}_scene_depth.npy"
    if not obj_d.exists():
        obj_d = frame_dir / "depth" / f"{cam}_obj_depth.npy"
        scn_d = frame_dir / "depth" / f"{cam}_scene_depth.npy"
    if not (png.exists() and obj_d.exists() and scn_d.exists()):
        return None

    comp_rgb = cv2.cvtColor(cv2.imread(str(comp)), cv2.COLOR_BGR2RGB)
    fg = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if fg is None or fg.ndim != 3 or fg.shape[2] != 4:
        return None
    alpha = fg[:, :, 3].astype(np.float32) / 255.0
    obj_depth = np.load(obj_d)
    scene_depth = np.load(scn_d)
    # composite rule (same arrays, same thresholds)
    visible = (alpha > float(alpha_thr)) & \
              (obj_depth <= scene_depth + float(depth_margin))
    return {
        "cam": cam,
        "comp_rgb": comp_rgb,
        "alpha": alpha,
        "visible": visible,
        "target": visible,          # harmonize acts on visible object pixels
        "obj_depth": obj_depth,
        "scene_depth": scene_depth,
        "frame_dir": Path(frame_dir),
    }


def ring_region(target: np.ndarray, r_in: int = 2,
                r_out: int = 12) -> np.ndarray:
    """Background ring around the target: dilate(r_out) minus dilate(r_in)."""
    tgt = target.astype(np.uint8)

    def dil(r: int) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(tgt, k).astype(bool)
    return dil(r_out) & ~dil(r_in)


def feather_mask(alpha: np.ndarray, visible: np.ndarray,
                 blur: int = 5) -> np.ndarray:
    """Soft 0..1 mask: alpha restricted to visible pixels, Gaussian-feathered."""
    m = (alpha * visible.astype(np.float32))
    if blur > 0:
        m = cv2.GaussianBlur(m, (blur, blur), 0)
    return np.clip(m, 0.0, 1.0)
