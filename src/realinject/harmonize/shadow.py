"""Contact shadow for composited objects (2D approximation, CPU).

The Blender pass renders only the object (no shadow catcher), so injected
objects float. This darkens the background with a soft ground ellipse at the
object's contact line (bottom-centre of its silhouette, nudged toward the
light-away direction) and never touches the object's own pixels.

Honest scope: a 2D approximation, not a physically traced shadow — kills the
floating look; a true cast shadow needs a Cycles shadow-catcher pass.
"""
from __future__ import annotations

import cv2
import numpy as np


def build_footprint(target: np.ndarray, blur: int = 15,
                    dx_frac: float = 0.10, dy_frac: float = 0.10,
                    ) -> np.ndarray:
    """Soft contact-ellipse map (0..1) at the silhouette's bottom-centre."""
    if not target.any():
        return np.zeros(target.shape, np.float32)
    t = target.astype(np.uint8)
    ys, xs = np.nonzero(t)
    obj_w = int(xs.max() - xs.min()) + 1
    obj_h = int(ys.max() - ys.min()) + 1

    # ellipse centred at the silhouette's bottom-centre, nudged away from light
    cx = int(round((xs.min() + xs.max()) / 2 + dx_frac * obj_w))
    cy = int(round(ys.max() + dy_frac * obj_h))
    semi_w = max(int(round(0.62 * obj_w)), 4)
    semi_h = max(int(round(0.13 * obj_w)), 3)   # flat: seen at a shallow angle

    footprint = np.zeros(t.shape, np.float32)
    cv2.ellipse(footprint, (cx, cy), (semi_w, semi_h), 0, 0, 360, 1.0, -1)
    k = max(int(blur) | 1, 3)
    shadow = cv2.GaussianBlur(footprint, (k, k), 0)
    shadow[target] = 0.0                        # never darken the object itself
    peak = float(shadow.max())
    if peak > 1e-6:
        shadow = shadow / peak                  # blur dims it; restore
    return shadow


def add_contact_shadow(img_rgb: np.ndarray, target: np.ndarray,
                       strength: float = 0.55, blur: int = 15,
                       dx_frac: float = 0.10, dy_frac: float = 0.10,
                       ) -> np.ndarray:
    """Darken the background under `target` with a soft contact ellipse."""
    if not target.any() or strength <= 0:
        return img_rgb
    shadow = build_footprint(target, blur, dx_frac, dy_frac)
    out = img_rgb.astype(np.float32) * (1.0 - strength * shadow[..., None])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
