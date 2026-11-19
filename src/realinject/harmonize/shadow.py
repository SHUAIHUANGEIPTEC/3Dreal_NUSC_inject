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
    """Soft contact-ellipse map (0..1) at the silhouette's bottom-centre.

    Radial quadratic falloff ``(1 - r^2)^2`` instead of a flat-filled
    ellipse: the flat version keeps full strength out to the ellipse rim
    and relies on a small Gaussian blur for softness, which at our object
    sizes (semi-axes of 100-300 px vs blur 15 px) reads as a hard-edged
    black pancake. The falloff peaks at the contact point and reaches zero
    smoothly at the rim, so no blur is load-bearing for edge quality.
    """
    if not target.any():
        return np.zeros(target.shape, np.float32)
    t = target.astype(np.uint8)
    ys, xs = np.nonzero(t)
    obj_w = int(xs.max() - xs.min()) + 1
    obj_h = int(ys.max() - ys.min()) + 1

    # ellipse centred at the silhouette's bottom-centre, nudged away from light
    cx = float((xs.min() + xs.max()) / 2 + dx_frac * obj_w)
    cy = float(ys.max() + dy_frac * obj_h)
    semi_w = max(0.62 * obj_w, 4.0)
    semi_h = max(0.13 * obj_w, 3.0)   # flat: seen at a shallow angle

    yy = np.arange(target.shape[0], dtype=np.float32)[:, None]
    xx = np.arange(target.shape[1], dtype=np.float32)[None, :]
    r2 = ((xx - cx) / semi_w) ** 2 + ((yy - cy) / semi_h) ** 2
    footprint = np.clip(1.0 - r2, 0.0, 1.0) ** 2   # C1 at the rim

    if int(blur) > 0:
        k = max(int(blur) | 1, 3)
        footprint = cv2.GaussianBlur(footprint, (k, k), 0)
    peak = float(footprint.max())
    if peak > 1e-6:
        footprint = footprint / peak        # normalise BEFORE masking the
    footprint[target] = 0.0                 # object (bool mask!), so contact
    return footprint                        # strength survives the mask-out


def foreground_gate(footprint: np.ndarray, target: np.ndarray,
                    obj_depth: np.ndarray | None,
                    scene_depth: np.ndarray | None,
                    fg_factor: float = 0.30) -> np.ndarray:
    """Zero the footprint where the scene is a NEAR foreground occluder.

    The naive rule ``scene_depth < median(object_depth) - eps`` is wrong for
    a contact shadow: the ground under the object is in the image BELOW the
    silhouette, so it is always nearer than the object body (measured: it
    killed ~90% of the footprint — the shadow vanished off the road while
    the metrics still excluded a phantom shadow region). Ground must be
    kept; only scene returns much closer than the contact distance
    (``fg_factor * d_contact``, default 0.3 — a real occluder between
    camera and target, not the road) are treated as foreground.
    """
    if footprint is None or target is None or not target.any():
        return footprint
    if obj_depth is None or scene_depth is None:
        return footprint

    t = target.astype(bool)
    ys = np.flatnonzero(t.any(axis=1))
    y0, y1 = int(ys.min()), int(ys.max())
    band = np.zeros_like(t)
    band[max(y1 - max(1, int(0.15 * (y1 - y0 + 1))), y0):y1 + 1] = \
        t[max(y1 - max(1, int(0.15 * (y1 - y0 + 1))), y0):y1 + 1]
    vals = np.asarray(obj_depth)[band]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return footprint
    d_contact = float(np.median(vals))

    sd = np.asarray(scene_depth, dtype=float)
    fg = np.isfinite(sd) & (sd < float(fg_factor) * d_contact)
    return np.where(fg, 0.0, footprint).astype(np.float32)


def add_contact_shadow(img_rgb: np.ndarray, target: np.ndarray,
                       strength: float = 0.40, blur: int = 15,
                       dx_frac: float = 0.10, dy_frac: float = 0.10,
                       footprint: np.ndarray | None = None,
                       ) -> np.ndarray:
    """Darken the background under `target` with a soft contact ellipse.

    Pass a prebuilt (and optionally gated) ``footprint`` so the applied
    shadow and the map reported to the metrics stay identical.
    """
    if not target.any() or strength <= 0:
        return img_rgb
    if footprint is None:
        footprint = build_footprint(target, blur, dx_frac, dy_frac)
    out = img_rgb.astype(np.float32) * \
        (1.0 - float(strength) * footprint[..., None])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
