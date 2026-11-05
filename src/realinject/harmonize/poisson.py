"""Poisson (gradient-domain) blending of the target region, with fallback.

cv2.seamlessClone keeps the object's gradients while adopting the local
background's low frequencies — removes the hard composite seam. The crop
window is clamped inside the image first (seamlessClone rejects windows
crossing the border); any residual cv2 error falls back to a feathered
bilateral-smoothed seam band (no exception escapes).
"""
from __future__ import annotations

import cv2
import numpy as np


def _bbox(mask: np.ndarray, pad: int):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, h)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, w)
    if y1 - y0 < 4 or x1 - x0 < 4:
        return None
    return x0, y0, x1, y1


def blend(src_rgb: np.ndarray, dst_rgb: np.ndarray, target: np.ndarray,
          pad: int = 32, enabled: bool = True) -> np.ndarray:
    """Poisson-clone target pixels of `src_rgb` into `dst_rgb`.

    src and dst MUST differ (typically src = colour-corrected frame,
    dst = raw composite): cloning an image into itself is a no-op — the
    Poisson solution reproduces src exactly when boundaries already match.
    The result equals dst ON the mask boundary (seam continuity by
    construction) and carries src's gradients inside.
    """
    if not enabled or not target.any():
        return dst_rgb
    box = _bbox(target.astype(bool), pad)
    if box is None:
        return dst_rgb
    x0, y0, x1, y1 = box
    src = src_rgb[y0:y1, x0:x1].copy()
    mask = (target[y0:y1, x0:x1].astype(np.uint8)) * 255
    if mask.sum() == 0:
        return dst_rgb
    center = ((x0 + x1) // 2, (y0 + y1) // 2)
    try:
        out = cv2.seamlessClone(src, dst_rgb.copy(), mask, center,
                                cv2.NORMAL_CLONE)
        return out
    except cv2.error:
        return _fallback_feather(dst_rgb, target.astype(np.float32))


def _fallback_feather(img_rgb: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Edge-band bilateral smoothing where the seam would be."""
    band = cv2.dilate(target.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    band = band.astype(bool) & ~target.astype(bool)
    if not band.any():
        return img_rgb
    smooth = cv2.bilateralFilter(img_rgb, 9, 30, 30)
    out = img_rgb.copy()
    out[band] = smooth[band]
    return out
