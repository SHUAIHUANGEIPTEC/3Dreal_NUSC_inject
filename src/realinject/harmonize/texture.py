"""Noise / sharpness matching so the render stops looking 'too clean'."""
from __future__ import annotations

import cv2
import numpy as np


def _highpass(img: np.ndarray) -> np.ndarray:
    """Detail signal: image minus a 3x3 median (robust to structure)."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    med = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 3)
    return g - med.astype(np.float32)


def estimate_noise_sigma(img: np.ndarray, region: np.ndarray) -> float:
    """Std of the high-pass signal inside `region` (background noise level)."""
    if region.sum() < 64:
        return 0.0
    return float(_highpass(img)[region].std())


def needed_noise(img: np.ndarray, target: np.ndarray,
                 ring: np.ndarray) -> float:
    """Sigma of Gaussian noise to ADD on the target so its noise level
    approaches the background's (never denoises an already-noisier object)."""
    s_bg = estimate_noise_sigma(img, ring)
    s_obj = estimate_noise_sigma(img, target)
    var = s_bg ** 2 - s_obj ** 2
    return float(np.sqrt(var)) if var > 0 else 0.0


def add_noise(img: np.ndarray, target: np.ndarray, sigma: float,
              seed: int) -> np.ndarray:
    if sigma <= 1e-6 or not target.any():
        return img
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=img.shape[:2]).astype(np.float32)
    out = img.astype(np.float32)
    out[target] += noise[target][:, None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def unsharp(img: np.ndarray, target: np.ndarray, amount: float = 0.3,
            ksize: int = 3) -> np.ndarray:
    if amount <= 0 or not target.any():
        return img
    blur = cv2.GaussianBlur(img, (ksize, ksize), 0)
    out = img.astype(np.float32) + amount * (img.astype(np.float32) -
                                             blur.astype(np.float32))
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    res = img.copy()
    res[target] = out[target]
    return res


def smooth_edge_band(img: np.ndarray, target: np.ndarray,
                     band_px: int = 4) -> np.ndarray:
    """Bilateral-smooth a narrow band around the object silhouette.

    Softens the alpha fringe (semi-transparent edge mixtures) without
    touching the object interior or the background beyond the band.
    """
    if not target.any() or band_px <= 0:
        return img
    t = target.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * band_px + 1, 2 * band_px + 1))
    band = (cv2.dilate(t, k) & ~cv2.erode(t, k)).astype(bool)
    if not band.any():
        return img
    smooth = cv2.bilateralFilter(img, 9, 40, 40)
    out = img.copy()
    out[band] = smooth[band]
    return out
