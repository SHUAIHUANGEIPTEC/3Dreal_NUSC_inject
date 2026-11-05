"""LAB statistical colour matching: object region -> local background ring.

Work entirely in OpenCV's uint8 LAB encoding (L 0..255, a/b 128-centred) on a
float32 copy — gain/bias are affine, so the encoding scale cancels out, and
the inverse stays a plain LAB2RGB of a clipped uint8.
"""
from __future__ import annotations

import cv2
import numpy as np

_EPS = 1e-3
_SIGMA_CLIP = (0.7, 1.5)


def _smooth(series: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge handling (min 1 sample)."""
    s = np.asarray(series, dtype=float)
    if len(s) < 2 or window <= 1:
        return s.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.concatenate([np.full(pad, s[0]), s, np.full(pad, s[-1])])
    return np.convolve(padded, kernel, mode="valid")[: len(s)]


def compute_params(img_rgb: np.ndarray, target: np.ndarray,
                   ring: np.ndarray,
                   sigma_clip: tuple = _SIGMA_CLIP,
                   channels: tuple = ("L",)) -> dict | None:
    """Per-channel gain/bias mapping target stats onto ring stats.

    x' = (x - mu_obj) * (sigma_bg / sigma_obj) + mu_bg
       = gain * x + bias,  gain clipped to sigma_clip.

    Default channels = ("L",) only: matching CHROMA means would repaint a
    blue object gray (identity loss) — only exposure/lightness is a
    compositing artifact. Returns {"L": (g,b), ...} or None if too small.
    """
    if target.sum() < 64 or ring.sum() < 64:
        return None
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = {}
    for i, ch in enumerate(("L", "a", "b")):
        if ch not in channels:
            continue
        xo = lab[:, :, i][target]
        xb = lab[:, :, i][ring]
        mu_o, sd_o = float(xo.mean()), float(xo.std())
        mu_b, sd_b = float(xb.mean()), float(xb.std())
        gain = float(np.clip(sd_b / max(sd_o, _EPS), sigma_clip[0], sigma_clip[1]))
        bias = mu_b - gain * mu_o
        out[ch] = (gain, bias)
    return out or None


def apply_params(img_rgb: np.ndarray, params: dict,
                 mask: np.ndarray | None) -> np.ndarray:
    """Apply gain/bias for the channels present in `params` (others kept);
    `mask` (H,W) float 0..1 feathers the edit (None = apply everywhere).

    Mixing happens in LAB space; the LAB->RGB conversion runs exactly once,
    on the final array (mixing LAB-encoded values into RGB is a bug this
    function used to have — it produced garbage colours under a mask).
    Returns uint8 RGB.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    matched = lab.copy()
    for i, ch in enumerate(("L", "a", "b")):
        if ch not in params:
            continue
        g, b = params[ch]
        matched[:, :, i] = g * lab[:, :, i] + b
    if mask is not None:
        m = mask[:, :, None].astype(np.float32)
        matched = lab * (1.0 - m) + matched * m
    matched_u8 = np.clip(np.round(matched), 0, 255).astype(np.uint8)
    return cv2.cvtColor(matched_u8, cv2.COLOR_LAB2RGB)


def smooth_params(per_frame: list[dict | None],
                  window: int) -> list[dict | None]:
    """Temporal smoothing of gain/bias series (None gaps stay None).

    per_frame[i] is the params dict of frame i (or None when unavailable);
    smoothing only interpolates over frames where params exist. Channels
    are taken from whatever the first available params contain.
    """
    idx = [i for i, p in enumerate(per_frame) if p is not None]
    if not idx:
        return list(per_frame)
    channels = tuple(per_frame[idx[0]].keys())
    smooth = [None] * len(per_frame)
    for ch in channels:
        gains = np.array([per_frame[i][ch][0] for i in idx], dtype=float)
        biases = np.array([per_frame[i][ch][1] for i in idx], dtype=float)
        gs, bs = _smooth(gains, window), _smooth(biases, window)
        for j, i in enumerate(idx):
            if smooth[i] is None:
                smooth[i] = {}
            smooth[i][ch] = (float(gs[j]), float(bs[j]))
    return smooth
