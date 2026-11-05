"""Objective metrics: colour distance to background + temporal correction stability."""
from __future__ import annotations

import cv2
import numpy as np


def mean_delta_e(img_a: np.ndarray, img_b: np.ndarray,
                 mask: np.ndarray) -> float:
    """Mean CIE76 dE between two RGB images inside mask."""
    if mask.sum() == 0:
        return float("nan")
    la = cv2.cvtColor(img_a, cv2.COLOR_RGB2LAB).astype(np.float32)
    lb = cv2.cvtColor(img_b, cv2.COLOR_RGB2LAB).astype(np.float32)
    d = np.linalg.norm(la - lb, axis=2)
    return float(d[mask].mean())


def region_mean_color(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(3,) mean RGB of the masked region."""
    if mask.sum() == 0:
        return np.zeros(3)
    return img[mask].mean(axis=0).astype(float)


def target_vs_ring_de(img: np.ndarray, target: np.ndarray,
                      ring: np.ndarray) -> float:
    """Mean dE between the target region and its background ring.

    NOTE: for a strongly-coloured object this includes chroma identity
    (a blue truck vs gray road never reaches 0) — use lightness_chroma_gap
    to separate the compositing artifact (exposure) from identity (chroma).
    """
    if target.sum() == 0 or ring.sum() == 0:
        return float("nan")
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    mu_t = lab[target].mean(axis=0)
    mu_r = lab[ring].mean(axis=0)
    return float(np.linalg.norm(mu_t - mu_r))


def lightness_chroma_gap(img: np.ndarray, target: np.ndarray,
                         ring: np.ndarray) -> tuple[float, float]:
    """(dL, chroma_gap): exposure mismatch (target L - ring L, signed) and
    chroma distance in the a/b plane (identity — should be PRESERVED)."""
    if target.sum() == 0 or ring.sum() == 0:
        return float("nan"), float("nan")
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    mu_t = lab[target].mean(axis=0)
    mu_r = lab[ring].mean(axis=0)
    d_l = float(mu_t[0] - mu_r[0])
    chroma = float(np.linalg.norm(mu_t[1:] - mu_r[1:]))
    return d_l, chroma


def correction_flicker(raws: list[np.ndarray],
                       refineds: list[np.ndarray],
                       masks: list[np.ndarray]) -> float:
    """Mean |Δc_t - Δc_{t-1}| where c_t = target mean colour and
    Δ = refined - raw. A stable correction -> low value (no shimmer)."""
    deltas = []
    for raw, ref, m in zip(raws, refineds, masks, strict=True):
        if m.sum() == 0:
            deltas.append(None)
            continue
        deltas.append(region_mean_color(ref, m) - region_mean_color(raw, m))
    seq = [d for d in deltas if d is not None]
    if len(seq) < 2:
        return 0.0
    diffs = [np.abs(seq[i] - seq[i - 1]).mean() for i in range(1, len(seq))]
    return float(np.mean(diffs))
