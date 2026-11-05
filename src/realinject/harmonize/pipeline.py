"""Two-pass, per-camera temporal harmonization over an injection batch.

Pass A — scan frames (time order), build regions, compute per-frame LAB
         gain/bias + noise sigma; smooth each series (moving average) so the
         correction does not shimmer frame to frame.
Pass B — run the configured Refiner with the smoothed params, save
         refined/<cam>.jpg beside the composite, collect metrics.

Smoothing key is (scene, cam): both injected objects come from the same
model/lighting in one batch, and the composite alpha covers them jointly —
per-(scene,cam) series is the granularity the artifacts actually support.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from . import lab_match, masks, metrics, texture


def _frame_sort(frame_dir: Path) -> int:
    m = re.match(r"f(\d+)", frame_dir.name)
    return int(m.group(1)) if m else -1


def collect_cam_frames(batch_dir: Path, scene_name: str, cam: str,
                       cfg: dict) -> list[dict]:
    """All composited frames of one camera, time-ordered, with regions."""
    scene_dir = Path(batch_dir) / scene_name
    alpha_thr = float(cfg.get("alpha_threshold", 0.05))
    margin = float(cfg.get("depth_margin_m", 0.5))
    r_in = int(cfg.get("ring_inner_px", 2))
    r_out = int(cfg.get("ring_outer_px", 12))
    frames = []
    for gt_path in sorted(scene_dir.glob("f*/gt.json"),
                          key=lambda p: _frame_sort(p.parent)):
        fd = gt_path.parent
        rec = masks.load_frame_cam(fd, cam, alpha_thr, margin)
        if rec is None or not rec["target"].any():
            continue
        rec["ring"] = masks.ring_region(rec["target"], r_in, r_out)
        rec["feather"] = masks.feather_mask(rec["alpha"], rec["visible"])
        rec["frame_idx"] = _frame_sort(fd)
        rec["seed"] = hash((scene_name, cam, fd.name)) % (2 ** 31)
        frames.append(rec)
    return frames


def run_session(batch_dir: Path, scene_name: str, cams: list[str],
                cfg: dict, refiner) -> dict:
    """Two passes over every camera; writes refined/ + returns a report."""
    batch_dir = Path(batch_dir)
    window = int(cfg.get("smooth_window", 5))
    report = {"scenes": {}, "refiner": getattr(refiner, "name", "?")}

    for cam in cams:
        frames = collect_cam_frames(batch_dir, scene_name, cam, cfg)
        if not frames:
            continue

        # ---- pass A: params + noise, then temporal smoothing ----
        lab_channels = tuple(cfg.get("lab_channels", ["L"]))
        raw_params = []
        for f in frames:
            p = lab_match.compute_params(f["comp_rgb"], f["target"],
                                         f["ring"],
                                         sigma_clip=tuple(
                                             cfg.get("sigma_clip",
                                                     [0.7, 1.5])),
                                         channels=lab_channels)
            raw_params.append(p)
            f["noise_sigma"] = texture.needed_noise(
                f["comp_rgb"], f["target"], f["ring"])
        smoothed = lab_match.smooth_params(raw_params, window)
        for f, p in zip(frames, smoothed, strict=True):
            f["params"] = p            # None -> refiner skips LAB for that frame

        # ---- pass B: refine ----
        refined_list = refiner.refine_batch(frames, session=None)

        # ---- persist + metrics ----
        raws, refineds, tgts, rings = [], [], [], []
        de_before, de_after = [], []
        dl_before, dl_after, chroma_before, chroma_after = [], [], [], []
        for f, ref in zip(frames, refined_list, strict=True):
            out_dir = f["frame_dir"] / "refined"
            out_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(out_dir / f"{cam}.jpg"),
                        cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
            raws.append(f["comp_rgb"])
            refineds.append(ref)
            tgts.append(f["target"])
            # ring stats exclude the contact-shadow footprint (it darkens the
            # ground by design; counting it would fake a colour gap)
            ring_eff = f["ring"]
            smap = f.get("shadow_map")
            if smap is not None:
                ring_eff = ring_eff & (smap < 0.05)
            rings.append(ring_eff)
            de_before.append(metrics.target_vs_ring_de(
                f["comp_rgb"], f["target"], ring_eff))
            de_after.append(metrics.target_vs_ring_de(
                ref, f["target"], ring_eff))
            lb = metrics.lightness_chroma_gap(
                f["comp_rgb"], f["target"], ring_eff)
            la = metrics.lightness_chroma_gap(ref, f["target"], ring_eff)
            dl_before.append(lb[0])
            dl_after.append(la[0])
            chroma_before.append(lb[1])
            chroma_after.append(la[1])

        flicker_ref = metrics.correction_flicker(raws, refineds, tgts)
        # unsmoothed-params baseline for the anti-flicker claim
        from .refine import CPUHarmonizer
        uns_frames = []
        for f, p0 in zip(frames, raw_params, strict=True):
            g = dict(f)
            g["params"] = p0
            uns_frames.append(g)
        uns = CPUHarmonizer(cfg=dict(cfg)).refine_batch(uns_frames)
        flicker_unsm = metrics.correction_flicker(raws, uns, tgts)

        report["scenes"].setdefault(scene_name, {})[cam] = {
            "n_frames": len(frames),
            "dE_target_ring_before": _mean(de_before),
            "dE_target_ring_after": _mean(de_after),
            "dE_improvement_pct": _pct(de_before, de_after),
            "lightness_gap_before": _mean(dl_before),
            "lightness_gap_after": _mean(dl_after),
            "lightness_improvement_pct": _pct(dl_before, dl_after),
            "chroma_gap_before": _mean(chroma_before),
            "chroma_gap_after": _mean(chroma_after),
            "chroma_identity_kept_pct": _kept(chroma_before, chroma_after),
            "correction_flicker_smoothed": flicker_ref,
            "correction_flicker_unsmoothed": flicker_unsm,
            "flicker_reduction_pct": _pct([flicker_unsm], [flicker_ref]),
            "n_params_smoothed": sum(p is not None for p in smoothed),
        }

    report_path = batch_dir / "harmonize_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def _mean(xs) -> float:
    xs = [x for x in xs if x == x]  # drop NaN
    return float(np.mean(xs)) if xs else float("nan")


def _pct(before, after) -> float:
    b, a = _mean(before), _mean(after)
    if b != b or a != a or b == 0:
        return float("nan")
    return float(100.0 * (b - a) / b)


def _kept(before, after) -> float:
    """Percent of the original magnitude retained (chroma identity guard)."""
    b, a = abs(_mean(before)), abs(_mean(after))
    if b != b or a != a or b == 0:
        return float("nan")
    return float(100.0 * a / b)
