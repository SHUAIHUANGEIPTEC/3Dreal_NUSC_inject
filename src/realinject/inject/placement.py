"""Category-aware free-slot placement for real extracted assets.

Ports ``nusc-inject/scripts/run_batch.py::_plan_one``/``plan_tracks`` and adds
placement priors per nuScenes category: vehicles go to the ego lanes ahead,
small OOD targets (dogs, strollers, wheelchairs, wheeled devices) to the curb
band. Every placement still flows through ``anchor.build_gt`` so the ground snap
and ``T_model2global`` stay the single source of truth for both channels.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

#: category prefix -> (forward_m, lateral_m) prior; matched by longest prefix
CATEGORY_PRIORS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "vehicle": ((12.0, 30.0), (-4.0, 4.0)),
    "human.pedestrian": ((6.0, 20.0), (-5.0, 5.0)),
    "movable_object": ((6.0, 20.0), (-5.0, 5.0)),
}
DEFAULT_PRIOR = ((8.0, 25.0), (-4.0, 4.0))

#: robust ground re-snap defaults (override via cfg["ground_resnap"])
DEFAULT_GROUND_RESNAP: dict = {
    "enabled": True,
    "radius_m": 6.0,      # search circle; wider than build_gt's 4 m so the
                          # road is visible even when a truck sits on it
    "band_m": 0.4,        # keep points within [seed, seed+band] as ground
    "seed_pct": 2.0,      # low-percentile seed: lands on the road, not clutter
    "min_pts": 10,
    "max_shift_m": 6.0,   # hard cap even for confident estimates; the naive
                          # percentile can be polluted several metres up by
                          # buildings/trucks, so this must stay generous
    "max_shift_weak_m": 0.3,  # cap when the ground band had to fall back
    "min_shift_m": 0.01,  # smaller deltas are noise; keep build_gt's answer
}


def prior_for(category: str | None) -> tuple[tuple[float, float], tuple[float, float]]:
    best, best_len = DEFAULT_PRIOR, -1
    for prefix, prior in CATEGORY_PRIORS.items():
        if (category or "").startswith(prefix) and len(prefix) > best_len:
            best, best_len = prior, len(prefix)
    return best


def plan_one(cfg: dict, rng: np.random.Generator, model_cfg: dict,
             overrides: dict | None = None) -> dict:
    """One object/track start: ego-frame placement around the category prior."""
    oc = dict(cfg.get("objects") or {})
    if overrides:
        oc.update(overrides)
    category = model_cfg.get("category")
    (fwd_lo, fwd_hi), (lat_lo, lat_hi) = prior_for(category)
    # config ranges override the prior when present
    fwd_lo, fwd_hi = [float(v) for v in oc.get("forward_range_m", (fwd_lo, fwd_hi))]
    lat_lo, lat_hi = [float(v) for v in oc.get("lateral_range_m", (lat_lo, lat_hi))]
    yaw_lo, yaw_hi = [float(v) for v in oc.get("yaw_in_ego_deg_range", (0.0, 360.0))]
    ground = oc.get("ground", {"search_radius_m": 4.0, "z_percentile": 15.0})

    return {
        "model_cfg": dict(model_cfg),
        "placement_cfg": {
            "strategy": "random_forward",
            "offset_in_ego": [float(rng.uniform(fwd_lo, fwd_hi)),
                              float(rng.uniform(lat_lo, lat_hi)), 0.0],
            "yaw_in_ego_deg": float(rng.uniform(yaw_lo, yaw_hi)),
            "pitch_in_ego_deg": 0.0,
            "roll_in_ego_deg": 0.0,
            "ground": ground,
        },
    }


def plan_tracks(cfg: dict, scene_name: str, n_frames: int,
                rng: np.random.Generator, records: list,
                library_root: str = "") -> list:
    """Cross-frame tracks for one scene; each track draws one asset record."""
    oc = cfg.get("objects") or {}
    tc = oc.get("tracking", {})
    n_tracks = int(tc.get("count", oc.get("count", 2)))
    lo, hi = [int(v) for v in tc.get("length", (3, 8))]
    overrides = {}
    for key in ("forward_range_m", "lateral_range_m", "yaw_in_ego_deg_range"):
        if key in tc:
            overrides[key] = tc[key]
    if not records:
        raise ValueError("no passed assets in the library to place")

    # deal records without replacement when possible so a small demo batch
    # (e.g. --assets dog,stroller) actually uses every requested asset
    if n_tracks <= len(records):
        deal = [records[int(i)] for i in rng.permutation(len(records))[:n_tracks]]
    else:
        deal = [records[int(rng.integers(0, len(records)))]
                for _ in range(n_tracks)]

    tracks = []
    for i in range(n_tracks):
        length = int(rng.integers(lo, hi + 1))
        length = max(1, min(length, n_frames))
        max_start = n_frames - length
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        record = deal[i]
        model_cfg = model_cfg_for(record, library_root)
        one = plan_one(cfg, rng, model_cfg, overrides)
        tracks.append({
            "key": f"{scene_name}_t{i:02d}",
            "start": start,
            "length": length,
            "asset_id": record.asset_id,
            "model_cfg": one["model_cfg"],
            "placement_cfg": one["placement_cfg"],
            "T": None,  # resolved at first annotated frame
            "terminated": False,
            "end_reason": None,
        })
    return tracks


def model_cfg_for(record, library_root: str) -> dict:
    """AssetRecord -> nusc-inject model_cfg (real metric size, no rescaling)."""
    from pathlib import Path

    return {
        "path": str(Path(library_root) / record.canonical_path),
        "scale": 1.0,  # real extracted assets keep their metric size
        "albedo": 0.55,
        "intensity_gain_k": 200.0,
        "category": record.category,
        "asset_id": record.asset_id,
        "kind": record.kind,
        "dims_lwh_m": list(record.dims_lwh_m),
    }


def _ground_estimate(lidar_pts: np.ndarray, x: float, y: float,
                     radius_m: float, band_m: float, seed_pct: float,
                     min_pts: int) -> tuple[float, bool]:
    """Two-stage ground estimate; returns (z, confident).

    ``confident`` is False when the ground band had too few points and we
    fell back to the whole-neighbourhood median (which may sit on clutter).
    """
    pts = np.asarray(lidar_pts, dtype=float)
    d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
    near = pts[d2 <= radius_m * radius_m]
    if len(near) < min_pts:
        near = pts
    z = near[:, 2]
    seed = float(np.percentile(z, seed_pct))
    band = z[z <= seed + band_m]
    if len(band) < min_pts:
        return float(np.median(z)), False
    return float(np.median(band)), True


def robust_ground_z(lidar_pts: np.ndarray, x: float, y: float,
                    radius_m: float = 6.0, band_m: float = 0.4,
                    seed_pct: float = 2.0, min_pts: int = 10) -> float:
    """Ground z at (x, y) robust to elevated clutter (truck bodies, barriers).

    ``anchor.build_gt`` takes a fixed percentile of the neighbourhood; when
    non-ground returns dominate that circle (placement next to a parked truck)
    the percentile lands on the truck and the object floats. Two-stage
    estimate instead: seed from the LOW tail (road is the z-min mode), then
    median of the ground band around the seed.
    """
    return _ground_estimate(lidar_pts, x, y, radius_m, band_m, seed_pct,
                            min_pts)[0]


def resnap_ground(gt: dict, lidar_pts: np.ndarray,
                  cfg: dict | None = None) -> float:
    """Re-seat a freshly placed gt on ``robust_ground_z``; returns the shift.

    Mutates ``gt`` in place (pose translation, ground_z, placement offset,
    center translation). Frozen ``from_global`` poses must NOT be passed here —
    the caller skips them so a track keeps one world pose across frames.
    Returns the applied lidar-frame z shift in meters (0 = no change).
    """
    rc = dict(DEFAULT_GROUND_RESNAP)
    rc.update((cfg or {}).get("ground_resnap") or {})
    if not rc["enabled"]:
        return 0.0

    off = gt["placement"]["offset_in_lidar_frame"]
    z_new, confident = _ground_estimate(
        lidar_pts, float(off[0]), float(off[1]),
        radius_m=float(rc["radius_m"]), band_m=float(rc["band_m"]),
        seed_pct=float(rc["seed_pct"]), min_pts=int(rc["min_pts"]))
    delta = z_new - float(gt["ground_z_lidar"])
    if abs(delta) < float(rc["min_shift_m"]):
        return 0.0
    # confident estimates may travel far (the naive percentile can be
    # polluted metres up by buildings); weak estimates only fine-tune
    cap = float(rc["max_shift_m"] if confident else rc["max_shift_weak_m"])
    if abs(delta) > cap:
        log.warning(
            "ground resnap refused for placement (%.2f, %.2f): shift %.2f m "
            "exceeds %.2f m (%s estimate); keeping build_gt ground_z=%.2f",
            float(off[0]), float(off[1]), delta, cap,
            "confident" if confident else "weak", float(gt["ground_z_lidar"]))
        return 0.0

    t_m2l = np.array(gt["T_model2lidar"], dtype=float).reshape(4, 4)
    t_m2l[2, 3] += delta
    t_l2g = np.array(gt["T_lidar2global"], dtype=float).reshape(4, 4)
    t_m2g = t_l2g @ t_m2l
    gt["T_model2lidar"] = [float(v) for v in t_m2l.reshape(-1)]
    gt["T_model2global"] = [float(v) for v in t_m2g.reshape(-1)]
    gt["ground_z_lidar"] = z_new
    off[2] = z_new
    # center translation follows the shifted pose (rotation/scale untouched)
    raw_min = np.array(gt["raw_bbox_min"], dtype=float)
    raw_size = np.array(gt["raw_bbox_size"], dtype=float)
    center = raw_min + raw_size / 2.0
    gt["translation"] = [float(v) for v in
                         (t_m2g[:3, :3] @ center + t_m2g[:3, 3])]
    return float(delta)
