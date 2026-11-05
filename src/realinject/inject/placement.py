"""Category-aware free-slot placement for real extracted assets.

Ports ``nusc-inject/scripts/run_batch.py::_plan_one``/``plan_tracks`` and adds
placement priors per nuScenes category: vehicles go to the ego lanes ahead,
small OOD targets (dogs, strollers, wheelchairs, wheeled devices) to the curb
band. Every placement still flows through ``anchor.build_gt`` so the ground snap
and ``T_model2global`` stay the single source of truth for both channels.
"""
from __future__ import annotations

import numpy as np

#: category prefix -> (forward_m, lateral_m) prior; matched by longest prefix
CATEGORY_PRIORS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "vehicle": ((12.0, 30.0), (-4.0, 4.0)),
    "human.pedestrian": ((6.0, 20.0), (-5.0, 5.0)),
    "movable_object": ((6.0, 20.0), (-5.0, 5.0)),
}
DEFAULT_PRIOR = ((8.0, 25.0), (-4.0, 4.0))


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
