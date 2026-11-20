"""Quality gates for normalized assets.

Gates decide whether an asset may enter the injection pool. Rejections are
recorded with machine-readable reasons (``"l_out_of_range:5.12"``) so the
gates report can explain why e.g. a C-grade car crop was dropped.
"""
from __future__ import annotations

import re

DEFAULT_GATES: dict = {
    "realcar": {
        "grades": ["A"],
        "l_range": [3.5, 6.5],  # meters, length after yaw alignment
        "w_range": [1.4, 2.6],
        "h_range": [1.0, 2.0],
        "min_texture_std": 5.0,  # flat/degenerate texture guard (0-255 scale)
        "max_texture_black_frac": 0.05,  # UV-sampled near-black fraction;
        # catches atlases whose unpainted black regions the body UVs actually
        # land on (texture_std alone passes them: yellow-vs-black raises std)
        "bbox_manifest_tol": 0.2,  # |OBB - manifest L/W/H| / value
    },
    "av2": {
        "min_points": 300,
        "min_az_span_deg": 90.0,  # viewpoint coverage; below this the shell is a sliver
        "min_pts_per_m2": 150.0,
        "min_coverage": 0.01,
        "completion_min_az_span_deg": 200.0,  # DOG only: eligible for GPU completion
    },
    "glb": {
        # Objaverse/road_ood props: wide sanity window (unit-broken models
        # like the 4-km "Dog" die on the upper bound); texture gates match
        # realcar. No scan grade / no external manifest for this source.
        "l_range": [0.05, 40.0],
        "w_range": [0.05, 40.0],
        "h_range": [0.05, 40.0],
        "min_texture_std": 5.0,
        "max_texture_black_frac": 0.05,
    },
}

#: Real-world longest-extent priors per class (meters), copied from upstream
#: nusc-inject ``configs/ood_nuscenes.yaml -> size_by_class`` (the table that
#: keeps a fire_extinguisher from growing to 3 m). ``cone`` /
#: ``traffic_cone_02`` are local additions — upstream falls them back to the
#: wide road_facilities group which still admits a 1.8-m cone.
GLB_SIZE_BY_CLASS: dict[str, tuple[float, float]] = {
    # non-typical road users
    "boat": (1.5, 4.0), "canoe": (2.5, 5.0), "golfcart": (2.0, 3.0),
    "horse_buggy": (2.0, 3.5), "horse_carriage": (2.5, 4.0),
    "kayak": (2.5, 4.0), "sled": (1.0, 2.5), "snowmobile": (1.5, 2.5),
    "surfboard": (1.5, 2.5),
    # construction / industrial
    "generator": (0.8, 2.5), "lawn_mower": (0.8, 1.8),
    "plow__farm_equipment": (1.5, 3.5),
    "tractor__farm_equipment": (2.5, 4.5),
    # dumped / household
    "armoire": (1.5, 2.4), "automatic_washer": (0.6, 1.2),
    "bathtub": (1.5, 2.0), "bed": (1.9, 2.2), "bookcase": (1.5, 2.4),
    "cabinet": (1.0, 2.2), "plastic_bag": (0.3, 0.8),
    "refrigerator": (1.4, 2.2), "sink": (0.5, 1.2), "sofa": (1.8, 3.0),
    "table": (1.0, 2.4), "toilet": (0.5, 0.8),
    # road facilities
    "awning": (1.5, 4.0), "bench": (1.2, 2.5), "billboard": (2.0, 4.0),
    "chair": (0.6, 1.3), "fire_alarm": (0.2, 0.5),
    "fire_extinguisher": (0.4, 0.7), "fireplug": (0.6, 1.0),
    "flagpole": (2.0, 4.0), "garbage": (0.3, 0.8),
    "lamppost": (3.0, 6.0), "mail_slot": (0.2, 0.5),
    "mailbox__at_home": (0.5, 1.2), "manhole": (0.5, 0.8),
    "milestone": (0.3, 0.8), "newsstand": (1.0, 2.5),
    "parking_meter": (0.8, 1.3), "pole": (2.0, 5.0),
    "postbox__public": (0.9, 1.6), "signboard": (0.5, 1.5),
    "silo": (2.5, 5.0), "stop_sign": (1.2, 2.2),
    "street_sign": (1.5, 3.0), "streetlight": (3.0, 6.0),
    "telephone_booth": (2.0, 3.0), "telephone_pole": (4.0, 8.0),
    # local extensions (upstream group fallback too wide for these)
    "cone": (0.3, 0.8), "traffic_cone_02": (0.3, 0.8),
}

_GLB_CLASS_SUFFIX = re.compile(r"(.+)_\d{3,4}_[0-9A-Za-z]+$")


def size_range_for_class(stem: str) -> tuple[float, float] | None:
    """Longest-extent prior for a GLB stem (``fireplug_2630_<uid>`` -> class)."""
    m = _GLB_CLASS_SUFFIX.match(stem)
    cls = m.group(1) if m else stem
    return GLB_SIZE_BY_CLASS.get(cls)


def _merge(base: dict, override: dict | None) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def gates_from_cfg(cfg: dict | None) -> dict:
    return _merge(DEFAULT_GATES, (cfg or {}).get("gates"))


def _in_range(value: float, bounds: list[float]) -> bool:
    return float(bounds[0]) <= value <= float(bounds[1])


def gate_realcar(dims_lwh: tuple[float, float, float], texture_std: float,
                 manifest_entry: dict | None, gates: dict | None = None,
                 texture_black_frac: float | None = None,
                 ) -> tuple[bool, list[str]]:
    """Gate one realcar record; dims are the yaw-aligned OBB (L, W, H).

    ``texture_black_frac``: fraction of UV-sampled vertex colors that are
    near-black (None = check skipped, for callers without a texture handy).
    """
    g = _merge(DEFAULT_GATES["realcar"], (gates or {}).get("realcar"))
    reasons: list[str] = []
    length, width, height = dims_lwh

    grade = (manifest_entry or {}).get("grade")
    if grade is not None and grade not in g["grades"]:
        reasons.append(f"grade:{grade}")

    if not _in_range(length, g["l_range"]):
        reasons.append(f"l_out_of_range:{length:.2f}")
    if not _in_range(width, g["w_range"]):
        reasons.append(f"w_out_of_range:{width:.2f}")
    if not _in_range(height, g["h_range"]):
        reasons.append(f"h_out_of_range:{height:.2f}")
    if texture_std < g["min_texture_std"]:
        reasons.append(f"texture_flat:{texture_std:.1f}")

    if texture_black_frac is not None and \
            texture_black_frac > float(g["max_texture_black_frac"]):
        reasons.append(f"texture_black:{texture_black_frac:.3f}")

    if manifest_entry:
        tol = g["bbox_manifest_tol"]
        for axis, ours, ref in (
            ("l", length, manifest_entry.get("L")),
            ("w", width, manifest_entry.get("W")),
            ("h", height, manifest_entry.get("H")),
        ):
            if ref:
                rel = abs(ours - float(ref)) / max(float(ref), 1e-6)
                if rel > tol:
                    reasons.append(f"bbox_disagrees_{axis}:{ours:.2f}vs{ref}")
    return not reasons, reasons


def gate_glb(dims_lwh: tuple[float, float, float], texture_std: float,
             gates: dict | None = None,
             texture_black_frac: float | None = None,
             ) -> tuple[bool, list[str]]:
    """Gate one GLB-sourced mesh (Objaverse / road_ood).

    Delegates to :func:`gate_realcar` with the ``glb`` window and no external
    manifest entry (no scan grade, no manifest bbox to disagree with) — the
    texture / black-fraction guards stay identical.
    """
    g = _merge(DEFAULT_GATES["glb"], (gates or {}).get("glb"))
    return gate_realcar(dims_lwh, texture_std, None, {"realcar": g},
                        texture_black_frac=texture_black_frac)


def gate_av2(entry: dict, gates: dict | None = None) -> tuple[bool, list[str]]:
    """Gate one AV2 track using the extraction manifest stats."""
    g = _merge(DEFAULT_GATES["av2"], (gates or {}).get("av2"))
    reasons: list[str] = []
    n = int(entry.get("n", 0))
    az = float(entry.get("az_span_deg", 0.0))
    per_m2 = float(entry.get("pts_per_m2", 0.0))
    coverage = float(entry.get("coverage", 0.0))

    if n < g["min_points"]:
        reasons.append(f"too_few_points:{n}")
    if az < g["min_az_span_deg"]:
        reasons.append(f"az_span_low:{az:.0f}")
    if per_m2 < g["min_pts_per_m2"]:
        reasons.append(f"density_low:{per_m2:.0f}")
    if coverage < g["min_coverage"]:
        reasons.append(f"coverage_low:{coverage:.3f}")
    return not reasons, reasons


def completion_candidate(entry: dict, gates: dict | None = None) -> bool:
    """Entry wide enough to carry a single-view 3D completion request.

    The az-span gate is the whole criterion: a track swept over
    completion_min_az_span_deg has seen the object from enough sides for the
    mirror+proxy (CPU) or generative (GPU) completion to close it; anything
    below stays a half-shell point asset. The former ``cat == "DOG"``
    special-case blocked every vehicle from the same numeric gate.
    """
    g = _merge(DEFAULT_GATES["av2"], (gates or {}).get("av2"))
    return float(entry.get("az_span_deg", 0.0)) >= g["completion_min_az_span_deg"]
