"""Quality gates for normalized assets.

Gates decide whether an asset may enter the injection pool. Rejections are
recorded with machine-readable reasons (``"l_out_of_range:5.12"``) so the
gates report can explain why e.g. a C-grade car crop was dropped.
"""
from __future__ import annotations

DEFAULT_GATES: dict = {
    "realcar": {
        "grades": ["A"],
        "l_range": [3.5, 6.5],  # meters, length after yaw alignment
        "w_range": [1.4, 2.6],
        "h_range": [1.0, 2.0],
        "min_texture_std": 5.0,  # flat/degenerate texture guard (0-255 scale)
        "bbox_manifest_tol": 0.2,  # |OBB - manifest L/W/H| / value
    },
    "av2": {
        "min_points": 300,
        "min_az_span_deg": 90.0,  # viewpoint coverage; below this the shell is a sliver
        "min_pts_per_m2": 150.0,
        "min_coverage": 0.01,
        "completion_min_az_span_deg": 200.0,  # DOG only: eligible for GPU completion
    },
}


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
                 manifest_entry: dict | None, gates: dict | None = None) -> tuple[bool, list[str]]:
    """Gate one realcar record; dims are the yaw-aligned OBB (L, W, H)."""
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
    """DOG tracks wide enough to carry a single-view 3D completion request.

    Report §7.2: only ~347 deg views show enough of the animal; everything
    below completion_min_az_span_deg stays a half-shell point asset.
    """
    g = _merge(DEFAULT_GATES["av2"], (gates or {}).get("av2"))
    return (
        str(entry.get("cat", "")).upper() == "DOG"
        and float(entry.get("az_span_deg", 0.0)) >= g["completion_min_az_span_deg"]
    )
