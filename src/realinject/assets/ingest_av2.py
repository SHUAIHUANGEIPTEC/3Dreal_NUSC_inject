"""Argoverse2 object-cloud ingest.

``av2_assets/_merged/<CAT>_<shorttrack>.npz`` holds one track's aggregated
half-shell: ``pts`` (N,3, object-local meters, Z-up, bbox-centred),
``intensity`` (N,) uint8 reflectance, ``dims`` (3,) box LWH, ``ego``
(n_frames,3) per-frame ego positions, ``cat``. Per-track stats live in
``_merged/manifest.json``; a conditioning image for completion requests sits
in ``av2_assets/<scene>/view/<CAT>_<shorttrack>.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import normalize, quality
from .manifest import AssetRecord

#: AV2 category -> nuScenes category used in the annotation export.
#: DOG has no nuScenes counterpart: it is an intentional OOD category.
AV2_CATEGORY_MAP = {
    "DOG": "human.pedestrian.animal",
    "REGULAR_VEHICLE": "vehicle.car",
    "MOTORCYCLE": "vehicle.motorcycle",
    "WHEELED_DEVICE": "human.pedestrian.personal_mobility",
    "WHEELED_RIDER": "human.pedestrian.other",
    "STROLLER": "human.pedestrian.stroller",
    "WHEELCHAIR": "human.pedestrian.wheelchair",
}
FALLBACK_CATEGORY = "movable_object.anomaly"


def load_av2_manifest(av2_dir: str | Path) -> list[dict]:
    path = Path(av2_dir) / "manifest.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else data.get("tracks", [])


def _find_scene_dir(av2_scene_root: Path, npz_name: str) -> Path | None:
    """Scene dir holding the track npz (``_merged`` is an aggregate, not a scene).

    Prefers a scene that also carries the ``view/`` conditioning image.
    """
    if not av2_scene_root.is_dir():
        return None
    fallback = None
    for scene_dir in sorted(av2_scene_root.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name == "_merged":
            continue
        if not (scene_dir / npz_name).exists():
            continue
        if (scene_dir / "view").is_dir():
            return scene_dir
        if fallback is None:
            fallback = scene_dir
    return fallback


def ingest_av2(assets_root: str | Path, library_root: str | Path,
               cfg: dict | None = None, limit: int | None = None) -> list[AssetRecord]:
    """Normalize every AV2 merged track; returns records with gate verdicts."""
    cfg = cfg or {}
    assets_root = Path(assets_root).resolve()
    library_root = Path(library_root)
    av2_dir = assets_root / cfg.get("av2_dir", "av2_assets/_merged")
    scene_root = assets_root / cfg.get("av2_scene_dir", "av2_assets")

    records: list[AssetRecord] = []
    entries = load_av2_manifest(av2_dir)
    if limit:
        entries = entries[:limit]
    for entry in entries:
        cat = str(entry.get("cat", "")).upper()
        short = str(entry.get("track", ""))[:8]
        npz_name = f"{cat}_{short}.npz"
        npz_path = av2_dir / npz_name
        if not npz_path.exists():
            continue
        asset_id = f"av2/{cat}_{short}"
        try:
            stats = normalize.canonicalize_av2(
                npz_path, library_root / "canonical" / "av2" / npz_name)
        except Exception as exc:
            records.append(AssetRecord(
                asset_id=asset_id, kind="av2_cloud",
                source_path=str(npz_path.relative_to(assets_root)),
                canonical_path="", category=AV2_CATEGORY_MAP.get(cat, FALLBACK_CATEGORY),
                dims_lwh_m=(0.0, 0.0, 0.0),
                quality={"passed": False, "reasons": [f"ingest_failed:{type(exc).__name__}"]},
            ))
            continue

        passed, reasons = quality.gate_av2(entry, cfg.get("gates"))
        scene_dir = _find_scene_dir(scene_root, npz_name)
        view = (scene_dir / "view" / f"{cat}_{short}.png") if scene_dir else None
        conditioning = {}
        if view is not None and view.exists():
            conditioning["view_image"] = str(view.relative_to(assets_root))

        records.append(AssetRecord(
            asset_id=asset_id,
            kind="av2_cloud",
            source_path=str(npz_path.relative_to(assets_root)),
            canonical_path=str(Path(stats["canonical_path"]).relative_to(library_root)),
            category=AV2_CATEGORY_MAP.get(cat, FALLBACK_CATEGORY),
            dims_lwh_m=stats["dims_lwh_m"],
            raw_bbox_min=stats["raw_bbox_min"],
            raw_bbox_size=stats["raw_bbox_size"],
            quality={
                "passed": passed,
                "reasons": reasons,
                "n_points": stats["n_points"],
                "cloud_extent": stats["cloud_extent"],
                "az_span_deg": entry.get("az_span_deg"),
                "pts_per_m2": entry.get("pts_per_m2"),
                "coverage": entry.get("coverage"),
                "frames": entry.get("frames"),
                "silhouette": entry.get("silhouette"),
            },
            completion={
                "status": "none",
                "candidate": quality.completion_candidate(entry, cfg.get("gates")),
            },
            conditioning=conditioning,
            tracks=[{
                "track": entry.get("track"),
                "n_frames": entry.get("frames"),
                "box_LWH": entry.get("box_LWH"),
                "ego_key": "ego",  # per-frame ego positions stored in the canonical npz
            }],
        ))
    return records
