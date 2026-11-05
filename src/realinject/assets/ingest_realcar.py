"""3DRealCar crop ingest.

Each ``realcar_assets/<scene>/`` holds ``mesh.obj`` (Y-up, ~45 deg rotated,
roof/glass holes), ``mesh.mtl``, ``mesh.jpg`` (4096^2 texture) and
``meta.json``. The extraction manifest carries the authoritative grade and
OBB L/W/H per scene; ``meta.dims`` is the *inflated* axis-aligned extent and is
only used as a cross-check.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import normalize, quality
from .complete_runner import completion_artifact
from .manifest import AssetRecord

CATEGORY = "vehicle.car"


def load_realcar_manifest(assets_root: str | Path, manifest_name: str) -> dict[str, dict]:
    """scene name -> manifest entry (L/W/H, grade, nv/nf)."""
    path = Path(assets_root) / manifest_name
    if not path.exists():
        return {}
    entries = json.loads(path.read_text())
    return {str(e.get("scene")): e for e in entries}


def ingest_realcar(assets_root: str | Path, library_root: str | Path,
                   cfg: dict | None = None, limit: int | None = None) -> list[AssetRecord]:
    """Normalize every 3DRealCar crop; returns records with gate verdicts."""
    cfg = cfg or {}
    assets_root = Path(assets_root).resolve()
    library_root = Path(library_root)
    realcar_dir = assets_root / cfg.get("realcar_dir", "realcar_assets")
    manifest = load_realcar_manifest(
        assets_root, cfg.get("realcar_manifest", "realcar_assets_manifest.json"))

    records: list[AssetRecord] = []
    scenes = sorted(p for p in realcar_dir.iterdir() if p.is_dir()) if realcar_dir.is_dir() else []
    if limit:
        scenes = scenes[:limit]
    for scene_dir in scenes:
        scene = scene_dir.name
        if not (scene_dir / "mesh.obj").exists():
            continue
        try:
            stats = normalize.canonicalize_realcar(
                scene_dir, assets_root, library_root / "canonical" / "realcar" / f"{scene}.npz")
        except Exception as exc:  # keep going: one broken crop must not kill the build
            records.append(AssetRecord(
                asset_id=f"realcar/{scene}", kind="realcar_mesh",
                source_path=str(scene_dir.relative_to(assets_root)),
                canonical_path="", category=CATEGORY, dims_lwh_m=(0.0, 0.0, 0.0),
                grade=(manifest.get(scene) or {}).get("grade"),
                quality={"passed": False, "reasons": [f"ingest_failed:{type(exc).__name__}"]},
            ))
            continue

        entry = manifest.get(scene)
        passed, reasons = quality.gate_realcar(
            stats["dims_lwh_m"], stats["texture_std"], entry, cfg.get("gates"))
        meta = _read_meta(scene_dir / "meta.json")

        # route-D completion (fan hole fill): swap in the completed mesh and
        # refresh the hole statistics the report tracks
        comp = completion_artifact(library_root, "realcar", scene)
        canonical_rel = str(Path(stats["canonical_path"]).relative_to(library_root))
        completion = {"status": "none"}
        if comp:
            canonical_rel = comp["canonical"]
            completion = {"status": comp["status"], "method": comp.get("method"),
                          "completed_at": comp.get("completed_at")}
            cstats = comp.get("stats", {})
            stats = {**stats,
                     "boundary_edges": int(cstats.get(
                         "boundary_edges_after", stats["boundary_edges"])),
                     "watertight": bool(cstats.get("watertight_after",
                                                   stats["watertight"])),
                     "holes_filled": cstats.get("filled", 0)}
        records.append(AssetRecord(
            asset_id=f"realcar/{scene}",
            kind="realcar_mesh",
            source_path=str(scene_dir.relative_to(assets_root)),
            canonical_path=canonical_rel,
            category=CATEGORY,
            dims_lwh_m=stats["dims_lwh_m"],
            raw_bbox_min=stats["raw_bbox_min"],
            raw_bbox_size=stats["raw_bbox_size"],
            grade=(entry or {}).get("grade"),
            quality={
                "passed": passed,
                "reasons": reasons,
                "n_verts": stats["n_verts"],
                "n_faces": stats["n_faces"],
                "boundary_edges": stats["boundary_edges"],
                "watertight": stats["watertight"],
                "texture_std": stats["texture_std"],
                "yaw_deg": stats["yaw_deg"],
                "aabb_extent": stats["aabb_extent"],
                "meta_dims": meta.get("dims"),
                "manifest_lwh": [entry.get("L"), entry.get("W"), entry.get("H")] if entry else None,
            },
            completion=completion,
            conditioning={"texture": stats["texture_path"]},
            tracks=[],
        ))
    return records


def _read_meta(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
