"""Ingest GLB sources (road_ood / Objaverse) as canonical mesh records.

Reads ``<assets.root>/<glb_dir>/*.glb`` (``glb_sources`` may be a symlink to
the shared road_ood library), converts each to ``<library>/canonical/glb/
<stem>.npz`` (cached by mtime), runs the ``gates.glb`` window, and appends
``kind="glb_mesh"`` records. Records are NOT part of the default dealing pool
(``run_inject_gauss`` filters ``realcar_mesh``); select them explicitly with
``--assets glb/<stem>``.

Category is fixed to ``movable_object.anomaly`` — the OOD annotation class
the upstream pipeline also uses for road_ood inserts.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import quality
from .complete_runner import completion_artifact
from .glb_convert import glb_to_npz
from .manifest import AssetRecord

log = logging.getLogger("ingest_glb")

CATEGORY = "movable_object.anomaly"


def ingest_glb(assets_root: str | Path, library_root: str | Path,
               cfg: dict | None = None, limit: int | None = None,
               ) -> list[AssetRecord]:
    """Convert + gate every GLB under ``cfg['glb_dir']``; returns records."""
    cfg = cfg or {}
    assets_root = Path(assets_root).resolve()
    library_root = Path(library_root)
    glb_dir = assets_root / str(cfg.get("glb_dir", "glb_sources"))
    if not glb_dir.is_dir():
        return []
    glbs = sorted(glb_dir.glob("*.glb"))
    if limit:
        glbs = glbs[:limit]

    records: list[AssetRecord] = []
    for glb in glbs:
        stem = glb.stem
        out = library_root / "canonical" / "glb" / f"{stem}.npz"
        source_rel = str(glb.relative_to(assets_root))
        try:
            stale = not out.exists() or glb.stat().st_mtime > out.stat().st_mtime
            if not stale:
                with np.load(out) as data:
                    # npz written before size priors existed must be redone
                    stale = "size_scale" not in data.files
            if stale:
                glb_to_npz(glb, out)
            with np.load(out) as data:
                canon = {k: data[k] for k in data.files}
            dims = tuple(float(v) for v in canon["dims_lwh"])
            colors = np.asarray(canon["colors"], dtype=np.float32)
            std = float(colors.std() * 255.0)
            black = float((colors.max(axis=1) < 20.0 / 255.0).mean())
            passed, reasons = quality.gate_glb(
                dims, std, cfg.get("gates"), texture_black_frac=black)

            canonical_rel = str(out.relative_to(library_root))
            completion = {"status": "none"}
            comp = completion_artifact(library_root, "realcar", stem)
            if comp and comp.get("canonical"):
                canonical_rel = comp["canonical"]
                completion = {"status": comp["status"],
                              "method": comp.get("method"),
                              "completed_at": comp.get("completed_at")}

            records.append(AssetRecord(
                asset_id=f"glb/{stem}",
                kind="glb_mesh",
                source_path=source_rel,
                canonical_path=canonical_rel,
                category=CATEGORY,
                dims_lwh_m=dims,
                raw_bbox_min=tuple(float(v) for v in canon["raw_bbox_min"]),
                raw_bbox_size=tuple(float(v) for v in canon["raw_bbox_size"]),
                quality={
                    "passed": passed,
                    "reasons": reasons,
                    "n_verts": int(len(canon["verts"])),
                    "n_faces": int(len(canon["faces"])),
                    "texture_std": std,
                    "texture_black_frac": black,
                    "yaw_deg": float(np.degrees(float(canon["yaw_rad"]))),
                },
                completion=completion,
                tracks=[],
            ))
        except Exception as exc:  # noqa: BLE001 - one bad glb must not kill the build
            log.warning("ingest %s failed: %s", glb.name, exc)
            records.append(AssetRecord(
                asset_id=f"glb/{stem}",
                kind="glb_mesh",
                source_path=source_rel,
                canonical_path="",
                category=CATEGORY,
                dims_lwh_m=(0.0, 0.0, 0.0),
                quality={"passed": False,
                         "reasons": [f"ingest_failed:{type(exc).__name__}"]},
                tracks=[],
            ))
    return records
