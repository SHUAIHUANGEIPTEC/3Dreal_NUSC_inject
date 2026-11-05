"""Route-D completion runner: complete assets and write them back to the library.

CPU completion (this machine has no GPU):

- **realcar**: fan-fill boundary loops (roof/glass holes) — genuine geometry
  repair so 3DGS initialization does not leak light;
- **AV2 candidates**: mirror the half-shell across its narrow-axis mid-plane
  and fit a closed convex-hull proxy (dims-correct, not a shape
  reconstruction — GPU route D replaces it with TRELLIS/Hunyuan3D-2 output).

Artifacts land in ``<library>/completed/<kind>/<key>.npz`` + ``.json``; the
asset-library build picks them up (canonical npz swapped, quality stats
recomputed, ``completion.status`` set) and the gates re-run on the merged view.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import numpy as np

from . import complete_cpu
from .manifest import AssetRecord

log = logging.getLogger("complete")


def _completed_dir(library_root: Path, kind: str) -> Path:
    d = Path(library_root) / "completed" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def completion_artifact(library_root, kind: str, key: str) -> dict | None:
    """Load a completion sidecar if present (else None)."""
    p = Path(library_root) / "completed" / kind / f"{key}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _save(library_root: Path, kind: str, key: str, canon: dict,
          stats: dict, status: str, source: str) -> dict:
    out_dir = _completed_dir(library_root, kind)
    npz_path = out_dir / f"{key}.npz"
    np.savez_compressed(npz_path, **canon)
    rel = str(npz_path.relative_to(Path(library_root)))
    info = {
        "status": status,
        "method": stats.get("method"),
        "source": source,
        "canonical": rel,
        "stats": stats,
        "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / f"{key}.json").write_text(json.dumps(info, indent=2))
    return info


def complete_realcar(record: AssetRecord, library_root) -> dict | None:
    """Hole-fill one realcar canonical mesh; returns the sidecar dict."""
    src = Path(library_root) / record.canonical_path
    if not src.exists():
        log.warning("skip %s: canonical missing %s", record.asset_id, src)
        return None
    with np.load(src) as data:
        canon = {k: data[k] for k in data.files}
    if "verts" not in canon:
        log.warning("skip %s: not a mesh canonical", record.asset_id)
        return None
    try:
        completed, stats = complete_cpu.complete_realcar_canon(canon)
    except Exception as exc:  # noqa: BLE001 - one bad crop must not kill the run
        log.warning("complete %s failed: %s", record.asset_id, exc)
        return None
    key = record.asset_id.split("/", 1)[-1]
    info = _save(library_root, "realcar", key, completed, stats,
                 status="cpu_poisson", source=record.canonical_path)
    log.info("realcar %s: filled=%d watertight=%s verts %d->%d",
             record.asset_id, stats.get("filled", 0),
             stats.get("watertight_after"), stats.get("n_verts_before"),
             stats.get("n_verts_after"))
    return info


def complete_av2(record: AssetRecord, library_root, n_samples: int = 20000) -> dict | None:
    """Mirror + closed-proxy one AV2 candidate; returns the sidecar dict."""
    src = Path(library_root) / record.canonical_path
    if not src.exists():
        log.warning("skip %s: canonical missing %s", record.asset_id, src)
        return None
    with np.load(src, allow_pickle=True) as data:
        canon = {k: data[k] for k in data.files}
    if "xyz" not in canon:
        log.warning("skip %s: not a cloud canonical", record.asset_id)
        return None
    try:
        completed, stats = complete_cpu.complete_av2_canon(
            canon, n_samples=n_samples)
    except Exception as exc:  # noqa: BLE001
        log.warning("complete %s failed: %s", record.asset_id, exc)
        return None
    key = record.asset_id.split("/", 1)[-1]
    info = _save(library_root, "av2", key, completed, stats,
                 status="cpu_proxy", source=record.canonical_path)
    log.info("av2 %s: method=%s proxy=%d shell=%d watertight=%s az=%.0f",
             record.asset_id, stats.get("method"), stats.get("n_proxy_samples"),
             stats.get("n_shell"), stats.get("watertight_after"),
             stats.get("az_span_deg", 0))
    return info


def run_completion(records: list[AssetRecord], library_root,
                   *, realcar: bool = True, av2_candidates: bool = True,
                   n_samples: int = 20000) -> dict:
    """Complete every eligible record; returns a summary report."""
    library_root = Path(library_root)
    done_realcar = done_av2 = skipped = failed = 0
    details = []
    for rec in records:
        if not rec.quality.get("passed"):
            skipped += 1
            continue
        info = None
        if realcar and rec.kind == "realcar_mesh":
            info = complete_realcar(rec, library_root)
            done_realcar += 1 if info else 0
        elif av2_candidates and rec.kind == "av2_cloud" and \
                rec.completion.get("candidate"):
            info = complete_av2(rec, library_root, n_samples=n_samples)
            done_av2 += 1 if info else 0
        else:
            skipped += 1
            continue
        if info is None:
            failed += 1
        else:
            details.append({"asset_id": rec.asset_id, **info})
    report = {
        "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "realcar_completed": done_realcar,
        "av2_completed": done_av2,
        "skipped": skipped,
        "failed": failed,
        "details": details,
    }
    return report
