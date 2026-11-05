#!/usr/bin/env python
"""Route-D asset completion: complete eligible assets and write back to the library.

CPU completion (no GPU required):
- realcar: fan hole-fill (roof/glass) -> watertight mesh for 3DGS init
- AV2 completion candidates: mirror half-shell + closed convex-hull proxy

Artifacts land in ``<library>/completed/``; re-run ``build_asset_library.py``
afterwards so the manifest picks them up and the gates re-run.

    python scripts/complete_assets.py                 # all eligible
    python scripts/complete_assets.py --kind av2      # candidates only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realinject import paths  # noqa: E402
from realinject.assets.complete_runner import run_completion  # noqa: E402
from realinject.assets.manifest import AssetLibrary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/assets_ingest.yaml")
    ap.add_argument("--library", default=None)
    ap.add_argument("--kind", default="all", choices=["all", "realcar", "av2"])
    ap.add_argument("--n-samples", type=int, default=20000,
                    help="proxy surface samples per AV2 candidate")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("complete_assets")

    cfg = paths.load_config(args.config)
    library_root = Path(args.library) if args.library else Path(
        str((cfg.get("library") or {}).get("path", "data/library")))
    manifest = library_root / "assets_manifest.json"
    if not manifest.exists():
        print(f"FATAL: {manifest} missing — run build_asset_library.py first",
              file=sys.stderr)
        return 2
    lib = AssetLibrary.load(manifest)
    records = lib.passed()
    log.info("library: %d passed records", len(records))

    report = run_completion(
        records, library_root,
        realcar=args.kind in ("all", "realcar"),
        av2_candidates=args.kind in ("all", "av2"),
        n_samples=args.n_samples)

    out = library_root / "completion_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[complete] realcar={report['realcar_completed']} "
          f"av2={report['av2_completed']} skipped={report['skipped']} "
          f"failed={report['failed']}")
    print(f"[complete] report -> {out}")
    print("[complete] next: python scripts/build_asset_library.py  "
          "(re-ingest + gates on the completed library)")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
