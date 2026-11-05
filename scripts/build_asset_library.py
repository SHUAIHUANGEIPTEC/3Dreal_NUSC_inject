#!/usr/bin/env python
"""Build the normalized asset library (canonical npz + manifest + gates report).

Usage:
    conda activate nusc-inject
    python scripts/build_asset_library.py                 # uses local.yaml paths
    python scripts/build_asset_library.py --limit 5       # smoke test on 5 per source
    python scripts/build_asset_library.py --assets-root /path/to/3D-assets --library /tmp/lib
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realinject import paths  # noqa: E402
from realinject.assets.ingest_av2 import ingest_av2  # noqa: E402
from realinject.assets.ingest_realcar import ingest_realcar  # noqa: E402
from realinject.assets.manifest import AssetLibrary  # noqa: E402


def _summary(assets_root: Path, library_root: Path, lib: AssetLibrary) -> str:
    lines = [
        f"assets root : {assets_root}",
        f"library     : {library_root}",
        f"records     : {len(lib)}  passed: {len(lib.passed())}",
    ]
    for kind, recs in sorted(lib.by_kind().items()):
        lines.append(f"  {kind:14s} {len(recs):4d} passed")
    cat = lib.by_category()
    for name, recs in sorted(cat.items()):
        lines.append(f"  {name:42s} {len(recs):4d}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/assets_ingest.yaml")
    ap.add_argument("--assets-root", default=None)
    ap.add_argument("--library", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="ingest at most N assets per source (smoke test)")
    args = ap.parse_args(argv)

    cfg = paths.load_config(args.config)
    assets_cfg = dict(cfg.get("assets") or {})
    assets_root = Path(args.assets_root) if args.assets_root else Path(
        str(assets_cfg.get("root", "")))
    library_root = Path(args.library) if args.library else Path(
        str((cfg.get("library") or {}).get("path", "data/library")))

    if not assets_root.is_dir():
        print(f"FATAL: assets root not found: {assets_root}\n"
              f"  copy local.yaml.example to local.yaml and set assets.root, "
              f"or pass --assets-root", file=sys.stderr)
        return 2

    ingest_cfg = dict(assets_cfg)
    ingest_cfg["gates"] = cfg.get("gates")
    ingest_cfg["gaussian_init"] = cfg.get("gaussian_init")

    records = ingest_realcar(assets_root, library_root, ingest_cfg, limit=args.limit)
    records += ingest_av2(assets_root, library_root, ingest_cfg, limit=args.limit)

    lib = AssetLibrary(
        records,
        gates=cfg.get("gates"),
        assets_root=str(assets_root),
        library_root=str(library_root),
    )
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    manifest_path = lib.save(library_root / "assets_manifest.json", generated_at=stamp)
    gates_path = library_root / "gates_report.json"
    gates_path.write_text(json.dumps(
        {"generated_at": stamp, **lib.gates_report()}, indent=2, ensure_ascii=False))

    print(_summary(assets_root, library_root, lib))
    print(f"\nmanifest: {manifest_path}")
    print(f"gates   : {gates_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
