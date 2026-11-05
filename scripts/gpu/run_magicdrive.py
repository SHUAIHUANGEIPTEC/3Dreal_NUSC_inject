#!/usr/bin/env python
"""GPU stub: MagicDrive bbox_add generative insertion (checkpoint required).

    python scripts/gpu/run_magicdrive.py --package data/outputs/diffusion/gs01/magicdrive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", required=True)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="trained MagicDrive checkpoint (bbox_add)")
    args = ap.parse_args()

    pkg = Path(args.package)
    specs = pkg / "inputs" / "specs"
    if not specs.exists():
        raise SystemExit(f"no specs in {pkg}/inputs")
    if not args.checkpoint:
        raise SystemExit(
            "MagicDrive bbox_add needs a trained checkpoint:\n"
            "  train per the MagicDrive README (nuScenes, bbox conditions),\n"
            "  then re-run with --checkpoint /path/to/ckpt\n"
            "see docs/GPU_DIFFUSION.md route E")
    print(f"[magicdrive] checkpoint={args.checkpoint} specs={specs}")
    print("[magicdrive] not wired yet — placeholder package only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
