#!/usr/bin/env python
"""GPU: asset completion for half-shell / hole-fill via 3D generative models.

Dispatches per candidate to TRELLIS (native gaussian output), Hunyuan3D-2
(best texture) or a GINA-class single-view model. Completed assets land in
``<package>/output/<asset_id>/`` and should be copied into
``data/library/completed/`` before re-running ``build_asset_library``.

    python scripts/gpu/run_asset_completion.py \
        --package data/outputs/diffusion/gs01/complete --model trellis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", required=True)
    ap.add_argument("--model", default="trellis",
                    choices=["trellis", "hunyuan3d2", "gina"])
    args = ap.parse_args()

    pkg = Path(args.package)
    manifest = pkg / "inputs" / "manifest.json"
    if not manifest.exists():
        # write_package stores the manifest as inputs/manifest (no suffix)
        alt = pkg / "inputs" / "manifest"
        if alt.exists():
            manifest = alt
        else:
            raise SystemExit(f"no candidates manifest in {pkg}/inputs")
    rows = json.loads(manifest.read_text())

    if args.model == "trellis":
        hint = ("pip install git+https://github.com/microsoft/TRELLIS.git\n"
                "  # or follow TRELLIS README; needs >=24GB GPU")
    elif args.model == "hunyuan3d2":
        hint = ("git clone https://github.com/Tencent/Hunyuan3D-2 && "
                "pip install -e Hunyuan3D-2")
    else:
        hint = "# GINA-class single-view 3D: install per its repo README"
    print(f"[complete] model={args.model} candidates={len(rows)}")
    print(f"[complete] install hint:\n  {hint}")

    for row in rows:
        src = pkg / row["dir"]
        out = pkg / "output" / row["asset_id"].replace("/", "_")
        out.mkdir(parents=True, exist_ok=True)
        print(f"[complete] {row['asset_id']}: view={src / 'view.png'} "
              f"half_shell={src / 'half_shell.npy'} -> {out}")
        # placeholder: real inference is model-specific; keep the contract
        (out / "PENDING.txt").write_text(
            f"model={args.model}\nrun the model on view.png + half_shell.npy\n")
    print("[complete] done (placeholders written; wire the model on the server)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
