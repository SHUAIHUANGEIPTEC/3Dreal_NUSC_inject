#!/usr/bin/env python
"""Convert textured GLB models (Objaverse / road_ood) to the canonical mesh npz.

Thin CLI over ``realinject.assets.glb_convert`` (same contract as
``normalize.canonicalize_realcar`` — see that module for the transform chain).

Usage:
    python scripts/glb_to_npz.py model.glb -o model.npz
    python scripts/glb_to_npz.py models_dir/ -o out_dir/            # batch *.glb
    python scripts/glb_to_npz.py models_dir/ -o out_dir/ --glob "fireplug_*.glb"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realinject.assets.glb_convert import glb_to_npz  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="one .glb file, or a directory of .glb files")
    ap.add_argument("-o", "--out", required=True,
                    help="output .npz (file mode) or output dir (dir mode)")
    ap.add_argument("--glob", default="*.glb",
                    help="filename pattern in dir mode (default *.glb)")
    args = ap.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    if src.is_file():
        stats = glb_to_npz(src, out)
        print(f"[glb2npz] {src.name}: {stats['n_verts']} verts "
              f"dims={tuple(round(v, 2) for v in stats['dims_lwh_m'])} "
              f"yaw={stats['yaw_deg']:.1f}deg -> {out}")
        return 0

    files = sorted(src.glob(args.glob))
    if not files:
        print(f"FATAL: no files match {args.glob} under {src}", file=sys.stderr)
        return 2
    ok = failed = 0
    for f in files:
        dst = out / f"{f.stem}.npz"
        try:
            st = glb_to_npz(f, dst)
            ok += 1
            print(f"[glb2npz] ok   {f.name}: {st['n_verts']} verts "
                  f"dims={tuple(round(v, 2) for v in st['dims_lwh_m'])}")
        except Exception as exc:  # noqa: BLE001 - batch keeps going
            failed += 1
            print(f"[glb2npz] FAIL {f.name}: {type(exc).__name__}: {exc}")
    print(f"[glb2npz] done: {ok} converted, {failed} failed -> {out}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
