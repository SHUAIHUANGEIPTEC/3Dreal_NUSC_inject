#!/usr/bin/env python
"""Prepare diffusion condition packages from a Gaussian-injection batch.

CPU-only: builds the inputs + command.sh for each route and checks the
condition-source contract (frame routes must trace to ``gauss/`` frames).
``run_local`` returns the placeholder result — real inference runs the
package's command.sh on the GPU server.

    python scripts/prepare_diffusion.py --run-id gs01 --route all
    python scripts/prepare_diffusion.py --run-id gs01 --route viewcrafter --frame 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realinject import paths  # noqa: E402
from realinject.diffusion import FRAME_ROUTES, get_backend  # noqa: E402

ALL_ROUTES = ("viewcrafter", "stagg_b1", "sd_inpaint", "complete", "magicdrive")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/inject_gauss.yaml")
    ap.add_argument("--run-id", default="gs01")
    ap.add_argument("--route", default="all",
                    choices=[*ALL_ROUTES, "all"])
    ap.add_argument("--scene", default="scene-0061")
    ap.add_argument("--frame", type=int, default=4)
    ap.add_argument("--cam", default="CAM_FRONT")
    ap.add_argument("--preset", default="rotate",
                    choices=["rotate", "elevate", "loop"])
    ap.add_argument("--model", default="trellis",
                    choices=["trellis", "hunyuan3d2", "gina"])
    args = ap.parse_args(argv)

    cfg = paths.load_config(args.config)
    routes = ALL_ROUTES if args.route == "all" else (args.route,)
    out_root = paths.resolve("data/outputs/diffusion") / args.run_id

    n_ok = 0
    for route in routes:
        backend = get_backend(route)
        out_dir = out_root / route
        print(f"[prepare] {route} ({backend.role}) -> {out_dir}")
        try:
            if route == "complete":
                pkg = backend.prepare(cfg, out_dir, model=args.model)
            elif route == "magicdrive":
                pkg = backend.prepare(cfg, out_dir, run_id=args.run_id,
                                      scene=args.scene)
            else:
                pkg = backend.prepare(cfg, out_dir, run_id=args.run_id,
                                      scene=args.scene, frame=args.frame,
                                      cam=args.cam, preset=args.preset)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  SKIP: {exc}", file=sys.stderr)
            continue

        sources = pkg.meta.get("condition_sources", [])
        gauss_hit = any("/gauss/" in str(s) for s in sources)
        print(f"  inputs: {sorted(pkg.meta.get('inputs', {}))}")
        print(f"  condition_sources: {sources}")
        if route in FRAME_ROUTES:
            assert gauss_hit, f"{route}: condition not traced to /gauss/ frames"
            print("  gauss trace: OK")
        result = backend.run_local(pkg.path)
        print(f"  run_local: {result['status']} ({result.get('note', '')})")
        n_ok += 1

    print(f"[prepare] {n_ok}/{len(routes)} package(s) under {out_root}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
