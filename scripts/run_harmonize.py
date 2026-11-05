#!/usr/bin/env python
"""Two-pass CPU harmonization of a Gaussian-injection batch.

Reconstructs alpha/visible masks from persisted artifacts (gauss/*.png +
depth/*.npy), smooths the colour correction over time (anti-flicker), refines
each composited frame and writes refined/<cam>.jpg beside composite/. Metrics
land in harmonize_report.json inside the batch dir.

    python scripts/run_harmonize.py --run-id gs01 --scene scene-0061
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realinject import paths  # noqa: E402

ALL_CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/harmonize.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--batch-dir", default=None, help="override injection.batch_dir")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--refiner", default=None,
                    choices=["none", "cpu", "sd_inpaint", "magicdrive"])
    args = ap.parse_args(argv)

    cfg = paths.load_config(args.config)
    if args.run_id:
        batch_root = Path(str(cfg.get("output_dir", "data/outputs"))) / args.run_id
        cfg.setdefault("injection", {})["batch_dir"] = str(batch_root)
    if args.batch_dir:
        cfg["injection"]["batch_dir"] = str(paths.resolve(args.batch_dir))
    if args.scene:
        cfg["dataset"]["scene"] = args.scene

    from realinject.harmonize import get_refiner, run_session

    hcfg = cfg["harmonize"]
    refiner = get_refiner(args.refiner or hcfg.get("refiner", "cpu"), cfg=hcfg)
    batch_dir = Path(cfg["injection"]["batch_dir"])
    if not batch_dir.is_dir():
        print(f"FATAL: batch dir not found: {batch_dir} "
              f"(run scripts/run_inject_gauss.py first)", file=sys.stderr)
        return 2
    scene = cfg["dataset"]["scene"]
    cams = cfg.get("cam_channels") or ALL_CAMS

    print(f"[harmonize] batch={batch_dir} scene={scene} "
          f"refiner={refiner.name} cams={len(cams)}")
    report = run_session(batch_dir, scene, cams, hcfg, refiner)

    n_improved = 0
    for sc, per_cam in report["scenes"].items():
        for cam, r in per_cam.items():
            print(f"  {sc}/{cam}: frames={r['n_frames']}")
            print(f"    exposure |dL| {abs(r['lightness_gap_before']):.1f} -> "
                  f"{abs(r['lightness_gap_after']):.1f} "
                  f"({r['lightness_improvement_pct']:.0f}% better)")
            print(f"    chroma identity kept: "
                  f"{r['chroma_identity_kept_pct']:.0f}% "
                  f"(gap {r['chroma_gap_before']:.1f} -> "
                  f"{r['chroma_gap_after']:.1f})")
            print(f"    flicker smoothed {r['correction_flicker_smoothed']:.2f} "
                  f"vs unsmoothed {r['correction_flicker_unsmoothed']:.2f} "
                  f"({r['flicker_reduction_pct']:.0f}% lower)")
            print(f"    dE(target,ring) {r['dE_target_ring_before']:.1f} -> "
                  f"{r['dE_target_ring_after']:.1f}")
            if abs(r["lightness_gap_after"]) <= abs(r["lightness_gap_before"]):
                n_improved += 1
    print(f"[harmonize] exposure improved on {n_improved} camera(s); "
          f"report -> {batch_dir / 'harmonize_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
