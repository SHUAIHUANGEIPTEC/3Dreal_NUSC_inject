#!/usr/bin/env python
"""Harmonization before/after figure for docs.

    python scripts/make_harmonize_ab.py --run-id gs01 --frame f004
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default="gs01")
    ap.add_argument("--scene", default="scene-0061")
    ap.add_argument("--frame", default="f004")
    ap.add_argument("--cam", default="CAM_FRONT")
    ap.add_argument("--out", default="docs/figures/fig_harmonize_ab.png")
    args = ap.parse_args(argv)

    import cv2

    from realinject.harmonize.masks import load_frame_cam, ring_region

    batch = ROOT / "data" / "outputs" / args.run_id
    fd = batch / args.scene / args.frame
    rec = load_frame_cam(fd, args.cam)
    if rec is None:
        print(f"FATAL: no composite artifacts under {fd}", file=sys.stderr)
        return 2
    refined_p = fd / "refined" / f"{args.cam}.jpg"
    if not refined_p.exists():
        print(f"FATAL: {refined_p} missing (run scripts/run_harmonize.py)",
              file=sys.stderr)
        return 2
    refined = cv2.cvtColor(cv2.imread(str(refined_p)), cv2.COLOR_BGR2RGB)

    comp = rec["comp_rgb"]
    target = rec["target"]
    ring = ring_region(target)
    overlay = comp.copy()
    overlay[target] = (0.65 * overlay[target] + 0.35 * np.array([255, 80, 80]))
    overlay[ring] = (0.7 * overlay[ring] + 0.3 * np.array([80, 160, 255]))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), dpi=130)
    panels = [
        ("composite (before)", comp),
        ("regions: target / ring", overlay),
        ("refined (after)", refined),
    ]
    for ax, (title, img) in zip(axes, panels, strict=True):
        ax.imshow(np.clip(img, 0, 255).astype(np.uint8))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"harmonize {args.scene} {args.frame} {args.cam} — "
                 "exposure match + contact shadow (target red, ring blue)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
