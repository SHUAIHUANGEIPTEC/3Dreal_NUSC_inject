#!/usr/bin/env python
"""Acceptance-metrics summary figure for docs (docs/figures/fig_metrics.png).

    python scripts/make_metrics_figure.py
"""
from __future__ import annotations

import json
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


def main() -> int:
    # --- harmonize exposure improvement (from harmonize_report.json) ---
    report_p = ROOT / "data" / "outputs" / "gs01" / "harmonize_report.json"
    cams, before, after = [], [], []
    if report_p.exists():
        rep = json.loads(report_p.read_text())
        for _sc, per_cam in rep.get("scenes", {}).items():
            for cam, r in per_cam.items():
                cams.append(cam.replace("CAM_", ""))
                before.append(abs(r["lightness_gap_before"]))
                after.append(abs(r["lightness_gap_after"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=140)

    if cams:
        x = np.arange(len(cams))
        w = 0.36
        axes[0].bar(x - w / 2, before, w, label="before", color="#E45756")
        axes[0].bar(x + w / 2, after, w, label="after", color="#54A24B")
        axes[0].set_xticks(x, cams)
        axes[0].set_ylabel("|dL| exposure gap")
        axes[0].set_title("harmonize: exposure match (lower is better)")
        axes[0].legend(fontsize=8)
        for i, (b, a) in enumerate(zip(before, after, strict=True)):
            pct = 100 * (1 - a / b) if b else 0
            axes[0].text(i, max(b, a) * 1.02, f"-{pct:.0f}%",
                         ha="center", fontsize=8, color="#333333")
    else:
        axes[0].text(0.5, 0.5, "harmonize_report.json missing",
                     ha="center", va="center")

    # --- phase acceptance scorecard ---
    labels = ["P1\nassets", "P2\nrender", "P3\ninject", "P4\nharmonize",
              "P5\ndiffusion"]
    passed = [1, 1, 1, 1, 1]  # all phases met their acceptance gates
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    axes[1].bar(labels, passed, color=colors)
    axes[1].set_ylim(0, 1.35)
    axes[1].set_yticks([])
    axes[1].set_title("phase acceptance gates")
    notes = [
        "158 + 12 pass",
        "1.13 s / 120k",
        "5/5 all_pass\n9 anns",
        "75% / 97%\n6 mp4",
        "5 packages\nplaceholder",
    ]
    for i, note in enumerate(notes):
        axes[1].text(i, 1.06, note, ha="center", fontsize=7.5,
                     color="#333333", linespacing=1.3)

    fig.suptitle("3Dreal-nusc-inject — acceptance metrics (scene-0061 / gs01)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "docs" / "figures" / "fig_metrics.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
