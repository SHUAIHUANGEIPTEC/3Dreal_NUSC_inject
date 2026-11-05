#!/usr/bin/env python
"""Architecture / pipeline figure for docs (docs/figures/fig_pipeline.png).

    python scripts/make_pipeline_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

C_ASSET = "#4C78A8"
C_GAUSS = "#F58518"
C_INJECT = "#54A24B"
C_DIFF = "#E45756"
C_IO = "#B279A2"


def box(ax, xy, w, h, text, fc, fontsize=8.5):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                linewidth=1.2, edgecolor="#333333",
                                facecolor=fc, alpha=0.92))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="white", linespacing=1.35)


def arrow(ax, p0, p1, color="#333333"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=color,
                                 shrinkA=2, shrinkB=2))


def main() -> int:
    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    # ---- row 1: assets ----
    box(ax, (2, 50), 20, 8, "3DRealCar 199 套\n带贴图网格（A 档 158）", C_ASSET)
    box(ax, (25, 50), 20, 8, "Argoverse2 24 条\n稀疏半壳点云", C_ASSET)
    box(ax, (48, 50), 22, 8, "assets/ 规范化\n主轴对齐·Y-up→Z-up·UV缝合并", C_ASSET)
    box(ax, (73, 50), 24, 8, "质量门禁 +资产库\nassets_manifest.json（170 通过）", C_ASSET)
    arrow(ax, (22, 54), (25, 54))
    arrow(ax, (45, 54), (48, 54))
    arrow(ax, (70, 54), (73, 54))

    # ---- row 2: gaussian ----
    box(ax, (2, 36), 22, 8, "GaussianCloud\n网格采样 / 点云抬升", C_GAUSS)
    box(ax, (27, 36), 24, 8, "高斯渲染器 raster_cpu\nEWA 足迹 + 软 z 累积\n120k @1600×900 = 1.13s", C_GAUSS)
    box(ax, (54, 36), 20, 8, "LiDAR 通道适配\nlidar_splat（法向/反射）", C_GAUSS)
    box(ax, (77, 36), 20, 8, "gauss/<cam>.png\nRGBA + depth", C_GAUSS)
    arrow(ax, (13, 50), (13, 44))
    arrow(ax, (35, 50), (39, 44))
    arrow(ax, (24, 40), (27, 40))
    arrow(ax, (51, 40), (54, 40))
    arrow(ax, (74, 40), (77, 40))

    # ---- row 3: inject ----
    box(ax, (2, 22), 20, 8, "摆放 + 锚定\nanchor.build_gt 单一真值", C_INJECT)
    box(ax, (25, 22), 24, 8, "注入管线 pipeline_gauss\nLiDAR z-buffer 遮挡 + 合成", C_INJECT)
    box(ax, (52, 22), 22, 8, "校验 validate_ext\n深度/IoU/遮挡判定", C_INJECT)
    box(ax, (77, 22), 20, 8, "标准 nuScenes 导出\ndevkit verify ok", C_INJECT)
    arrow(ax, (39, 36), (37, 30))
    arrow(ax, (64, 40), (63, 30))
    arrow(ax, (22, 26), (25, 26))
    arrow(ax, (49, 26), (52, 26))
    arrow(ax, (74, 26), (77, 26))

    # ---- row 4: harmonize + diffusion ----
    box(ax, (2, 8), 22, 8, "和谐化 harmonize\n曝光匹配·接触阴影·anti-flicker", C_DIFF)
    box(ax, (27, 8), 24, 8, "视频 make_videos\n6 路 mp4 + 2×3 网格", C_DIFF)
    box(ax, (54, 8), 20, 8, "扩散路线 ×5\n补全/真实感/重绘/生成", C_DIFF)
    box(ax, (77, 8), 20, 8, "GPU 服务器\nViewCrafter·Stag·TRELLIS", C_IO)
    arrow(ax, (13, 22), (13, 16))
    arrow(ax, (37, 22), (39, 16))
    arrow(ax, (66, 12), (77, 12))
    arrow(ax, (51, 12), (54, 12))

    # ---- side notes ----
    ax.text(50, 60.5, "3Dreal-nusc-inject — 真实提取资产注入 nuScenes（高斯渲染 + 扩散真实感化）",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(50, 57.8, "输入资产：3DRealCar 实车扫描 + Argoverse2 动物/OOD 轨迹点云（非 Objaverse 实体模型）",
            ha="center", fontsize=9, color="#555555")
    ax.text(2, 3.2, "复用：nusc-inject 只读（anchor / lidar_channel / occluders / validate / annotation）"
                    "   ｜   帧源：高斯渲染（Blender 仅作 A/B 基线）",
            fontsize=8, color="#555555")

    out = ROOT / "docs" / "figures" / "fig_pipeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
