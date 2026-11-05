#!/usr/bin/env python
"""Render the injection before/after figure for docs.

    python scripts/make_inject_figure.py --batch-dir data/outputs/gs01 --frame f004
"""
from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch-dir", default="data/outputs/gs01")
    ap.add_argument("--scene", default="scene-0061")
    ap.add_argument("--frame", default="f004")
    ap.add_argument("--cam", default="CAM_FRONT")
    ap.add_argument("--out", default="docs/figures/fig_inject_before_after.png")
    args = ap.parse_args(argv)

    import cv2

    from realinject import paths

    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_absolute():
        batch_dir = ROOT / batch_dir
    frame_dir = batch_dir / args.scene / args.frame
    if not frame_dir.is_dir():
        print(f"FATAL: {frame_dir} missing (run scripts/run_inject_gauss.py first)",
              file=sys.stderr)
        return 2

    cfg = paths.load_config("configs/inject_gauss.yaml")
    from nusc_inject import nusc_io

    meta = nusc_io.NuScenesMini(cfg["dataset"]["root"], cfg["dataset"]["version"])
    gt = json.loads((frame_dir / "gt.json").read_text())
    gt = gt if isinstance(gt, list) else [gt]
    sample_token = gt[0]["sample_token"]
    sd = meta.sd_for_sample(sample_token, args.cam)
    orig = cv2.cvtColor(cv2.imread(str(meta.sensor_path(sd))), cv2.COLOR_BGR2RGB)

    comp = cv2.cvtColor(cv2.imread(str(frame_dir / "composite" / f"{args.cam}.jpg")),
                        cv2.COLOR_BGR2RGB)
    gauss = cv2.imread(str(frame_dir / "gauss" / f"{args.cam}.png"),
                       cv2.IMREAD_UNCHANGED)
    alpha = gauss[..., 3].astype(np.float32) / 255.0 if gauss.shape[2] == 4 else \
        np.ones(gauss.shape[:2], np.float32)
    fg = gauss[..., :3].astype(np.float32)
    fg_comp = fg * alpha[..., None] + 255.0 * (1 - alpha[..., None])

    # lidar object points projected into the camera (cross-modal check)
    k = np.array(meta.calib[sd["calibrated_sensor_token"]]["camera_intrinsic"], float)
    lidar = np.load(frame_dir / "lidar" / "obj_points_lidar.npy")
    # T_l2c = inv(cam2global) @ lidar2global (each sensor has its own pose)
    from nusc_inject import transforms as tf

    cs = meta.calib[sd["calibrated_sensor_token"]]
    pose = meta.poses[sd["ego_pose_token"]]
    t_c2g = (tf.make_T(pose["rotation"], pose["translation"])
             @ tf.make_T(cs["rotation"], cs["translation"]))
    lidar_sd = meta.sd_for_sample(sample_token, cfg["dataset"]["lidar_channel"])
    lcs = meta.calib[lidar_sd["calibrated_sensor_token"]]
    lpose = meta.poses[lidar_sd["ego_pose_token"]]
    t_l2g = (tf.make_T(lpose["rotation"], lpose["translation"])
             @ tf.make_T(lcs["rotation"], lcs["translation"]))
    t_l2c = np.linalg.inv(t_c2g) @ t_l2g
    pts_c = tf.transform_pts(t_l2c, lidar[:, :3])
    front = pts_c[:, 2] > 0.1
    u = k[0, 0] * pts_c[front, 0] / pts_c[front, 2] + k[0, 2]
    v = k[1, 1] * pts_c[front, 1] / pts_c[front, 2] + k[1, 2]
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    ok = (ui >= 0) & (ui < orig.shape[1]) & (vi >= 0) & (vi < orig.shape[0])

    overlay = comp.copy()
    overlay[vi[ok], ui[ok]] = [255, 0, 0]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), dpi=130)
    panels = [
        ("original", orig),
        ("gaussian fg (alpha)", fg_comp.astype(np.uint8)),
        ("composite", comp),
        ("composite + lidar obj pts", overlay),
    ]
    for ax, (title, img) in zip(axes, panels, strict=True):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    n_obj = len(gt)
    fig.suptitle(f"{args.scene} {args.frame} — {n_obj} injected object(s), "
                 f"{args.cam} (red = injected LiDAR returns)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
