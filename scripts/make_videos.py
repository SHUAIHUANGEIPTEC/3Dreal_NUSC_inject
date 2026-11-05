#!/usr/bin/env python
"""Assemble multi-camera sequence videos from a batch run.

Per camera: refined/<cam>.jpg if present else composite/<cam>.jpg, in frame
order -> data/videos/<run_id>/cam_<CAM>.mp4, plus a 2x3 grid video.

    python scripts/make_videos.py --run-id gs01 --scene scene-0061
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realinject import paths  # noqa: E402
from realinject.stag.video import grid_2x3, write_mp4  # noqa: E402

ALL_CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def _frame_dirs(batch_dir: Path, scene: str) -> list[Path]:
    d = batch_dir / scene

    def key(p: Path):
        m = re.match(r"f(\d+)", p.name)
        return int(m.group(1)) if m else -1
    return sorted([p for p in d.glob("f*") if p.is_dir()], key=key)


def load_cam_sequence(frame_dirs: list[Path], cam: str,
                      dataset_root: Path | None = None,
                      meta=None) -> np.ndarray | None:
    """refined -> composite -> original nuScenes image (object may simply
    not be visible from that camera — physically correct fallback)."""
    frames = []
    for fd in frame_dirs:
        p = fd / "refined" / f"{cam}.jpg"
        if not p.exists():
            p = fd / "composite" / f"{cam}.jpg"
        if not p.exists():
            # original nuScenes frame: spec.json first, gt.json as fallback
            # (gt.json always exists and carries sample_token)
            token = None
            for name in ("spec.json", "gt.json"):
                src = fd / name
                if not src.exists():
                    continue
                data = json.loads(src.read_text())
                if isinstance(data, list):
                    data = data[0] if data else {}
                token = data.get("sample_token")
                if name == "spec.json":
                    fname = data.get("cam_filenames", {}).get(cam)
                    if fname:
                        p = Path(dataset_root) / fname
                        break
                if token and meta is not None:
                    sd = meta.sd_for_sample(token, cam)
                    p = Path(dataset_root) / sd["filename"]
                    break
            if not p.exists():
                continue
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not frames:
        return None
    t = min(len(f) for f in frames)
    return np.stack([f for f in frames[:t]])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/harmonize.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--batch-dir", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--out", default=None, help="output dir (default output_dir/videos/<run_id>)")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args(argv)

    cfg = paths.load_config(args.config)
    run_id = args.run_id or Path(str(cfg.get("injection", {}).get(
        "batch_dir", "gs01"))).name
    if args.batch_dir:
        batch_dir = paths.resolve(args.batch_dir)
    else:
        batch_dir = Path(str(cfg.get("output_dir", "data/outputs"))) / run_id
    scene = args.scene or cfg["dataset"]["scene"]
    out_root = Path(args.out) if args.out else (
        Path(str(cfg.get("output_dir", "data/outputs"))) / "videos" / run_id)
    if not out_root.is_absolute():
        out_root = ROOT / out_root

    frame_dirs = _frame_dirs(batch_dir, scene)
    if not frame_dirs:
        print(f"FATAL: no frames under {batch_dir / scene}", file=sys.stderr)
        return 2
    print(f"[videos] {len(frame_dirs)} frames, scene={scene}")

    meta = None
    dataset_root = Path(str((cfg.get("dataset") or {}).get("root", "")))
    try:
        from nusc_inject import nusc_io

        meta = nusc_io.NuScenesMini(str(dataset_root), cfg["dataset"]["version"])
    except Exception as exc:  # noqa: BLE001 - fallback path only
        print(f"  (nuScenes meta unavailable, original-frame fallback off: {exc})")

    seqs: dict[str, np.ndarray] = {}
    for cam in ALL_CAMS:
        arr = load_cam_sequence(frame_dirs, cam, dataset_root, meta)
        if arr is None:
            print(f"  {cam}: no frames")
            continue
        seqs[cam] = arr
        path = write_mp4(np.repeat(arr, args.repeat, axis=0),
                         out_root / f"cam_{cam}.mp4", fps=args.fps)
        print(f"  {cam}: {len(arr)} frames -> {path.name}")

    if seqs:
        order = [c for c in ALL_CAMS if c in seqs]
        # pad missing cams with black so the 2x3 grid stays aligned
        h, w = next(iter(seqs.values())).shape[1:3]
        t = min(len(a) for a in seqs.values())
        padded = {}
        for c in ALL_CAMS:
            if c in seqs:
                padded[c] = seqs[c][:t]
            else:
                padded[c] = np.zeros((t, h, w, 3), np.uint8)
        grid = grid_2x3(padded, ALL_CAMS)
        write_mp4(np.repeat(grid, args.repeat, axis=0),
                  out_root / "grid_2x3.mp4", fps=args.fps)
        print(f"  grid: {len(order)}/6 cams -> grid_2x3.mp4")
    print(f"[videos] done -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
