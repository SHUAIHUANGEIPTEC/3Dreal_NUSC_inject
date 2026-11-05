"""Output helpers: PLY point clouds, PNG frame sequences, MP4 videos."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_ply(points: np.ndarray, colors: np.ndarray | None, path) -> None:
    """ASCII PLY with optional per-point RGB (MeshLab/CloudCompare openable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {n}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            fh.write("property uchar red\nproperty uchar green\n"
                     "property uchar blue\n")
        fh.write("end_header\n")
        if colors is None:
            for p in points:
                fh.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")
        else:
            colors = np.asarray(colors)
            for p, c in zip(points, colors, strict=True):
                fh.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} "
                         f"{int(c[0])} {int(c[1])} {int(c[2])}\n")
    print(f"  saved point cloud ({n} pts) -> {path}")


def save_frames(frames: np.ndarray, out_dir, name: str) -> list[Path]:
    """Save (T, H, W, 3) uint8 frames as <name>_000.png ..."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, f in enumerate(np.asarray(frames)):
        p = out_dir / f"{name}_{i:03d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        paths.append(p)
    print(f"  saved {len(paths)} frames -> {name}_*.png")
    return paths


def write_mp4(frames: np.ndarray, path, fps: int = 8) -> Path:
    """Write (T, H, W, 3) uint8 RGB frames to an MP4 (mp4v codec)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames)
    t, h, w = frames.shape[:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {path}")
    try:
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    print(f"  saved video ({t} frames @ {fps}fps) -> {path}")
    return path


def grid_2x3(frames_per_cam: dict[str, np.ndarray], order: list[str]
             ) -> np.ndarray:
    """Tile 6 camera sequences into one (T, 2H, 3W, 3) grid video."""
    seqs = [np.asarray(frames_per_cam[c]) for c in order]
    t = min(len(s) for s in seqs)
    h, w = seqs[0].shape[1:3]
    out = np.zeros((t, 2 * h, 3 * w, 3), dtype=np.uint8)
    # layout: FL F FR / BL B BR
    slots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for s, (r, c) in zip(seqs, slots, strict=False):
        out[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = s[:t]
    return out
