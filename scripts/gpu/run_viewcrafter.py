#!/usr/bin/env python
"""Run ViewCrafter novel-view synthesis on the GPU server (path A).

Wraps ViewCrafter's official inference.py as a subprocess with preflight
checks (CUDA, repo, checkpoints) and copies the result back into
data/outputs/viewcrafter/<name>/.

Prerequisites on the GPU server (see docs/GPU_DIFFUSION.md §1):
  1. git clone https://github.com/Drexubery/ViewCrafter  (default: third_party/ViewCrafter)
  2. its own conda env (python 3.9 + their requirements.txt + pytorch3d + xformers)
  3. checkpoints:
       checkpoints/model.ckpt          <- HF Drexubery/ViewCrafter_25
       checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth <- naverlabs link
  4. python scripts/prepare_viewcrafter.py   (CPU, can be done before copying)

Usage:
    python scripts/gpu/run_viewcrafter.py --package data/outputs/viewcrafter/<name>
        [--viewcrafter-root third_party/ViewCrafter] [--python /path/to/vc-python]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import paths  # noqa: E402

CKPT = "model.ckpt"
DUST3R = "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"


def fail(msg: str) -> int:
    print(f"[run_viewcrafter] ERROR: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True,
                    help="prepare_viewcrafter output dir (contains ref/traj)")
    ap.add_argument("--viewcrafter-root", default=None,
                    help="ViewCrafter clone (default: cfg viewcrafter.root "
                         "or third_party/ViewCrafter)")
    ap.add_argument("--python", default=sys.executable,
                    help="python of the ViewCrafter conda env")
    args = ap.parse_args()

    pkg = paths.resolve(args.package)
    ref, traj = pkg / "inputs" / "ref.png", pkg / "inputs" / "traj.txt"
    if not (ref.exists() and traj.exists()):
        return fail(f"package incomplete: {pkg} (need ref.png + traj.txt; "
                    f"run scripts/prepare_viewcrafter.py first)")

    # --- CUDA preflight (must run under the ViewCrafter env's python) ---
    cuda = subprocess.call(
        [args.python, "-c", "import torch,sys; "
                            "sys.exit(0 if torch.cuda.is_available() else 1)"])
    if cuda != 0:
        return fail("torch.cuda.is_available() is False — run this on the "
                    "GPU server with the ViewCrafter conda env activated")

    # --- ViewCrafter repo + checkpoints preflight ---
    vc = paths.resolve(args.viewcrafter_root) if args.viewcrafter_root \
        else paths.resolve("third_party/ViewCrafter")
    if not (vc / "inference.py").exists():
        return fail(
            f"ViewCrafter not found at {vc}\n"
            f"  git clone https://github.com/Drexubery/ViewCrafter {vc}\n"
            f"  (full setup: docs/GPU_DIFFUSION.md §1)")
    for rel, hint in ((f"checkpoints/{CKPT}",
                       "wget https://huggingface.co/Drexubery/ViewCrafter_25/"
                       "resolve/main/model.ckpt -P checkpoints/"),
                      (f"checkpoints/{DUST3R}",
                       "wget https://download.europe.naverlabs.com/ComputerVision/"
                       "DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/")):
        if not (vc / rel).exists():
            return fail(f"missing {vc / rel}\n  {hint}")

    # --- run official inference ---
    name = pkg.name
    out_dir = pkg / "output"
    out_dir.mkdir(exist_ok=True)
    cmd = [args.python, "inference.py",
           "--image_dir", str(ref),
           "--mode", "single_view_txt",
           "--traj_txt", str(traj),
           "--ckpt_path", f"./checkpoints/{CKPT}",
           "--model_path", f"./checkpoints/{DUST3R}",
           "--elevation", "5", "--center_scale", "1.0",
           "--prompt", "Rotating view of a scene",
           "--video_length", "25", "--height", "576", "--width", "1024",
           "--exp_name", name,
           "--out_dir", str(out_dir)]
    print(f"[run_viewcrafter] cd {vc} && {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(vc))
    if rc != 0:
        return fail(f"inference.py failed (rc={rc})")

    # --- surface the results ---
    gen = out_dir / name
    diffusion = gen / "diffusion0.mp4"
    print(f"[run_viewcrafter] outputs in {gen}")
    if diffusion.exists():
        dest = pkg / "diffusion0.mp4"
        shutil.copy2(diffusion, dest)
        print(f"[run_viewcrafter] realistic video -> {dest}")
    print("  render0.mp4   (conditioned point renders)")
    print("  diffusion0.mp4 (photo-realistic ViewCrafter output)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
