#!/usr/bin/env python
"""B1 driver — executed by run_stag_diffusion.py with cwd=third_party/stag_run.

Official Stag route, B1 flavour:
  Du3d3d(opts)            -> setup_dust3r() (official __init__)
  obj.setup_diffusion()   -> method restored from ViewCrafter (missing in Stag release)
  obj.run_diffusion(x)    -> official call: image_guided_synthesis(...) 15-arg compatible
  save_video(...)         -> official utils.pvd_utils

Input: <package>/inputs/renderings.npy  [25,576,1024,3] float32 in [0,1], frame0 = ref photo.
"""
from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


def build_opts(args) -> Namespace:
    # defaults transcribed from Stag configs/train_config.py (official values)
    return Namespace(
        device=args.device,
        # image/dust3r (required by Du3d3d.__init__ setup_dust3r)
        image_dir="", model_path="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        batch_size=1, schedule="linear", niter=300, lr=0.01, min_conf_thr=3.0,
        # diffusion (official ViewCrafter/Stag inference values)
        ckpt_path=f"./checkpoints/{args.ckpt}",
        config="./configs/inference_pvd_1024.yaml",
        height=576, width=1024, video_length=25, bs=1,
        ddim_steps=50, ddim_eta=1.0,
        unconditional_guidance_scale=7.5, cfg_img=None,
        frame_stride=10, text_input=True, multiple_cond_cfg=False,
        timestep_spacing="uniform_trailing", guidance_rescale=0.7,
        prompt="Rotating view of a scene",   # official: "Fixed"
        seed=123, perframe_ae=True, n_samples=1,
        # unused by the B1 path (nvs_* methods only) but referenced elsewhere
        mode="single_view_txt", traj_txt=None,
        d_theta=10, d_phi=30, d_r=-0.2, d_x=0.0, d_y=0.0,
        mask_image=False, mask_pc=True, reduce_pc=False,
        bg_trd=0.0, dpt_trd=1.0, elevation=5.0, center_scale=1.0,
        out_dir="./output", exp_name="b1",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--ckpt", default="model_sparse.ckpt",
                    choices=["model_sparse.ckpt", "model.ckpt"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    pkg = Path(args.package).resolve()
    renderings = np.load(pkg / "inputs" / "renderings.npy")
    assert renderings.shape == (25, 576, 1024, 3), renderings.shape
    assert renderings.min() >= 0.0 and renderings.max() <= 1.0
    print(f"[b1] renderings {renderings.shape} "
          f"[{renderings.min():.2f},{renderings.max():.2f}] "
          f"frame0=ref({'yes' if (pkg / 'inputs' / 'ref.png').exists() else '?'})")

    # cwd is stag_run: './extern/dust3r', './checkpoints', './configs' resolve
    from du3d3d import Du3d3d  # noqa: E402  (official module)

    opts = build_opts(args)
    obj = Du3d3d(opts)                  # official __init__ (setup_dust3r)
    obj.setup_diffusion()               # restored from ViewCrafter

    x = torch.from_numpy(renderings)    # (T,H,W,3) float32
    with torch.no_grad(), torch.cuda.amp.autocast():
        out = obj.run_diffusion(x)      # official call -> (T,H,W,3) in [-1,1]
    out = out.clamp(-1.0, 1.0)

    from utils.pvd_utils import save_video
    save_video((out + 1.0) / 2.0, str(pkg / "output" / "diffusion0.mp4"))
    print(f"[b1] saved {pkg / 'output' / 'diffusion0.mp4'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
