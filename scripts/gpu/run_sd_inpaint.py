#!/usr/bin/env python
"""GPU stub: SD inpaint over the injected region (NOT runnable locally).

Run on a GPU server after copying the project there — see
docs/GPU_DIFFUSION.md for the full workflow.

    python scripts/gpu/run_sd_inpaint.py --package data/outputs/diffusion/gs01/sd_inpaint
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", required=True)
    ap.add_argument("--model", default="runwayml/stable-diffusion-inpainting")
    ap.add_argument("--strength", type=float, default=0.35)
    ap.add_argument("--prompt", default="photorealistic object, matching scene lighting")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    # --- dependency guard: only imports torch/diffusers on a GPU box ---
    import importlib

    try:
        import torch
        importlib.import_module("diffusers")
    except ImportError as e:
        raise SystemExit(
            "sd_inpaint needs the GPU environment:\n"
            "  pip install torch diffusers transformers accelerate\n"
            "see docs/GPU_DIFFUSION.md") from e

    pkg = Path(args.package)
    comp = pkg / "inputs" / "composite.jpg"
    mask = pkg / "inputs" / "mask.png"
    if not (comp.exists() and mask.exists()):
        raise SystemExit(f"package incomplete: {pkg} "
                         f"(need inputs/composite.jpg + inputs/mask.png)")

    import cv2
    import numpy as np
    from diffusers import StableDiffusionInpaintPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe = pipe.to(device)
    image = cv2.cvtColor(cv2.imread(str(comp)), cv2.COLOR_BGR2RGB)
    mask_img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    out = pipe(prompt=args.prompt, image=image, mask_image=mask_img,
               num_inference_steps=args.steps, strength=args.strength).images[0]
    dest = pkg / "output" / "inpainted.png"
    dest.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(dest), cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR))
    print(f"[run_sd_inpaint] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
