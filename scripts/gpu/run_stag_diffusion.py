#!/usr/bin/env python
"""Preflight + launch for the STRICT official Stag route (B1) on a GPU server.

Assembled tree (done locally, CPU): third_party/stag_run/  =
  Stag-main (official du3d3d.py + setup_diffusion restored)
  + ViewCrafter utils/ lvdm/ extern/dust3r/ configs/inference_pvd_1024.yaml

Weights (one download, shared with path A):
  third_party/stag_run/checkpoints/
    model_sparse.ckpt                              <- HF Drexubery/ViewCrafter_25_sparse
    DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth           <- naverlabs link

Usage (GPU server, ViewCrafter conda env):
  python scripts/gpu/run_stag_diffusion.py \
      --package data/outputs/stag_diffusion/baseline_freeze_time
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import paths  # noqa: E402

STAG_RUN = ROOT / "third_party" / "stag_run"
CKPT_SPARSE = "model_sparse.ckpt"
CKPT_PLAIN = "model.ckpt"
DUST3R = "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
DL_SPARSE = ("wget https://huggingface.co/Drexubery/ViewCrafter_25_sparse/"
             "resolve/main/model_sparse.ckpt -P third_party/stag_run/checkpoints/")
DL_DUST3R = ("wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/"
             "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth "
             "-P third_party/stag_run/checkpoints/")


def fail(msg: str) -> int:
    print(f"[run_stag_b1] ERROR: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--python", default=sys.executable,
                    help="python of the ViewCrafter conda env")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    pkg = paths.resolve(args.package)
    if not (pkg / "inputs" / "renderings.npy").exists():
        return fail(f"missing inputs/renderings.npy in {pkg} "
                    f"(run scripts/prepare_diffusion.py --route stagg_b1 first)")

    # --- CUDA (under the ViewCrafter env's python) ---
    if subprocess.call([args.python, "-c",
                        "import torch,sys; sys.exit(0 if "
                        "torch.cuda.is_available() else 1)"]) != 0:
        return fail("CUDA not available — GPU server + ViewCrafter env required")

    # --- assembled tree ---
    missing = [p for p in ("du3d3d.py", "utils/pvd_utils.py",
                           "utils/diffusion_utils.py", "lvdm",
                           "extern/dust3r/dust3r",
                           "configs/inference_pvd_1024.yaml")
               if not (STAG_RUN / p).exists()]
    if missing:
        return fail("stag_run tree incomplete: " + ", ".join(missing) +
                    "\n  (assembly lives in the repo — re-copy third_party/)")

    # --- weights ---
    ck = STAG_RUN / "checkpoints"
    ckpt = CKPT_SPARSE if (ck / CKPT_SPARSE).exists() else CKPT_PLAIN
    if not (ck / ckpt).exists():
        return fail(f"missing {ck / ckpt}\n  {DL_SPARSE}\n"
                    f"  (model.ckpt fallback: HF Drexubery/ViewCrafter_25)")
    if not (ck / DUST3R).exists():
        return fail(f"missing {ck / DUST3R}\n  {DL_DUST3R}")

    # --- launch driver with cwd=stag_run (official relative paths) ---
    driver = ROOT / "scripts" / "gpu" / "_stag_b1_driver.py"
    cmd = [args.python, str(driver), "--package", str(pkg),
           "--ckpt", ckpt, "--device", args.device]
    print(f"[run_stag_b1] cwd={STAG_RUN}\n[run_stag_b1] {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(STAG_RUN))
    if rc != 0:
        return fail(f"B1 driver failed (rc={rc})")

    out = pkg / "output" / "diffusion0.mp4"
    print(f"[run_stag_b1] {'OK -> ' + str(out) if out.exists() else 'driver done (check output)'}")
    print("  render0.mp4   = our GT conditioning (with injected object)")
    print("  diffusion0.mp4 = STRICT official Stag diffusion output")
    return 0 if out.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
