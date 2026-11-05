"""Stable-Diffusion inpaint over the injected region (route C, GPU server)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .backend import Package, assert_gauss_trace, placeholder_run, write_package


class SDInpaintBackend:
    name = "sd_inpaint"
    role = "harmonize"

    def available(self) -> tuple[bool, str]:
        return False, "run on the GPU server (torch + diffusers)"

    def prepare(self, cfg: dict, out_dir: str | Path, *, run_id: str,
                scene: str, frame: int, cam: str = "CAM_FRONT",
                strength: float = 0.35, steps: int = 30,
                prompt: str = "photorealistic object, matching scene lighting",
                **_kw) -> Package:
        import cv2

        from .. import paths
        from ..harmonize.masks import load_frame_cam

        batch_dir = paths.resolve(cfg.get("output_dir", "data/outputs")) / run_id
        frame_dir = batch_dir / scene / f"f{frame:03d}"
        gauss_png = frame_dir / "gauss" / f"{cam}.png"
        if not gauss_png.exists():
            raise FileNotFoundError(
                f"missing Gaussian frame {gauss_png} — conditions must come "
                "from the Gaussian renderer")
        rec = load_frame_cam(frame_dir, cam)
        if rec is None:
            raise FileNotFoundError(f"no composite artifacts for {cam} in {frame_dir}")

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        comp_p = out / "_comp_src.jpg"
        mask_p = out / "_mask_src.png"
        cv2.imwrite(str(comp_p), cv2.cvtColor(rec["comp_rgb"], cv2.COLOR_RGB2BGR))
        mask = (rec["visible"].astype(np.uint8)) * 255
        cv2.imwrite(str(mask_p), mask)
        cmd = (f"python scripts/gpu/run_sd_inpaint.py --package {out} "
               f"--strength {strength} --steps {steps} --prompt {prompt!r}")
        meta = {
            "condition_sources": [str(gauss_png), str(frame_dir / "composite" / f"{cam}.jpg")],
            "renderer": "realinject.gauss.raster_cpu",
            "params": {"strength": strength, "steps": steps, "prompt": prompt},
        }
        assert_gauss_trace(meta)
        pkg = write_package(
            out, backend=self.name, role=self.role, command=cmd,
            inputs={"composite": comp_p, "mask": mask_p}, meta=meta)
        comp_p.unlink(missing_ok=True)
        mask_p.unlink(missing_ok=True)
        return pkg

    def run_local(self, package: str | Path) -> dict:
        return placeholder_run(self.name)

    def collect(self, package: str | Path, out_dir: str | Path) -> dict:
        from .backend import read_package

        pkg = read_package(package)
        outputs = sorted((pkg.output).glob("*.png"))
        return {"status": "collected" if outputs else "pending",
                "outputs": [str(p) for p in outputs]}
