"""Generative insertion (route E, MagicDrive bbox_add) — condition package only.

Consumes the injected nuScenes-format bboxes + camera parameters from a batch's
specs.json; output replaces the composite in the object region only. Requires a
trained MagicDrive checkpoint on the GPU server (out of scope for the CPU stub).
"""
from __future__ import annotations

import json
from pathlib import Path

from .backend import Package, placeholder_run, write_package


class MagicDriveBackend:
    name = "magicdrive"
    role = "generate"

    def available(self) -> tuple[bool, str]:
        return False, "run on the GPU server (MagicDrive checkpoint required)"

    def prepare(self, cfg: dict, out_dir: str | Path, *, run_id: str,
                scene: str, **_kw) -> Package:
        from .. import paths

        batch_dir = paths.resolve(cfg.get("output_dir", "data/outputs")) / run_id
        specs_p = batch_dir / "specs.json"
        if not specs_p.exists():
            raise FileNotFoundError(f"{specs_p} missing — run run_inject_gauss.py first")
        specs = json.loads(specs_p.read_text())
        scene_specs = specs.get("specs", [])
        if not scene_specs:
            raise FileNotFoundError(f"no specs in {specs_p}")

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cmd = (f"python scripts/gpu/run_magicdrive.py --package {out} "
               f"--scene {scene}")
        meta = {
            "condition_sources": [str(specs_p)],
            "renderer": "n/a (bbox-conditioned generation)",
            "params": {"scene": scene, "n_samples": len(scene_specs)},
            "gpu": {"weights": "MagicDrive checkpoint (train first)"},
        }
        return write_package(out, backend=self.name, role=self.role, command=cmd,
                             inputs={"specs": json.dumps(
                                 {"run_id": run_id, "scene": scene,
                                  "specs": scene_specs}, indent=2)},
                             meta=meta)

    def run_local(self, package: str | Path) -> dict:
        return placeholder_run(self.name)

    def collect(self, package: str | Path, out_dir: str | Path) -> dict:
        from .backend import read_package

        pkg = read_package(package)
        outputs = sorted((pkg.output).glob("*.mp4")) + \
            sorted((pkg.output).glob("*.png"))
        return {"status": "collected" if outputs else "pending",
                "outputs": [str(p) for p in outputs]}
