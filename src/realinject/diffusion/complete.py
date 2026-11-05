"""Asset completion (route D, GPU server): half-shell / hole-fill via 3D generative models.

Inputs per completion candidate:
- AV2: the track's conditioning view image + half-shell points (canonical npz)
- realcar: a rendered orbit of the filled mesh + the mesh itself (Poisson-filled)

Backends named in docs/GPU_DIFFUSION.md: TRELLIS (native gaussian output),
Hunyuan3D-2 (best texture), GINA-3D-style single-view (image + half-shell).
The package records which model to use; the GPU script dispatches.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .backend import Package, placeholder_run, write_package

DEFAULT_MODEL = "trellis"  # trellis | hunyuan3d2 | gina


class CompletionBackend:
    name = "complete"
    role = "complete"

    def available(self) -> tuple[bool, str]:
        return False, "run on the GPU server (TRELLIS / Hunyuan3D-2 / GINA-class)"

    def prepare(self, cfg: dict, out_dir: str | Path, *, run_id: str = "",
                model: str = DEFAULT_MODEL, **_kw) -> Package:
        from .. import paths
        from ..assets.manifest import AssetLibrary

        library_root = paths.resolve((cfg.get("library") or {}).get("path",
                                                                    "data/library"))
        lib = AssetLibrary.load(library_root / "assets_manifest.json")
        candidates = [r for r in lib.passed() if r.completion.get("candidate")]
        if not candidates:
            raise FileNotFoundError(
                "no completion candidates in the library "
                "(DOG tracks with az_span >= completion_min_az_span_deg)")

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "inputs" / "candidates").mkdir(parents=True, exist_ok=True)
        rows = []
        for rec in candidates:
            view = rec.conditioning.get("view_image")
            if not view:
                continue
            src = Path(str((cfg.get("assets") or {}).get("root", ""))) / view
            npz = library_root / rec.canonical_path
            if not (src.exists() and npz.exists()):
                continue
            with np.load(npz) as data:
                pts = np.asarray(data["xyz"], dtype=np.float32)
            dst = out / "inputs" / "candidates" / f"{rec.asset_id.replace('/', '_')}"
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "view.png").write_bytes(src.read_bytes())
            np.save(dst / "half_shell.npy", pts)
            rows.append({"asset_id": rec.asset_id, "category": rec.category,
                         "dims_lwh_m": list(rec.dims_lwh_m),
                         "dir": str(dst.relative_to(out))})
        if not rows:
            raise FileNotFoundError("completion candidates lack conditioning images")

        cmd = (f"python scripts/gpu/run_asset_completion.py --package {out} "
               f"--model {model}")
        meta = {
            "condition_sources": [r["dir"] + "/view.png" for r in rows],
            "renderer": "realinject.assets (canonical library)",
            "params": {"model": model, "n_candidates": len(rows)},
            "gpu": {"weights": "TRELLIS / Hunyuan3D-2 (install on server)"},
            "candidates": rows,
        }
        return write_package(out, backend=self.name, role=self.role, command=cmd,
                             inputs={"manifest": json.dumps(rows, indent=2)},
                             meta=meta)

    def run_local(self, package: str | Path) -> dict:
        return placeholder_run(self.name)

    def collect(self, package: str | Path, out_dir: str | Path) -> dict:
        from .backend import read_package

        pkg = read_package(package)
        done = list((pkg.path / "output").rglob("*.npz")) + \
            list((pkg.path / "output").rglob("*.glb"))
        return {"status": "collected" if done else "pending",
                "outputs": [str(p) for p in done],
                "note": "completed assets land in data/library/completed/"}
