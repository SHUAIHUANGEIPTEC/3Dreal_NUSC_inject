"""ViewCrafter novel-view synthesis (route A, GPU server)."""
from __future__ import annotations

from pathlib import Path

from .backend import Package, assert_gauss_trace, placeholder_run, write_package

VC_W, VC_H = 1024, 576  # ViewCrafter_25 native resolution

# phi (yaw deg), theta (pitch deg), r (relative) keypoint sequences
PRESETS: dict[str, dict[str, list[float]]] = {
    "rotate": {"phi": [0, 15, 30, 15, 0], "theta": [0, 0, 0, 0, 0],
               "r": [0, 0, 0, 0, 0]},
    "elevate": {"phi": [0, 0, 0, 0, 0], "theta": [0, -6, -12, -6, 0],
                "r": [0, 0, 0, 0, 0]},
    "loop": {"phi": [0, 8, 16, 24], "theta": [0, 0, 0, 0], "r": [0, 0, 0, 0]},
}


def _traj_text(preset: str) -> str:
    p = PRESETS[preset]
    return "\n".join(" ".join(f"{v:g}" for v in seq)
                     for seq in (p["phi"], p["theta"], p["r"])) + "\n"


def pick_reference(frame_dir: Path, cam: str) -> Path:
    """refined -> composite -> fail (the reference MUST contain the object)."""
    for sub in ("refined", "composite"):
        p = frame_dir / sub / f"{cam}.jpg"
        if p.exists():
            return p
    raise FileNotFoundError(f"no refined/composite image for {cam} in {frame_dir}")


class ViewCrafterBackend:
    name = "viewcrafter"
    role = "realism"

    def available(self) -> tuple[bool, str]:
        return False, ("run on the GPU server (third_party/ViewCrafter, "
                       "weights shared with stag1_inject)")

    def prepare(self, cfg: dict, out_dir: str | Path, *, run_id: str,
                scene: str, frame: int, cam: str = "CAM_FRONT",
                preset: str = "rotate", **_kw) -> Package:
        import cv2

        from .. import paths

        batch_dir = paths.resolve(cfg.get("output_dir", "data/outputs")) / run_id
        frame_dir = batch_dir / scene / f"f{frame:03d}"
        gauss_png = frame_dir / "gauss" / f"{cam}.png"
        if not gauss_png.exists():
            raise FileNotFoundError(
                f"missing Gaussian frame {gauss_png} — conditions must come "
                "from the Gaussian renderer (run_inject_gauss.py)")
        ref_src = pick_reference(frame_dir, cam)
        img = cv2.imread(str(ref_src))
        if img is None:
            raise FileNotFoundError(f"cannot read {ref_src}")
        ref = cv2.resize(img, (VC_W, VC_H), interpolation=cv2.INTER_AREA)

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ref_path = out / "_ref_src.jpg"  # staged, copied into inputs by write_package
        cv2.imwrite(str(ref_path), ref)
        cmd = (f"python scripts/gpu/run_viewcrafter.py "
               f"--package {out} --viewcrafter-root third_party/ViewCrafter")
        meta = {
            "condition_sources": [str(gauss_png), str(ref_src)],
            "renderer": "realinject.gauss.raster_cpu",
            "params": {"preset": preset, "resolution": [VC_W, VC_H],
                       "elevation": 5.0, "center_scale": 1.0},
            "gpu": {"weights": "stag1_inject/third_party/ViewCrafter/checkpoints"},
        }
        assert_gauss_trace(meta)
        pkg = write_package(
            out, backend=self.name, role=self.role, command=cmd,
            inputs={"ref": ref_path, "traj": _traj_text(preset)}, meta=meta)
        ref_path.unlink(missing_ok=True)
        return pkg

    def run_local(self, package: str | Path) -> dict:
        return placeholder_run(self.name)

    def collect(self, package: str | Path, out_dir: str | Path) -> dict:
        from .backend import read_package

        pkg = read_package(package)
        outputs = sorted((pkg.output).glob("*.mp4"))
        return {"status": "collected" if outputs else "pending",
                "outputs": [str(p) for p in outputs]}
