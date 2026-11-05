"""Official Stag diffusion core (route B1, GPU server).

Conditioning = our GT geometry rendered by the **Gaussian renderer** along a
turntable orbit around the injected object (25 frames @1024x576, frame 0 =
harmonized reference photo — official ``nvs_single_view`` behaviour). The
background is held static (the original scene image); swap in stag3d
full-scene renders once the 4D pipeline is ported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .backend import Package, assert_gauss_trace, placeholder_run, write_package

VC_W, VC_H = 1024, 576
VIDEO_LEN = 25


def _orbit_cam(base_cam: dict, center: np.ndarray, phi_deg: float,
               elevation_deg: float = 5.0, dist_scale: float = 1.0) -> dict:
    """Look-at camera orbiting ``center`` at the base pose's distance."""
    t = np.asarray(base_cam["T_cam2global"], dtype=float).reshape(4, 4)
    eye0 = t[:3, 3]
    dist = float(np.linalg.norm(eye0 - center)) * dist_scale
    # azimuth of the original view direction, then sweep by phi
    d0 = center - eye0
    azimuth = float(np.arctan2(d0[1], d0[0]))
    az = azimuth + np.deg2rad(phi_deg)
    el = np.deg2rad(elevation_deg)
    eye = center + dist * np.array([np.cos(el) * np.cos(az),
                                    np.cos(el) * np.sin(az),
                                    np.sin(el)])
    f = center - eye
    f = f / np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x = x / np.linalg.norm(x)
    y = np.cross(f, x)
    out = np.eye(4)
    out[:3, :3] = np.column_stack([x, y, f])
    out[:3, 3] = eye
    cam = dict(base_cam)
    cam["T_cam2global"] = out.tolist()
    return cam


class StagB1Backend:
    name = "stagg_b1"
    role = "realism"

    def available(self) -> tuple[bool, str]:
        return False, ("run on the GPU server (third_party/stag_run, "
                       "weights shared with stag1_inject)")

    def prepare(self, cfg: dict, out_dir: str | Path, *, run_id: str,
                scene: str, frame: int, cam: str = "CAM_FRONT",
                phi_sweep_deg: float = 15.0, **_kw) -> Package:
        import cv2

        from .. import paths
        from ..assets.convert import record_to_cloud
        from ..assets.manifest import AssetLibrary
        from ..gauss.raster_cpu import render_gaussians

        batch_dir = paths.resolve(cfg.get("output_dir", "data/outputs")) / run_id
        frame_dir = batch_dir / scene / f"f{frame:03d}"
        gauss_png = frame_dir / "gauss" / f"{cam}.png"
        if not gauss_png.exists():
            raise FileNotFoundError(
                f"missing Gaussian frame {gauss_png} — conditions must come "
                "from the Gaussian renderer")
        gt_path = frame_dir / "gt.json"
        import json

        gt = json.loads(gt_path.read_text())
        gt = gt[0] if isinstance(gt, list) else gt
        t_m2g = np.array(gt["T_model2global"], dtype=float).reshape(4, 4)
        center = t_m2g[:3, 3] + t_m2g[:3, :3] @ (
            np.asarray(gt.get("raw_bbox_size", [0, 0, 0]), float) / 2.0)

        # camera spec of the frame (frustum-culled set; use the named cam)
        from nusc_inject import camera_channel, nusc_io

        meta = nusc_io.NuScenesMini(cfg["dataset"]["root"], cfg["dataset"]["version"])
        sample_token = gt["sample_token"]
        cams = camera_channel.build_job(cfg, meta, [gt], frame_dir)
        if not cams:
            raise FileNotFoundError(f"no camera in frustum for {frame_dir}")
        base_cam = next((c for c in cams if c["name"] == cam), cams[0])

        # cloud: the injected asset's canonical npz -> GaussianCloud
        library_root = paths.resolve((cfg.get("library") or {}).get("path",
                                                                    "data/library"))
        lib = AssetLibrary.load(library_root / "assets_manifest.json")
        asset_id = None
        for name in ("spec.json", "gt.json"):
            src = frame_dir / name
            if not src.exists():
                continue
            data = json.loads(src.read_text())
            data = data if isinstance(data, list) else [data]
            for o in data:
                if o.get("model_path"):
                    asset_id = Path(o["model_path"]).stem
                    break
            if asset_id:
                break
        record = next((r for r in lib.records
                       if r.canonical_path.endswith(f"{asset_id}.npz")), None)
        if record is None:
            raise FileNotFoundError(f"cannot resolve injected asset for {frame_dir}")
        cloud = record_to_cloud(record, library_root,
                                {"gaussian_init": {"mesh_samples": 60_000}})

        # original scene image as static background
        sd = meta.sd_for_sample(sample_token, cam)
        bg = cv2.cvtColor(cv2.imread(str(meta.sensor_path(sd))), cv2.COLOR_BGR2RGB)
        scale = VC_W / bg.shape[1]
        bg_small = cv2.resize(bg, (VC_W, VC_H), interpolation=cv2.INTER_AREA)

        # turntable orbit: phi in [-sweep, +sweep], 25 frames
        phis = np.linspace(-phi_sweep_deg, phi_sweep_deg, VIDEO_LEN)
        frames = np.zeros((VIDEO_LEN, VC_H, VC_W, 3), dtype=np.float32)
        for i, phi in enumerate(phis):
            orb_cam = _orbit_cam(base_cam, center, float(phi))
            # resize intrinsics to 1024x576
            k = np.asarray(orb_cam["K"], dtype=float)
            k[:2] *= scale
            orb_cam = dict(orb_cam)
            orb_cam["K"] = k.tolist()
            orb_cam["width"], orb_cam["height"] = VC_W, VC_H
            out = render_gaussians(orb_cam, cloud, t_m2g, cfg)
            alpha = out.alpha[..., None]
            comp = out.rgb.astype(np.float32) * alpha + bg_small * (1 - alpha)
            frames[i] = comp / 255.0

        # frame 0 = harmonized reference photo (official nvs_single_view)
        from .realism_viewcrafter import pick_reference

        ref_src = pick_reference(frame_dir, cam)
        ref = cv2.imread(str(ref_src))
        ref = cv2.resize(ref, (VC_W, VC_H), interpolation=cv2.INTER_AREA)
        frames[0] = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ref_path = out / "_ref_src.jpg"
        cv2.imwrite(str(ref_path), ref)
        cmd = (f"python scripts/gpu/run_stag_diffusion.py --package {out} "
               f"--stag-root third_party/stag_run")
        meta_out = {
            "condition_sources": [str(gauss_png), str(ref_src),
                                  str(library_root / record.canonical_path)],
            "renderer": "realinject.gauss.raster_cpu",
            "params": {"video_length": VIDEO_LEN, "resolution": [VC_W, VC_H],
                       "phi_sweep_deg": phi_sweep_deg,
                       "background": "static original scene image",
                       "frame0": "harmonized reference photo"},
            "gpu": {"weights": "stag1_inject/third_party/stag_run/checkpoints"},
        }
        assert_gauss_trace(meta_out)
        pkg = write_package(
            out, backend=self.name, role=self.role, command=cmd,
            inputs={"renderings": frames, "ref": ref_path}, meta=meta_out)
        ref_path.unlink(missing_ok=True)
        # preview of the raw conditioning sequence
        preview = (frames * 255).astype(np.uint8)
        try:
            from ..stag.video import write_mp4

            write_mp4(preview, out / "render0.mp4", fps=8)
        except Exception:  # noqa: BLE001 - preview is best-effort
            pass
        return pkg

    def run_local(self, package: str | Path) -> dict:
        return placeholder_run(self.name)

    def collect(self, package: str | Path, out_dir: str | Path) -> dict:
        from .backend import read_package

        pkg = read_package(package)
        outputs = sorted((pkg.output).glob("*.mp4"))
        return {"status": "collected" if outputs else "pending",
                "outputs": [str(p) for p in outputs]}
