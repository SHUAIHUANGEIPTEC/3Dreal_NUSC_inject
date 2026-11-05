"""Gaussian injection pipeline: one sample, N objects, all cameras.

Mirrors ``nusc_inject.pipeline.run_sample``'s step order (anchor -> LiDAR ->
camera -> composite -> validate -> spec) but the camera channel is the Gaussian
renderer and the compositor is ``composite_ext.composite_gauss`` (which takes
the splat depth directly). Everything else — placement anchor, range-image
occlusion, occluder boxes, annotation spec — is imported from the read-only
sibling library via ``bridge``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from ..assets.convert import mesh_to_gaussians, points_to_gaussians
from ..gauss.lidar_splat import cloud_to_lidar_payload
from ..gauss.model import GaussianCloud
from ..gauss.raster_cpu import RenderOut, render_gaussians
from ..gauss.render import save_rgba_png
from . import bridge, composite_ext, validate_ext

log = logging.getLogger("pipeline_gauss")


def load_cloud(path: str | Path, cfg: dict | None = None) -> GaussianCloud:
    """Load a canonical asset npz and build its GaussianCloud (by npz kind)."""
    with np.load(path, allow_pickle=True) as data:
        canon = {k: data[k] for k in data.files}
    if "xyz" in canon:
        return points_to_gaussians(canon, cfg)
    return mesh_to_gaussians(canon, cfg)


def merge_renders(renders: list[RenderOut]) -> RenderOut:
    """Merge per-object renders: nearest opaque surface wins color/depth."""
    if len(renders) == 1:
        return renders[0]
    h, w = renders[0].alpha.shape
    depth = np.full((h, w), np.inf, dtype=np.float64)
    owner = np.full((h, w), -1, dtype=np.int32)
    for i, r in enumerate(renders):
        m = r.mask & (r.depth < depth)
        depth[m] = r.depth[m]
        owner[m] = i
    alpha = np.ones((h, w), dtype=np.float64)
    for r in renders:
        alpha *= 1.0 - np.clip(r.alpha.astype(np.float64), 0.0, 1.0)
    alpha = 1.0 - alpha
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, r in enumerate(renders):
        m = owner == i
        rgb[m] = r.rgb[m]
    mask = alpha > 1e-4
    depth = np.where(mask, depth, np.inf)
    return RenderOut(
        rgb=rgb,
        alpha=alpha.astype(np.float32),
        depth=depth.astype(np.float32),
        mask=mask,
        info={"n_objects": len(renders),
              "n_px": int(mask.sum())},
    )


def run_sample(cfg: dict, meta, sample_index: int, objects: list,
               out_dir: Path, cloud_cache: dict | None = None,
               track_keys: list | None = None,
               drop_invisible: bool = False) -> dict:
    """Inject ``objects`` into one sample; returns the upstream-shaped result.

    objects: [{"model_cfg": {...}, "placement_cfg": {...}}] — model_cfg.path
    must point at a canonical asset npz (see assets/convert.py).
    """
    bridge.bootstrap(cfg)
    from nusc_inject import anchor, annotation, camera_channel, lidar_channel, nusc_io
    from nusc_inject import occluders as occ_mod

    out_dir = Path(out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    cloud_cache = {} if cloud_cache is None else cloud_cache
    track_keys = list(track_keys) if track_keys else [None] * len(objects)
    dropped: list[int] = []

    sample = meta.sample(int(sample_index))
    lidar_sd = meta.sd_for_sample(sample["token"], cfg["dataset"]["lidar_channel"])
    if not lidar_sd["filename"].startswith("samples/"):
        raise SystemExit(f"FATAL: not a keyframe lidar file: {lidar_sd['filename']}")
    scene_pts = nusc_io.load_lidar_bin(meta.sensor_path(lidar_sd))

    # ---- clouds (cached per model path) ----
    for o in objects:
        path = o["model_cfg"]["path"]
        if path not in cloud_cache:
            cloud_cache[path] = load_cloud(path, cfg)
            log.info("loaded cloud %s: %d gaussians", Path(path).name,
                     len(cloud_cache[path]))

    # ---- Step 1: placement anchor (single truth for both channels) ----
    gts = []
    for o in objects:
        cloud = cloud_cache[o["model_cfg"]["path"]]
        dims = o["model_cfg"].get("dims_lwh_m")
        if dims:
            raw_size = np.asarray(dims, dtype=float)
        else:
            lo, hi = cloud.bounds()
            raw_size = hi - lo
        raw_min = np.zeros(3)  # canonical frame is bbox-min at the origin
        gt = anchor.build_gt(
            cfg, meta, scene_pts, sample=sample,
            model_cfg=o["model_cfg"], placement_cfg=o["placement_cfg"],
            raw_bbox=(raw_min, raw_size),
            model_verts=np.asarray(cloud.xyz, dtype=float),
        )
        gts.append(gt)

    def _payload():
        return [cloud_to_lidar_payload(
            cloud_cache[o["model_cfg"]["path"]], gt, cfg)
            for o, gt in zip(objects, gts, strict=True)]

    # ---- scene occluders (dense bbox bodies for the occlusion z-buffers) ----
    t_l2g = np.array(gts[0]["T_lidar2global"], dtype=float).reshape(4, 4)
    occ_boxes = occ_mod.build_scene_occluder_boxes(meta, sample["token"], cfg, t_l2g)
    log.info("scene occluder boxes: %d", len(occ_boxes))

    # ---- Step 2: LiDAR injection (joint) ----
    lr = lidar_channel.inject(scene_pts, _payload(), cfg, occluder_boxes=occ_boxes)
    if drop_invisible and any(n == 0 for n in lr["num_obj_kept_per_object"]):
        failed = [i for i, n in enumerate(lr["num_obj_kept_per_object"]) if n == 0]
        keep = [i for i in range(len(objects)) if i not in failed]
        log.warning("invisible objects %s dropped (0 lidar pts)", failed)
        dropped.extend(failed)
        if not keep:
            return {"status": "no_visible", "dropped": dropped,
                    "sample": sample, "out_dir": out_dir}
        objects = [objects[i] for i in keep]
        gts = [gts[i] for i in keep]
        track_keys = [track_keys[i] for i in keep]
        lr = lidar_channel.inject(scene_pts, _payload(), cfg, occluder_boxes=occ_boxes)
    if any(n == 0 for n in lr["num_obj_kept_per_object"]):
        if drop_invisible:
            return {"status": "no_visible", "dropped": dropped,
                    "sample": sample, "out_dir": out_dir}
        raise SystemExit(f"FATAL: an object survived with 0 points: "
                         f"{lr['num_obj_kept_per_object']}")

    fmt = cfg["lidar_channel"].get("out_format", "n5")
    bin_name = "injected.pcd.bin" if fmt == "n5" else "injected.bin"
    nusc_io.save_lidar_bin(out_dir / "lidar" / bin_name, lr["injected"])
    np.save(out_dir / "lidar" / "obj_mask.npy", lr["obj_mask"])
    np.save(out_dir / "lidar" / "obj_points_lidar.npy", lr["obj_points_lidar"])
    np.save(out_dir / "lidar" / "obj_ids.npy", lr["obj_ids"])
    log.info("LiDAR: scene %d + obj %d (per-object %s)",
             lr["num_scene_kept"], lr["num_obj_kept"],
             lr["num_obj_kept_per_object"])

    with open(out_dir / "gt.json", "w") as f:
        json.dump(gts if len(gts) > 1 else gts[0], f, indent=2)

    # ---- Step 3: camera render (Gaussian splat per in-frustum camera) ----
    try:
        cams = camera_channel.build_job(cfg, meta, gts, out_dir)
    except RuntimeError as e:
        if drop_invisible:
            log.warning("no camera sees objects: %s", e)
            return {"status": "no_camera", "dropped": dropped,
                    "sample": sample, "out_dir": out_dir}
        raise
    if not cams:
        if drop_invisible:
            return {"status": "no_camera", "dropped": dropped,
                    "sample": sample, "out_dir": out_dir}
        raise RuntimeError("objects not visible in any camera")
    cam_specs = {c["name"]: c for c in cams}
    cam_result = {"cams": cams, "cam_specs": cam_specs, "max_pixel_err": 0.0}

    gauss_dir = out_dir / "gauss"
    depth_dir = out_dir / "depth"
    gauss_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    per_cam: dict = {}
    for c in cams:
        name = c["name"]
        cam_spec = cam_specs[name]
        renders = []
        for o, gt in zip(objects, gts, strict=True):
            cloud = cloud_cache[o["model_cfg"]["path"]]
            t_m2g = np.array(gt["T_model2global"], dtype=float).reshape(4, 4)
            renders.append(render_gaussians(cam_spec, cloud, t_m2g, cfg))
        merged = merge_renders(renders)
        save_rgba_png(gauss_dir / f"{name}.png", merged)
        np.save(depth_dir / f"{name}_obj_depth.npy", merged.depth)

        sd = meta.sd_for_sample(sample["token"], name)
        bg = cv2.cvtColor(cv2.imread(str(meta.sensor_path(sd))), cv2.COLOR_BGR2RGB)
        fg_rgba = np.dstack([
            merged.rgb,
            (np.clip(merged.alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
        ])
        out_rgb, visible, alpha, obj_depth, scene_depth, info = (
            composite_ext.composite_gauss(
                name, fg_rgba, bg, merged.depth, cam_spec, cfg, scene_pts,
                T_lidar2global=t_l2g, occluder_boxes=occ_boxes))

        min_alpha = int(cfg.get("validation", {}).get("min_alpha_px", 500))
        if info["n_alpha_px"] < min_alpha:
            log.info("  %s: alpha px=%d < %d, skip camera", name,
                     info["n_alpha_px"], min_alpha)
            continue
        (out_dir / "composite").mkdir(exist_ok=True)
        cv2.imwrite(str(out_dir / "composite" / f"{name}.jpg"),
                    cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
        np.save(depth_dir / f"{name}_scene_depth.npy", scene_depth)
        per_cam[name] = {"cam_spec": cam_spec, "alpha": alpha, "visible": visible,
                         "n_visible_px": info["n_visible_px"],
                         "obj_depth": obj_depth, "scene_depth": scene_depth}
        log.info("  %s: visible=%d alpha=%d occluded=%d (%.2fs)",
                 name, info["n_visible_px"], info["n_alpha_px"],
                 info["n_occluded_px"], merged.info.get("t_sec", -1))
    if not per_cam:
        if drop_invisible:
            return {"status": "no_camera", "dropped": dropped,
                    "sample": sample, "out_dir": out_dir}
        raise SystemExit("FATAL: no camera composited")

    # ---- Step 4: validation (upstream + gaussian-specific) ----
    report = validate_ext.run_ext(cfg, gts, lr, cam_result, per_cam)
    with open(out_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    log.info("validation all_pass=%s", report["all_pass"])
    if not report["all_pass"] and drop_invisible:
        return {"status": "validation_failed", "report": report,
                "dropped": dropped, "sample": sample, "out_dir": out_dir}

    # ---- Step 5: sample spec for the annotation export ----
    files = {
        "lidar_filename": lidar_sd["filename"],
        "lidar_output_path": str(out_dir / "lidar" / bin_name),
        "cam_filenames": {name: meta.sd_for_sample(sample["token"], name)["filename"]
                          for name in per_cam},
        "cam_output_paths": {name: str(out_dir / "composite" / f"{name}.jpg")
                             for name in per_cam},
    }
    spec = annotation.build_sample_spec(sample, gts, lr, files,
                                        track_keys=track_keys)
    with open(out_dir / "spec.json", "w") as f:
        json.dump(spec, f, indent=2)
    return {"status": "ok", "sample": sample, "gts": gts, "lidar_result": lr,
            "cam_result": cam_result, "per_cam": per_cam, "report": report,
            "spec": spec, "out_dir": out_dir, "dropped": dropped}
