"""Composite the Gaussian-rendered foreground over the nuScenes background.

Same math as ``nusc_inject.composite.composite_onto`` (straight alpha blend,
depth-based visibility with constant + range-proportional margin), but the
object depth comes **from the Gaussian rasterizer** instead of being re-derived
from sparse surface samples — our splat depth already has the holes filled, so
re-deriving would regress it. Scene depth (LiDAR + occluder boxes) and the
depth-hole/gap helpers are reused from upstream.
"""
from __future__ import annotations

import numpy as np


def composite_gauss(name: str, fg_rgba: np.ndarray, bg_rgb: np.ndarray,
                    obj_depth: np.ndarray, cam_spec: dict, cfg: dict,
                    scene_pts_n5: np.ndarray,
                    T_lidar2global: np.ndarray | None = None,
                    occluder_boxes: list | None = None):
    """Composite one camera.

    fg_rgba: (H, W, 4) straight-alpha foreground from the Gaussian renderer
    obj_depth: (H, W) float32 object depth, inf = empty (from the renderer)
    Returns (uint8 RGB, visible, alpha, obj_depth, scene_depth, info) — the
    exact tuple ``composite_onto`` returns, so validate/per-camera code is
    interchangeable.
    """
    from nusc_inject import composite as _up
    from nusc_inject import occluders as occ_mod
    from nusc_inject import transforms as tf

    comp = cfg["composite"]
    alpha_thr = float(comp.get("alpha_threshold", 0.05))
    margin_base = float(comp.get("depth_margin_m", 0.2))
    margin_rel = float(comp.get("depth_margin_rel", 0.02))

    height, width = bg_rgb.shape[:2]
    k = np.array(cam_spec["K"], dtype=float)
    t_g2c = np.linalg.inv(
        np.array(cam_spec["T_cam2global"], dtype=float).reshape(4, 4))
    if T_lidar2global is None:
        t_l2g = np.eye(4)
    else:
        t_l2g = np.asarray(T_lidar2global, dtype=float).reshape(4, 4)
    t_l2c = t_g2c @ t_l2g  # lidar -> global -> camera

    obj_depth = np.asarray(obj_depth, dtype=np.float32)
    if obj_depth.shape != (height, width):
        raise ValueError(f"obj_depth shape {obj_depth.shape} != {(height, width)}")

    scene_cam = tf.transform_pts(t_l2c, scene_pts_n5[:, :3])
    scene_depth = _up._zbuffer_depth(scene_cam, k, width, height)
    scene_depth = _up._dilate_scene_depth(
        scene_depth, _up._fill_radius_px(k, occ_mod.scene_gap_fill_deg(cfg)))
    if occluder_boxes:
        scene_depth = np.minimum(
            scene_depth, occ_mod.box_depth_raster(occluder_boxes, k, t_l2c,
                                                  width, height))

    fg = fg_rgba.astype(np.float32) / 255.0
    if fg.shape[0] != height or fg.shape[1] != width:
        raise ValueError(f"foreground size {fg.shape[:2]} != background {(height, width)}")
    alpha = fg[:, :, 3] if fg.shape[2] == 4 else np.ones((height, width), np.float32)

    # the splat depth can still miss a pixel inside the alpha region (thin
    # edges); fill those holes first so the depth test stays meaningful
    obj_depth = _up._fill_depth_holes(obj_depth, alpha > alpha_thr)
    unknown_obj = ~np.isfinite(obj_depth)
    margin = margin_base + margin_rel * np.where(np.isfinite(obj_depth), obj_depth, 0.0)
    visible = (alpha > alpha_thr) & (
        unknown_obj | (obj_depth <= scene_depth + margin))

    bg = bg_rgb.astype(np.float32) / 255.0
    blended = fg[:, :, :3] * alpha[:, :, None] + bg * (1.0 - alpha[:, :, None])
    out = np.where(visible[:, :, None], blended, bg)

    info = {
        "name": name,
        "n_visible_px": int(visible.sum()),
        "n_alpha_px": int((alpha > alpha_thr).sum()),
        "n_occluded_px": int(((alpha > alpha_thr) & ~visible).sum()),
    }
    return (out * 255.0).astype(np.uint8), visible, alpha, obj_depth, scene_depth, info
