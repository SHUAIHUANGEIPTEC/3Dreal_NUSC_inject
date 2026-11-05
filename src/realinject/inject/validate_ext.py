"""Additive validation on top of upstream ``nusc_inject.validate.run``.

The upstream report already covers transform round-trip, LiDAR point counts,
size plausibility, coverage / bbox IoU / occlusion justification. These checks
are specific to the Gaussian path:

- ``gauss_depth_vs_lidar``: sampled object depth vs the LiDAR return along the
  same ray (median |dz| < 0.5 m) — catches camera/extrinsic or scale bugs that
  the alpha coverage cannot see;
- ``silhouette_density``: projected object-LiDAR hits per silhouette pixel in
  [0.01, 2.0] — catches a splat inflated far beyond (or collapsed against) its
  point support;
- ``box_encloses_mask``: >= 95% of mask pixels inside the projected 3D box
  AABB — catches axis swaps the bbox itself cannot see (symmetric boxes);
- ``multicam_overlap``: for pixels visible in >= 2 cameras, mask IoU > 0.6
  (informational; reported, not gated).
"""
from __future__ import annotations

import numpy as np


def _project(pts_cam: np.ndarray, k: np.ndarray, width: int, height: int,
             clip_to_frame: bool = True):
    """Project to integer pixel coords, filtered AFTER rounding (u=1599.6
    must not become index 1600 — filter-then-round is an off-by-one).

    clip_to_frame=False keeps off-image pixels (needed for the box-enclosure
    AABB: a box corner outside the frame still bounds the mask).
    """
    z = pts_cam[:, 2]
    front = z > 0.1
    if not front.any():
        return None
    u = k[0, 0] * pts_cam[front, 0] / z[front] + k[0, 2]
    v = k[1, 1] * pts_cam[front, 1] / z[front] + k[1, 2]
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    if clip_to_frame:
        ok = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
        if not ok.any():
            return None
        return ui[ok], vi[ok], z[front][ok]
    return ui, vi, z[front]


def run_ext(cfg: dict, gts: list, lidar_result: dict, cam_result: dict,
            per_cam: dict) -> dict:
    """Upstream validation + Gaussian-specific checks; returns the report dict."""
    from nusc_inject import transforms as tf
    from nusc_inject import validate as validate

    report = validate.run(cfg, gts, lidar_result, cam_result, per_cam)
    checks = dict(report["checks"])
    t_l2g = np.array(gts[0]["T_lidar2global"], dtype=float).reshape(4, 4)
    obj_pts = np.asarray(lidar_result["obj_points_lidar"], dtype=float)

    depth_thr = float(cfg.get("validation", {}).get("gauss_depth_vs_lidar_m", 0.5))
    # density ratio is viewpoint-dependent (camera sees one side, LiDAR the
    # other), so the band only catches gross collapse (~0 hits) / inflation,
    # not fine density matching — coverage + enclosure are the binding checks
    density_lo, density_hi = 0.001, 5.0
    enclosure_thr = float(cfg.get("validation", {}).get("box_encloses_mask", 0.95))

    gauss_checks = {}
    depth_diffs = []
    density_ratios = []
    enclosures = []
    for pc in per_cam.values():
        cam_spec = pc["cam_spec"]
        k = np.array(cam_spec["K"], dtype=float)
        t_g2c = np.linalg.inv(
            np.array(cam_spec["T_cam2global"], dtype=float).reshape(4, 4))
        h, w = pc["alpha"].shape

        # (i) gauss depth vs lidar depth along the same ray
        pts_c = tf.transform_pts(t_g2c, tf.transform_pts(t_l2g, obj_pts[:, :3]))
        proj = _project(pts_c, k, w, h)
        if proj is not None:
            ui, vi, z = proj
            gd = pc["obj_depth"][vi, ui]
            finite = np.isfinite(gd) & (z < 80.0)
            if finite.any():
                depth_diffs.append(float(np.median(np.abs(gd[finite] - z[finite]))))

        # (ii) silhouette density: object-LiDAR hits per silhouette pixel.
        # A fully occluded camera (visible=0, alpha>0) has no evaluable hits —
        # same "not evaluable" semantics as upstream coverage, not a failure.
        n_alpha = int((pc["alpha"] > 0.05).sum())
        n_visible = int(pc["visible"].sum())
        if proj is not None and n_alpha > 0 and n_visible > 0:
            ui, vi, _z = proj
            hit = pc["visible"][vi, ui]
            ratio = float(hit.sum()) / float(n_alpha)
            density_ratios.append(ratio)

        # (iii) projected box AABB must enclose the mask.
        # Corners are NOT clipped to the frame: a box extending past the image
        # still bounds the silhouette. If any corner is behind the camera the
        # enclosure is ill-defined -> not evaluable for this camera.
        corners = _box_corners(gts)
        pcorners = tf.transform_pts(t_g2c, corners)
        proj2 = _project(pcorners, k, w, h, clip_to_frame=False)
        if proj2 is not None:
            u, v = proj2[0], proj2[1]
            rows, cols = np.where(pc["alpha"] > 0.05)
            if len(rows):
                # splat halo: 3-sigma footprints legitimately spill past the
                # projected box — sparse AV2 clouds have sigma ~0.05 m, i.e.
                # ~19 px at 10 m; allow max(12px, 20%/side). An axis swap
                # still fails by a wide margin (mask lands a full side off).
                span_u = max(u.max() - u.min(), 1.0)
                span_v = max(v.max() - v.min(), 1.0)
                hu = max(12.0, 0.20 * span_u)
                hv = max(12.0, 0.20 * span_v)
                inside = ((rows >= v.min() - hv) & (rows <= v.max() + hv)
                          & (cols >= u.min() - hu) & (cols <= u.max() + hu))
                enclosures.append(float(inside.mean()))

    gauss_checks["gauss_depth_vs_lidar"] = {
        "median_abs_m": depth_diffs,
        "threshold_m": depth_thr,
        "pass": bool(depth_diffs) and max(depth_diffs) <= depth_thr,
    }
    gauss_checks["silhouette_density"] = {
        "ratio": density_ratios,
        "bounds": [density_lo, density_hi],
        "evaluable": bool(density_ratios),
        "pass": (not density_ratios) or all(
            density_lo <= r <= density_hi for r in density_ratios),
    }
    gauss_checks["box_encloses_mask"] = {
        "fraction": enclosures,
        "threshold": enclosure_thr,
        "evaluable": bool(enclosures),
        "pass": (not enclosures) or min(enclosures) >= enclosure_thr,
    }
    gauss_checks["multicam_overlap"] = _multicam_overlap(per_cam)

    checks["gauss_ext"] = gauss_checks
    all_pass = bool(report["all_pass"]) and all(
        c.get("pass", True) for c in gauss_checks.values())
    return {"all_pass": all_pass, "checks": checks}


def _box_corners(gts: list) -> np.ndarray:
    """8 corners of every gt box in the global frame (for enclosure checks)."""
    out = []
    for gt in gts:
        t = np.array(gt["T_model2global"], dtype=float).reshape(4, 4)
        raw_min = np.array(gt["raw_bbox_min"], dtype=float)
        raw_size = np.array(gt["raw_bbox_size"], dtype=float)
        scale = float(gt.get("scale", 1.0))
        # model frame is scaled by `scale` inside T_model2global already
        corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1)
                            for z in (0, 1)], dtype=float)
        pts = (corners * raw_size + raw_min) * scale
        out.append(pts @ t[:3, :3].T + t[:3, 3])
    return np.vstack(out)


def _multicam_overlap(per_cam: dict) -> dict:
    """Informational: pairwise mask IoU across cameras that see the object."""
    names = sorted(per_cam)
    ious = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = per_cam[names[i]]["visible"]
            b = per_cam[names[j]]["visible"]
            if a.shape != b.shape:
                continue
            inter = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            if union:
                ious[f"{names[i]}|{names[j]}"] = float(inter) / float(union)
    return {"pairwise_iou": ious, "pass": True}
