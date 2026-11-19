"""Placement priors, render merging, bridge contract."""
from __future__ import annotations

import numpy as np
import pytest

from realinject.gauss.model import GaussianCloud, identity_quat
from realinject.gauss.raster_cpu import render_gaussians
from realinject.inject import placement
from realinject.inject.pipeline_gauss import merge_renders


def test_prior_for_categories():
    assert placement.prior_for("vehicle.car")[0] == (12.0, 30.0)
    assert placement.prior_for("human.pedestrian.animal")[0] == (6.0, 20.0)
    assert placement.prior_for(None) == placement.DEFAULT_PRIOR


def test_plan_one_uses_prior_and_is_deterministic():
    cfg = {"objects": {"ground": {"search_radius_m": 4.0, "z_percentile": 15.0}}}
    model = {"category": "vehicle.car", "path": "x.npz", "scale": 1.0}
    a = placement.plan_one(cfg, np.random.default_rng(0), model)
    b = placement.plan_one(cfg, np.random.default_rng(0), model)
    assert a["placement_cfg"]["offset_in_ego"] == b["placement_cfg"]["offset_in_ego"]
    fwd = a["placement_cfg"]["offset_in_ego"][0]
    assert 12.0 <= fwd <= 30.0


def test_plan_tracks_draws_records():
    from realinject.assets.manifest import AssetRecord

    recs = [AssetRecord(asset_id=f"a{i}", kind="realcar_mesh", source_path="s",
                        canonical_path=f"c{i}.npz", category="vehicle.car",
                        dims_lwh_m=(4.9, 1.9, 1.3)) for i in range(3)]
    cfg = {"objects": {"seed": 42, "tracking": {"count": 2, "length": [3, 5]},
                       "ground": {}}}
    tracks = placement.plan_tracks(cfg, "scene-0061", 10, np.random.default_rng(1), recs)
    assert len(tracks) == 2
    assert all(t["asset_id"].startswith("a") for t in tracks)
    assert all(t["T"] is None for t in tracks)
    assert all(1 <= t["length"] <= 5 for t in tracks)


def test_model_cfg_for_keeps_metric_scale():
    from realinject.assets.manifest import AssetRecord

    rec = AssetRecord(asset_id="realcar/x", kind="realcar_mesh", source_path="s",
                      canonical_path="c.npz", category="vehicle.car",
                      dims_lwh_m=(4.9, 1.9, 1.3))
    mc = placement.model_cfg_for(rec, "/lib")
    assert mc["scale"] == 1.0
    assert mc["path"] == "/lib/c.npz"
    assert mc["category"] == "vehicle.car"


def _render(eye, target, w=320, h=240, color=(1.0, 0.0, 0.0)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f = f / np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x = x / np.linalg.norm(x)
    y = np.cross(f, x)
    t = np.eye(4)
    t[:3, :3] = np.column_stack([x, y, f])
    t[:3, 3] = eye
    cam = {"name": "C", "K": [[400.0, 0, w / 2], [0, 400.0, h / 2], [0, 0, 1]],
           "T_cam2global": t.tolist(), "width": w, "height": h}
    n = 400
    rng = np.random.default_rng(0)
    cloud = GaussianCloud(
        xyz=rng.uniform(-0.5, 0.5, (n, 3)).astype(np.float32),
        rot=identity_quat(n), scale=np.full((n, 3), 0.03, np.float32),
        opacity=np.full(n, 0.99, np.float32),
        rgb=np.tile(np.asarray(color, np.float32), (n, 1)))
    return cam, cloud


def test_merge_renders_single_passthrough():
    cam, cloud = _render([0, -5, 0], [0, 0, 0])
    out = render_gaussians(cam, cloud, np.eye(4))
    merged = merge_renders([out])
    assert merged is out


def test_merge_renders_nearest_wins():
    cam, cloud_far = _render([0, -5, 0], [0, 0, 0], color=(0.0, 0.0, 1.0))
    _, cloud_near = _render([0, -5, 0], [0, 0, 0], color=(1.0, 0.0, 0.0))
    # same xy footprint, different depth along the view axis
    far = render_gaussians(cam, cloud_far, np.eye(4))
    t_near = np.eye(4)
    t_near[1, 3] = -1.0  # move the near cloud 1 m toward the camera
    near = render_gaussians(cam, cloud_near, t_near)
    merged = merge_renders([far, near])
    both = far.mask & near.mask
    assert both.sum() > 100
    # near (red) wins where both are opaque
    strong = both & (merged.alpha > 0.9)
    assert strong.sum() > 50
    assert np.allclose(merged.rgb[strong], [255, 0, 0], atol=5)
    assert np.all(merged.depth[strong] < far.depth[strong] + 1e-6)


def test_bridge_contract():
    from realinject.inject import bridge

    cfg = {"inject_nusc": {"root": str(_inject_root())}}
    if not (_inject_root() / "src").is_dir():
        pytest.skip("nusc-inject not available")
    bridge.bootstrap(cfg)  # raises if the contract is broken
    bridge.assert_upstream_contract()


def _inject_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[2].parent / "nusc-inject"


def _cluttered_scene():
    """Road at z=-1.2 with SPARSE returns (occluded by the truck) under a
    dense truck body at z=[0.5, 3.0]: the naive 15th percentile lands on the
    truck, the low-tail seed still finds the road."""
    rng = np.random.default_rng(0)
    road = np.column_stack([
        rng.uniform(-6, 6, 90), rng.uniform(-6, 6, 90),
        np.full(90, -1.2) + rng.normal(0, 0.02, 90)])
    truck = np.column_stack([
        rng.uniform(-1, 1, 660), rng.uniform(-1, 1, 660),
        rng.uniform(0.5, 3.0, 660)])
    return np.vstack([road, truck])


def test_robust_ground_z_ignores_elevated_clutter():
    pts = _cluttered_scene()
    z = placement.robust_ground_z(pts, 0.0, 0.0, radius_m=6.0)
    assert abs(z - (-1.2)) < 0.1, f"picked clutter: {z:.2f}"
    # the naive percentile build_gt uses lands on the truck here
    d2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
    near = pts[d2 <= 36.0]
    naive = float(np.percentile(near[:, 2], 15))
    assert naive > -0.5, "test scene must actually fool the naive percentile"


def test_resnap_ground_shifts_pose_and_translation():
    pts = _cluttered_scene()
    # a gt as build_gt would produce it if the percentile hit the truck
    gt = {
        "ground_z_lidar": 0.6,
        "T_model2lidar": np.eye(4).reshape(-1).tolist(),
        "T_lidar2global": np.eye(4).reshape(-1).tolist(),
        "raw_bbox_min": [0.0, 0.0, 0.0],
        "raw_bbox_size": [4.9, 1.9, 1.5],
        "translation": [2.45, 0.95, 0.75],
        "placement": {"offset_in_lidar_frame": [0.0, 0.0, 0.6]},
    }
    shifted = placement.resnap_ground(gt, pts, {"ground_resnap": {}})
    # object floated 1.8 m above road -> shift ~-1.8, ground_z re-estimated
    assert -2.0 < shifted < -1.5
    assert abs(gt["ground_z_lidar"] - (-1.2)) < 0.1
    t_m2l = np.array(gt["T_model2lidar"]).reshape(4, 4)
    assert abs(t_m2l[2, 3] - shifted) < 1e-9
    # translation follows the shift (rotation untouched: eye)
    assert abs(gt["translation"][2] - (0.75 + shifted)) < 1e-6
    assert gt["placement"]["offset_in_lidar_frame"][2] == gt["ground_z_lidar"]


def test_resnap_ground_noop_for_small_delta_and_disabled():
    pts = np.column_stack([
        np.random.default_rng(1).uniform(-5, 5, 300),
        np.random.default_rng(2).uniform(-5, 5, 300),
        np.full(300, -1.0)])
    gt = {
        "ground_z_lidar": -1.0,
        "T_model2lidar": np.eye(4).reshape(-1).tolist(),
        "T_lidar2global": np.eye(4).reshape(-1).tolist(),
        "raw_bbox_min": [0.0, 0.0, 0.0],
        "raw_bbox_size": [4.9, 1.9, 1.5],
        "translation": [2.45, 0.95, -0.25],
        "placement": {"offset_in_lidar_frame": [0.0, 0.0, -1.0]},
    }
    assert placement.resnap_ground(gt, pts, {"ground_resnap": {}}) == 0.0
    assert placement.resnap_ground(
        gt, pts, {"ground_resnap": {"enabled": False}}) == 0.0
    assert gt["ground_z_lidar"] == -1.0


def test_ground_contact_check_in_validate_ext():
    from realinject.inject import validate_ext

    cfg = {"validation": {"ground_contact_max_m": 0.35}}
    pts = _cluttered_scene()

    def _gt(bottom_z_lidar):
        t = np.eye(4)
        t[2, 3] = bottom_z_lidar
        return {"T_model2lidar": t.reshape(-1).tolist(),
                "T_model2global": t.reshape(-1).tolist(),
                "raw_bbox_min": [0.0, 0.0, 0.0],
                "raw_bbox_size": [4.9, 1.9, 1.5], "scale": 1.0}

    # seated: bbox bottom at road level passes
    seated = validate_ext._ground_contact(cfg, [_gt(-1.2)], pts)
    assert seated["evaluable"] and seated["pass"]
    # floating 1 m above road fails
    floating = validate_ext._ground_contact(cfg, [_gt(-0.2)], pts)
    assert floating["evaluable"] and not floating["pass"]
    assert floating["gap_m"][0] > 0.9
    # no scene points -> not evaluable, not failing
    skipped = validate_ext._ground_contact(cfg, [_gt(-0.2)], None)
    assert not skipped["evaluable"] and skipped["pass"]


def test_resnap_allows_large_shift_when_confident():
    """build_gt polluted by building returns (naive p15 = +2.8 m) while the
    road band is clearly visible -> a >3 m correction must go through."""
    pts = _cluttered_scene()  # road -1.2 with 90 band points (confident)
    gt = {
        "ground_z_lidar": 2.8,  # what the naive percentile produced
        "T_model2lidar": np.eye(4).reshape(-1).tolist(),
        "T_lidar2global": np.eye(4).reshape(-1).tolist(),
        "raw_bbox_min": [0.0, 0.0, 0.0],
        "raw_bbox_size": [4.9, 1.9, 1.5],
        "translation": [2.45, 0.95, 3.55],
        "placement": {"offset_in_lidar_frame": [0.0, 0.0, 2.8]},
    }
    shifted = placement.resnap_ground(gt, pts, {"ground_resnap": {}})
    assert shifted < -3.9, "confident large shift must not be capped at 3 m"
    assert abs(gt["ground_z_lidar"] - (-1.2)) < 0.1
