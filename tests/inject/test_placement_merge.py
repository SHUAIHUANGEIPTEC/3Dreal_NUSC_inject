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
