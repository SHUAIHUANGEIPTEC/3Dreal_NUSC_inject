"""Harmonize units: mask reconstruction, LAB params, smoothing, refiners."""
from __future__ import annotations

import numpy as np
import pytest

from realinject.harmonize import lab_match, masks, metrics
from realinject.harmonize.refine import get_refiner


# ---------------------------------------------------------------- masks
def _write_fake_frame(tmp_path, cam="CAM_FRONT"):
    import cv2
    h, w = 40, 60
    fd = tmp_path / "f001"
    (fd / "composite").mkdir(parents=True)
    (fd / "gauss").mkdir()
    (fd / "depth").mkdir()
    comp = np.full((h, w, 3), 100, np.uint8)
    cv2.imwrite(str(fd / "composite" / f"{cam}.jpg"), comp)
    fg = np.zeros((h, w, 4), np.uint8)
    fg[10:20, 20:30, :3] = 200
    fg[10:20, 20:30, 3] = 255
    cv2.imwrite(str(fd / "gauss" / f"{cam}.png"), fg)
    obj_d = np.full((h, w), np.inf, np.float32)
    obj_d[10:20, 20:30] = 5.0
    scn_d = np.full((h, w), np.inf, np.float32)
    scn_d[10:20, 20:30] = 9.0           # scene farther -> visible
    np.save(fd / "depth" / f"{cam}_obj_depth.npy", obj_d)
    np.save(fd / "depth" / f"{cam}_scene_depth.npy", scn_d)
    return fd


def test_mask_reconstruction_matches_composite_rule(tmp_path):
    fd = _write_fake_frame(tmp_path)
    rec = masks.load_frame_cam(fd, "CAM_FRONT", alpha_thr=0.05,
                               depth_margin=0.5)
    assert rec is not None
    expect = np.zeros((40, 60), bool)
    expect[10:20, 20:30] = True         # alpha>thr AND 5<=9+0.5
    np.testing.assert_array_equal(rec["visible"], expect)

    # occluded case: scene CLOSER than object -> not visible
    np.save(fd / "depth" / "CAM_FRONT_scene_depth.npy",
            np.where(np.isfinite(np.load(fd / "depth" / "CAM_FRONT_obj_depth.npy")),
                     3.0, np.inf).astype(np.float32))
    rec2 = masks.load_frame_cam(fd, "CAM_FRONT", 0.05, 0.5)
    assert not rec2["visible"].any()


def test_ring_and_feather(tmp_path):
    fd = _write_fake_frame(tmp_path)
    rec = masks.load_frame_cam(fd, "CAM_FRONT", 0.05, 0.5)
    target = rec["target"]
    ring = masks.ring_region(target, 2, 8)
    assert ring.any() and not (ring & target).any()
    f = masks.feather_mask(rec["alpha"], rec["visible"])
    assert f.min() >= 0.0 and f.max() <= 1.0
    assert f[target].mean() > 0.5


# ---------------------------------------------------------------- LAB
def _img_with_regions():
    img = np.full((50, 50, 3), 60, np.uint8)      # dark background
    img[10:30, 10:30] = 200                        # bright object
    target = np.zeros((50, 50), bool)
    target[10:30, 10:30] = True
    ring = np.zeros((50, 50), bool)
    ring[5:35, 5:35] = True
    ring &= ~target
    return img, target, ring


def test_apply_params_identity_roundtrip():
    img, _, _ = _img_with_regions()
    out = lab_match.apply_params(img, {"L": (1.0, 0.0)}, None)
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 2


def test_full_match_zeroes_lightness_gap():
    img, target, ring = _img_with_regions()
    p = lab_match.compute_params(img, target, ring, channels=("L",))
    assert p is not None
    out = lab_match.apply_params(img, p, None)    # global: gap scales by gain
    # with mask=None BOTH regions move; check the TRANSFORMED means satisfy
    # out_target = gain*raw_target + bias exactly:
    import cv2
    lab0 = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab1 = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    g, b = p["L"]
    np.testing.assert_allclose(lab1[:, :, 0][target],
                               g * lab0[:, :, 0][target] + b, atol=1.5)


def test_feathered_match_keeps_background():
    img, target, ring = _img_with_regions()
    p = lab_match.compute_params(img, target, ring, channels=("L",))
    feather = np.zeros((50, 50), np.float32)
    feather[target] = 1.0
    out = lab_match.apply_params(img, p, feather)
    np.testing.assert_array_equal(out[ring], img[ring])   # bg untouched


def test_smooth_params_skips_none():
    series = [{"L": (1.0, 0.0)}, None, {"L": (1.0, 10.0)},
              {"L": (1.0, 20.0)}, {"L": (1.0, 30.0)}]
    sm = lab_match.smooth_params(series, window=3)
    assert sm[1] is None
    assert sm[0] is not None and sm[4] is not None
    # smoothed middle value strictly between neighbours' extremes
    assert 0.0 < sm[2]["L"][1] < 30.0


def test_compute_params_defaults_to_lightness_only():
    img, target, ring = _img_with_regions()
    p = lab_match.compute_params(img, target, ring)
    assert set(p.keys()) == {"L"}


# ---------------------------------------------------------------- metrics
def test_lightness_chroma_gap():
    img, target, ring = _img_with_regions()
    dl, ch = metrics.lightness_chroma_gap(img, target, ring)
    assert dl > 20                        # object much brighter
    assert ch >= 0


def test_correction_flicker_zero_for_no_change():
    img, target, _ = _img_with_regions()
    imgs = [img] * 3
    masks_l = [target] * 3
    assert metrics.correction_flicker(imgs, imgs, masks_l) == 0.0


# ---------------------------------------------------------------- refiners
def test_refiner_switching():
    assert get_refiner("none").name == "none"
    assert get_refiner("cpu", cfg={}).name == "cpu"
    with pytest.raises(RuntimeError, match="GPU"):
        get_refiner("sd_inpaint")
    with pytest.raises(RuntimeError, match="GPU"):
        get_refiner("magicdrive")
    with pytest.raises(ValueError):
        get_refiner("bogus")
