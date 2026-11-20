"""M1 regression: no-camera/validation failures close only pose-less tracks.

A track that already froze its world pose rides through a rejected frame and
is emitted on later good frames; only tracks with ``T is None`` (no pose to
freeze) are terminated with the frame status.
"""
from __future__ import annotations

import numpy as np
import pytest

from realinject.inject import batch


class _FakeMeta:
    def __init__(self, n_frames: int):
        self._samples = [{"token": f"tok{i}"} for i in range(n_frames)]
        self.scene = {"scene-x": {"name": "scene-x",
                                  "first_sample_token": "tok0"}}
        self.sample_idx_by_token = {s["token"]: i
                                    for i, s in enumerate(self._samples)}

    def scene_samples(self, scene):
        return self._samples


def _track(length: int = 3) -> dict:
    return {"key": "t0", "asset_id": "asset0", "start": 0, "length": length,
            "T": None, "terminated": False, "end_reason": None,
            "model_cfg": {"model": "m"},
            "placement_cfg": {"ground": {}}}


def _fake_run_sample(statuses: list, objects_out: list | None = None):
    calls = {"n": 0}

    def fake(cfg, meta, idx, objects, out_dir, cloud_cache=None,
             track_keys=None, drop_invisible=True):
        if objects_out is not None:
            objects_out.append(list(objects))
        i = calls["n"]
        calls["n"] += 1
        status = statuses[i] if i < len(statuses) else statuses[-1]
        if status != "ok":
            return {"status": status}
        # seat anchor: offset [1, 2, -0.5] in an identity lidar frame
        # -> creation-frame seat global z = -0.5
        gts = [{
            "T_model2global": np.eye(4),
            "T_lidar2global": np.eye(4),
            "placement": {"offset_in_lidar_frame": [1.0, 2.0, -0.5]},
        } for _ in track_keys]
        return {"status": "ok",
                "spec": {"scene": "scene-x",
                         "objects": [{"track_key": k} for k in track_keys]},
                "gts": gts,
                "dropped": []}

    return fake


def _run(monkeypatch, tmp_path, statuses: list, n_frames: int):
    monkeypatch.setattr(batch.placement, "plan_tracks",
                        lambda *a, **k: [_track(n_frames)])
    monkeypatch.setattr(batch, "run_sample", _fake_run_sample(statuses))
    cfg = {"dataset": {"scene": "scene-x"}, "objects": {"seed": 42}}
    specs, summary = batch.run_tracking(cfg, _FakeMeta(n_frames), [],
                                        tmp_path, library_root="lib")
    return specs, summary


@pytest.mark.parametrize("bad_status", ["validation_failed", "no_camera"])
def test_pose_less_track_closed_with_frame_status(monkeypatch, tmp_path,
                                                   bad_status):
    specs, summary = _run(monkeypatch, tmp_path,
                          [bad_status, bad_status, bad_status], 3)
    assert specs == []
    assert summary["frames_annotated"] == 0
    assert summary["end_reasons"] == {bad_status: 1}
    assert summary["tracks"][0]["T_resolved"] is False


@pytest.mark.parametrize("bad_status", ["validation_failed", "no_camera"])
def test_frozen_track_rides_through_bad_frame(monkeypatch, tmp_path,
                                               bad_status):
    specs, summary = _run(monkeypatch, tmp_path,
                          ["ok", bad_status, "ok"], 3)
    assert summary["frames_annotated"] == 2
    assert summary["end_reasons"] == {"window_end": 1}
    track = summary["tracks"][0]
    assert track["end_reason"] == "window_end"
    assert track["T_resolved"] is True
    # the good frames still carry the frozen track
    assert [o["track_key"] for o in specs[0]["objects"]] == ["t0"]
    assert [o["track_key"] for o in specs[1]["objects"]] == ["t0"]


def test_frozen_frames_carry_creation_seat(monkeypatch, tmp_path):
    """Plan A: frame 0 resolves the seat; frozen frames pass it through."""
    objects_seen: list = []
    monkeypatch.setattr(batch.placement, "plan_tracks",
                        lambda *a, **k: [_track(3)])
    monkeypatch.setattr(batch, "run_sample",
                        _fake_run_sample(["ok", "ok", "ok"],
                                         objects_out=objects_seen))
    cfg = {"dataset": {"scene": "scene-x"}, "objects": {"seed": 42}}
    _specs, summary = batch.run_tracking(cfg, _FakeMeta(3), [],
                                         tmp_path, library_root="lib")
    assert summary["frames_annotated"] == 3

    # frame 0: fresh placement, no seat yet
    assert "seat_z_global" not in objects_seen[0][0]["placement_cfg"]
    # frames 1-2: frozen placement carries the creation seat (global z -0.5)
    for frozen in objects_seen[1] + objects_seen[2]:
        pc = frozen["placement_cfg"]
        assert "from_global" in pc
        assert pc["seat_z_global"] == pytest.approx(-0.5)
