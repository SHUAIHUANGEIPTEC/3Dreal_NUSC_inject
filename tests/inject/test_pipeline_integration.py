"""Integration: scene-0061 Gaussian injection + standard export (auto-skipped)."""
from __future__ import annotations

import json

import pytest

from realinject.assets.manifest import AssetLibrary
from realinject.inject import batch, bridge
from realinject.inject.export import export as export_dataset


@pytest.mark.integration
def test_scene0061_five_frames_and_export(dataset_root, tmp_path):
    """P3 acceptance: 5 frames all_pass, export verify ok, tokens queryable."""
    from nusc_inject import nusc_io

    from realinject import paths

    cfg = paths.load_config("configs/inject_gauss.yaml")
    cfg["dataset"]["scene"] = "scene-0061"
    cfg.pop("scenes", None)
    bridge.bootstrap(cfg)

    library_root = paths.resolve((cfg.get("library") or {}).get("path", "data/library"))
    lib = AssetLibrary.load(library_root / "assets_manifest.json")
    records = lib.passed()
    assert records, "asset library must contain passed records"

    meta = nusc_io.NuScenesMini(cfg["dataset"]["root"], cfg["dataset"]["version"])
    batch_dir = tmp_path / "batch"
    specs, summary = batch.run_tracking(cfg, meta, records, batch_dir,
                                        str(library_root), max_frames=5)
    assert summary["frames_annotated"] >= 3
    assert summary["n_tracks_realized"] >= 1
    batch.write_batch(batch_dir, specs, summary, "itest")

    # every annotated frame passed validation (frames may close early on
    # invisible objects, but annotated frames must be all_pass)
    reports = sorted(batch_dir.glob("scene-0061/f*/validation_report.json"))
    assert reports, "no validation reports written"
    for rep in reports:
        data = json.loads(rep.read_text())
        assert data["all_pass"], f"{rep} failed: {data['checks']}"

    # export + devkit verify
    export_root = tmp_path / "export"
    result = export_dataset(cfg, specs, export_root)
    assert result["n_new_annotations"] >= 1
    assert result["verify"].get("problems") in (None, [])
    assert result["instance_tokens"]
