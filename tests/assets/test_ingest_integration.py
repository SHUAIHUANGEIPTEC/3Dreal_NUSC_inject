"""Integration: ingest a few real assets end-to-end (auto-skipped without corpus)."""
from __future__ import annotations

import json

import pytest

from realinject.assets.ingest_av2 import ingest_av2
from realinject.assets.ingest_realcar import ingest_realcar
from realinject.assets.manifest import AssetLibrary


@pytest.mark.integration
def test_ingest_realcar_smoke(assets_root, library_root):
    cfg = {"gates": {}, "gaussian_init": {}}
    records = ingest_realcar(assets_root, library_root, cfg, limit=3)
    assert len(records) >= 1
    for r in records:
        assert r.asset_id.startswith("realcar/")
        assert r.category == "vehicle.car"
        assert r.kind == "realcar_mesh"
        if r.quality.get("passed"):
            length, w, h = r.dims_lwh_m
            assert 1.4 <= w <= 2.6 and 1.0 <= h <= 2.0
            assert (library_root / r.canonical_path).exists()


@pytest.mark.integration
def test_ingest_av2_smoke(assets_root, library_root):
    cfg = {"gates": {}, "gaussian_init": {}}
    records = ingest_av2(assets_root, library_root, cfg, limit=5)
    assert len(records) >= 1
    for r in records:
        assert r.asset_id.startswith("av2/")
        assert r.kind == "av2_cloud"
        if r.quality.get("passed"):
            assert (library_root / r.canonical_path).exists()


@pytest.mark.integration
def test_library_manifest_written(assets_root, library_root):
    cfg = {"gates": {}}
    records = ingest_realcar(assets_root, library_root, cfg, limit=2)
    records += ingest_av2(assets_root, library_root, cfg, limit=3)
    lib = AssetLibrary(records)
    path = lib.save(library_root / "assets_manifest.json", generated_at="test")
    data = json.loads(path.read_text())
    assert data["schema"] == "3dreal-assets/1"
    assert len(data["assets"]) == len(records)
    assert data["counts"]["total"] == len(records)
