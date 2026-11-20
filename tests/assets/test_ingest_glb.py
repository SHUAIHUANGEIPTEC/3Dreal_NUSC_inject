"""GLB source ingest: convert -> gates.glb -> glb_mesh records."""
from __future__ import annotations

import numpy as np
import pytest

from realinject.assets import quality
from realinject.assets.ingest_glb import ingest_glb


def _make_glb(directory, name: str, scale: float = 1.0):
    trimesh = pytest.importorskip("trimesh")
    from PIL import Image

    mesh = trimesh.creation.box(extents=(2.0 * scale, 1.0 * scale, 1.0 * scale))
    img = Image.new("RGB", (8, 8), (200, 180, 60))
    uv = np.zeros((len(mesh.vertices), 2), dtype=float)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.SimpleMaterial(image=img))
    path = directory / f"{name}.glb"
    mesh.export(str(path))
    return path


def test_gate_glb_window():
    # small prop inside the wide window passes texture guards
    ok, reasons = quality.gate_glb((2.0, 1.0, 1.0), 50.0,
                                   texture_black_frac=0.0)
    assert ok and reasons == []
    # unit-broken model (4-km scale) dies on the upper bound
    ok, reasons = quality.gate_glb((4992.0, 1367.0, 3981.0), 50.0,
                                   texture_black_frac=0.0)
    assert not ok and any(r.startswith("l_out_of_range") for r in reasons)
    # flat / black textures rejected like realcar
    ok, reasons = quality.gate_glb((2.0, 1.0, 1.0), 2.0, texture_black_frac=0.0)
    assert not ok and any(r.startswith("texture_flat") for r in reasons)
    ok, reasons = quality.gate_glb((2.0, 1.0, 1.0), 50.0, texture_black_frac=0.2)
    assert not ok and any(r.startswith("texture_black") for r in reasons)


def test_ingest_glb_pass_and_fail(tmp_path):
    root = tmp_path / "assets"
    (root / "glb_sources").mkdir(parents=True)
    _make_glb(root / "glb_sources", "prop_ok")
    _make_glb(root / "glb_sources", "prop_huge", scale=5000.0)

    lib = tmp_path / "lib"
    recs = ingest_glb(root, lib, {"gates": {}})
    assert len(recs) == 2

    ok = next(r for r in recs if r.asset_id == "glb/prop_ok")
    bad = next(r for r in recs if r.asset_id == "glb/prop_huge")

    assert ok.kind == "glb_mesh"
    assert ok.quality["passed"] is True
    assert ok.category == "movable_object.anomaly"
    assert ok.source_path == "glb_sources/prop_ok.glb"
    assert (lib / "canonical" / "glb" / "prop_ok.npz").exists()

    assert bad.quality["passed"] is False
    assert any(r.startswith("l_out_of_range") for r in bad.quality["reasons"])


def test_ingest_glb_missing_dir(tmp_path):
    assert ingest_glb(tmp_path / "nope", tmp_path / "lib", {}) == []
