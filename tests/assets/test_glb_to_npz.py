"""GLB -> canonical mesh npz adapter contract."""
from __future__ import annotations

import numpy as np
import pytest

from realinject.assets import glb_convert as conv


def _box_arrays():
    """A 2x1x1 box with a trivial UV layout (unit square per face)."""
    import trimesh
    m = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    verts = np.asarray(m.vertices, dtype=float)
    faces = np.asarray(m.faces, dtype=np.int32)
    uv = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
                 (len(verts) // 3 + 1, 1))[:len(verts)]
    tex = np.full((8, 8, 3), 200, dtype=np.uint8)   # mid-gray, not black
    return verts, faces, uv, tex


def test_canon_from_arrays_contract():
    verts, faces, uv, tex = _box_arrays()
    canon, stats = conv.canon_from_arrays(verts, faces, uv, tex_rgb=tex)

    assert canon["verts"].dtype == np.float32
    assert canon["colors"].shape == (len(verts), 3)
    assert canon["uv"].shape == (len(verts), 2)
    assert canon["normals"].shape == (len(verts), 3)
    # bbox-min contract: verts start at the origin
    np.testing.assert_allclose(canon["verts"].min(axis=0), 0.0, atol=1e-5)
    # raw bbox matches the un-shifted extent
    size = verts.max(axis=0) - verts.min(axis=0)
    np.testing.assert_allclose(np.sort(canon["raw_bbox_size"]),
                               np.sort(size), atol=1e-4)
    # mid-gray texture -> no black texels, plausible dims
    assert stats["texture_black_frac"] == 0.0
    assert all(d > 0 for d in stats["dims_lwh_m"])


def test_canon_vertex_color_fallback():
    verts, faces, uv, _ = _box_arrays()
    vcolors = np.full((len(verts), 4), 128, dtype=np.uint8)
    canon, stats = conv.canon_from_arrays(verts, faces, uv,
                                          vcolors=vcolors)
    assert canon["colors"].shape == (len(verts), 3)
    assert float(canon["colors"].mean()) == pytest.approx(128 / 255, abs=1e-3)
    assert stats["texture_black_frac"] == 0.0


def test_glb_roundtrip(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from PIL import Image

    mesh = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    img = Image.new("RGB", (8, 8), (200, 180, 60))
    # uniform texture: any uv samples the same warm texel
    uv = np.zeros((len(mesh.vertices), 2), dtype=float)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.SimpleMaterial(image=img))
    glb = tmp_path / "box.glb"
    mesh.export(str(glb))

    out = tmp_path / "box.npz"
    stats = conv.glb_to_npz(glb, out)
    assert out.exists()
    with np.load(out) as d:
        assert {"verts", "faces", "uv", "normals", "colors",
                "raw_bbox_min", "raw_bbox_size", "dims_lwh",
                "yaw_rad"} <= set(d.files)
        assert d["verts"].shape[1] == 3
        assert d["colors"].max() > 0.5      # sampled the warm texture
    assert stats["n_verts"] >= 8


def test_size_prior_scales_out_of_range():
    verts, faces, uv, tex = _box_arrays()          # longest extent = 2.0
    canon, stats = conv.canon_from_arrays(
        verts, faces, uv, tex_rgb=tex, size_range=(0.4, 0.7))
    assert stats["size_scale"] == pytest.approx(0.7 / 2.0)
    assert float(max(canon["dims_lwh"])) == pytest.approx(0.7, abs=1e-5)
    # raw bbox scales with the verts (placement math stays consistent)
    assert float(canon["raw_bbox_size"].max()) == pytest.approx(
        2.0 * stats["size_scale"], abs=1e-4)

    canon2, stats2 = conv.canon_from_arrays(
        verts, faces, uv, tex_rgb=tex, size_range=(1.0, 4.0))
    assert stats2["size_scale"] == 1.0              # in range: untouched
    assert float(max(canon2["dims_lwh"])) == pytest.approx(2.0, abs=1e-5)


def test_size_range_for_class_lookup():
    from realinject.assets.quality import size_range_for_class
    assert size_range_for_class("fire_extinguisher_2552_10bfa9d3") == (0.4, 0.7)
    assert size_range_for_class("cone_2417_10fb864c") == (0.3, 0.8)
    assert size_range_for_class("fireplug_2633_06917674") == (0.6, 1.0)
    assert size_range_for_class("British_Postbox") is None
