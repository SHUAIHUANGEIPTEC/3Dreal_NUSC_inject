"""Normalization: frame conventions, yaw alignment, texture sampling."""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from realinject.assets import normalize


def test_yup_to_blender_maps_y_to_z():
    pts = np.array([[1.0, 2.0, 3.0]])
    out = normalize.yup_to_blender(pts)
    # (x, y, z) -> (x, -z, y)
    assert np.allclose(out, [[1.0, -3.0, 2.0]])


def test_yaw_align_recovers_long_axis_and_dims():
    # 4.6 x 1.8 rectangle rotated 35 deg in the XZ plane (Y-up)
    w, length = 1.8, 4.6
    corners = np.array([
        [-length / 2, 0, -w / 2], [length / 2, 0, -w / 2],
        [length / 2, 0, w / 2], [-length / 2, 0, w / 2],
    ])
    a = np.radians(35.0)
    r = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rotated = corners @ r.T
    t, angle = normalize.yaw_align(rotated)
    aligned = rotated @ t[:3, :3].T
    extent = aligned.max(axis=0) - aligned.min(axis=0)
    assert np.isclose(extent[0], length, atol=1e-6)  # long axis on X
    assert np.isclose(extent[2], w, atol=1e-6)
    assert np.isclose(abs(np.degrees(angle)), 35.0, atol=1e-6)


def test_sample_texture_bilinear_corners():
    tex = np.array([
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [255, 255, 255]],
    ], dtype=np.uint8)
    # uv (0,0) = bottom-left = last row, first col
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    colors = normalize.sample_texture(tex, uv)
    assert np.allclose(colors[0], [0, 0, 1], atol=1e-6)
    assert np.allclose(colors[1], [1, 1, 1], atol=1e-6)
    assert np.allclose(colors[2], [1, 0, 0], atol=1e-6)
    assert colors.min() >= 0.0 and colors.max() <= 1.0


def test_merge_uv_seam_vertices_dedups():
    # two triangles sharing an edge, with duplicated vertices
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    f = np.array([[0, 1, 2], [3, 4, 5]])
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=np.zeros((6, 2)))
    out = normalize.merge_uv_seam_vertices(m)
    assert len(out.vertices) == 4
    assert len(out.faces) == 2


def test_boundary_edge_count_open_and_closed():
    box = trimesh.creation.box(extents=(1, 1, 1))
    assert normalize.boundary_edge_count(box) == 0
    open_box = trimesh.Trimesh(
        vertices=box.vertices, faces=box.faces[:-1], process=False)
    assert normalize.boundary_edge_count(open_box) == 3  # one quad = 3 exposed edges


def test_canonicalize_av2_translates_bbox_min_to_origin(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 2.0], [0.5, 0.2, 1.0]], np.float32)
    src = tmp_path / "DOG_test.npz"
    np.savez(src, pts=pts, intensity=np.array([10, 200, 0], np.uint8),
             dims=np.array([1.0, 0.5, 2.0]), ego=np.zeros((3, 3)), cat=np.array("DOG"))
    out = tmp_path / "canon.npz"
    stats = normalize.canonicalize_av2(src, out)
    canon = normalize.load_canonical(out)
    assert np.allclose(canon["xyz"].min(axis=0), 0.0, atol=1e-6)
    assert np.allclose(stats["dims_lwh_m"], [1.0, 0.5, 2.0])
    assert stats["n_points"] == 3
    assert stats["cat"] == "DOG"
    assert np.array_equal(canon["intensity"], np.array([10, 200, 0], np.uint8))


def test_canonicalize_av2_rejects_empty(tmp_path):
    src = tmp_path / "x.npz"
    np.savez(src, pts=np.zeros((1, 3)), intensity=np.zeros(1, np.uint8),
             dims=np.ones(3), ego=np.zeros((0, 3)), cat=np.array("DOG"))
    with pytest.raises(ValueError):
        normalize.canonicalize_av2(src, tmp_path / "o.npz")


@pytest.mark.integration
def test_canonicalize_realcar_known_asset(assets_root, library_root):
    scene = "2024_06_04_13_44_39"
    asset_dir = assets_root / "realcar_assets" / scene
    if not asset_dir.exists():
        pytest.skip("reference asset not present")
    stats = normalize.canonicalize_realcar(
        asset_dir, assets_root, library_root / "canonical" / "realcar" / f"{scene}.npz")
    # verified: OBB is 4.945 x 1.945 x 1.28 (manifest L/W/H), not the inflated AABB
    assert np.isclose(stats["dims_lwh_m"][0], 4.945, atol=0.05)
    assert np.isclose(stats["dims_lwh_m"][1], 1.945, atol=0.05)
    assert np.isclose(stats["dims_lwh_m"][2], 1.28, atol=0.02)
    assert stats["n_verts"] == 31879  # UV-seam merge: 34754 -> 31879
    assert stats["boundary_edges"] > 0  # roof/glass holes are real
    assert stats["texture_std"] > 5
    canon = normalize.load_canonical(library_root / "canonical" / "realcar" / f"{scene}.npz")
    assert float(canon["verts"].min()) >= -1e-4  # bbox min at origin
    assert canon["colors"].shape[0] == canon["verts"].shape[0]
