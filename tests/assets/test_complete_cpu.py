"""CPU hole filling on synthetic meshes."""
from __future__ import annotations

import numpy as np
import trimesh

from realinject.assets import complete_cpu


def test_boundary_loops_on_open_box():
    box = trimesh.creation.box(extents=(1, 1, 1))
    open_box = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-1], process=False)
    loops = complete_cpu.boundary_loops(open_box.faces)
    assert len(loops) == 1
    assert len(loops[0]) == 3  # one missing triangle: 3 boundary edges


def test_fill_restores_watertight_box():
    box = trimesh.creation.box(extents=(1, 1, 1))
    open_box = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-1], process=False)
    assert not open_box.is_watertight
    filled, stats = complete_cpu.fill_holes(open_box)
    assert stats["filled"] == 1 and stats["skipped"] == 0
    assert stats["watertight_after"]
    assert filled.is_watertight
    assert np.isclose(-filled.volume, 1.0, atol=1e-6) or np.isclose(filled.volume, 1.0, atol=1e-6)


def test_closed_mesh_untouched():
    box = trimesh.creation.box(extents=(1, 1, 1))
    filled, stats = complete_cpu.fill_holes(box)
    assert stats["n_loops"] == 0 and stats["filled"] == 0
    assert len(filled.faces) == len(box.faces)


def test_max_loop_len_skips_large_holes():
    box = trimesh.creation.box(extents=(1, 1, 1))
    open_box = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-1], process=False)
    filled, stats = complete_cpu.fill_holes(open_box, max_loop_len=2)
    assert stats["skipped"] == 1 and stats["filled"] == 0
    assert not filled.is_watertight
