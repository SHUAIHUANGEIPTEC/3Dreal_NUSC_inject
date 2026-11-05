"""CPU-side geometry completion (advisory, no GPU required).

The only CPU completion this project performs is boundary-loop hole filling:
3DRealCar scans systematically miss the roof/glass surfaces, and an open mesh
leaks light when initialized as Gaussians. Fan-fill from the loop centroid is
crude but deterministic and dependency-free (``trimesh.repair.fill_holes``
needs ``networkx``, absent from the shared env).

AV2 half-shells are deliberately NOT completed on CPU — report §7.2: a 0.6 m
animal spans 2-4 scan lines, so cross-frame aggregation cannot recover ears,
legs or tails; the half-shell ships as a Gaussian asset instead.
"""
from __future__ import annotations

import numpy as np
import trimesh


def boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    """Extract ordered boundary loops (vertex index arrays) from a triangle soup."""
    faces = np.asarray(faces, dtype=np.int64)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    # undirected edge use counts: an edge used once lies on the boundary
    und = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(und, axis=0, return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse] == 1]
    if len(boundary) == 0:
        return []
    # next[v] = w for each boundary directed edge (v, w); loops walk this map
    nxt: dict[int, int] = {}
    for v, w in boundary:
        nxt[int(v)] = int(w)
    loops: list[np.ndarray] = []
    visited: set[int] = set()
    for start in list(nxt):
        if start in visited:
            continue
        loop, cur = [], start
        while cur not in visited and cur in nxt:
            visited.add(cur)
            loop.append(cur)
            cur = nxt[cur]
        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def fill_holes(mesh: trimesh.Trimesh, max_loop_len: int = 512) -> tuple[trimesh.Trimesh, dict]:
    """Fan-fill small boundary loops; returns (new mesh, stats).

    Loops longer than ``max_loop_len`` are left open and counted as skipped so
    the caller can record a residual hole area.
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    loops = boundary_loops(faces)

    new_verts = list(verts)
    new_faces = list(faces)
    filled = skipped = 0
    filled_edges = 0
    for loop in loops:
        if len(loop) > max_loop_len:
            skipped += 1
            continue
        centroid = verts[loop].mean(axis=0)
        ci = len(new_verts)
        new_verts.append(centroid)
        for i in range(len(loop)):
            a, b = int(loop[i]), int(loop[(i + 1) % len(loop)])
            # a boundary edge (a,b) already appears in one face with this
            # direction; the fill face must traverse it opposite (b,a) to keep
            # a consistent outward orientation
            new_faces.append((b, a, ci))
        filled += 1
        filled_edges += len(loop)

    out = trimesh.Trimesh(
        vertices=np.asarray(new_verts, dtype=np.float64),
        faces=np.asarray(new_faces, dtype=np.int64),
        process=False,
    )
    stats = {
        "n_loops": len(loops),
        "filled": filled,
        "skipped": skipped,
        "filled_edges": filled_edges,
        "added_verts": len(new_verts) - len(verts),
        "added_faces": len(new_faces) - len(faces),
        "watertight_after": bool(out.is_watertight),
    }
    return out, stats
