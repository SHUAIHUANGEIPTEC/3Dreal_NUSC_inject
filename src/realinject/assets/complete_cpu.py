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
    # walk the boundary as a directed edge multimap: a non-manifold vertex
    # (two separate loops sharing a vertex) would clobber a single dict entry
    from collections import defaultdict

    nxt: dict[int, list[int]] = defaultdict(list)
    for v, w in boundary:
        nxt[int(v)].append(int(w))
    loops: list[np.ndarray] = []
    used_edges: set[tuple[int, int]] = set()
    for v0 in list(nxt):
        for w0 in list(nxt[v0]):
            if (v0, w0) in used_edges:
                continue
            loop, cur, prev = [], v0, None
            while True:
                if cur in used_edges and prev is None:
                    break
                # pick an unused outgoing edge from cur
                cand = [w for w in nxt.get(cur, [])
                        if (cur, w) not in used_edges]
                if not cand:
                    break
                w = cand[0]
                used_edges.add((cur, w))
                loop.append(cur)
                prev, cur = cur, w
                if cur == v0:
                    break
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


# ---------------------------------------------------------------------------
# route-D CPU completion: realcar hole fill + AV2 half-shell proxy
# ---------------------------------------------------------------------------

def alpha_shape_mesh(points: np.ndarray, alpha: float | None = None,
                     k_scale: float = 2.5) -> trimesh.Trimesh:
    """Closed surface from a (half-)shell: Delaunay + max-edge-length filter.

    A practical alpha-shape variant: keep tetrahedra whose longest edge is
    <= alpha (default: k_scale x median kNN distance). For a nearly complete
    shell (DOG az_span ~347 deg) this yields a closed proxy with the correct
    outer dimensions; it is a *proxy*, not a shape reconstruction — GPU route D
    (TRELLIS / Hunyuan3D-2) replaces it when available.
    """
    from scipy.spatial import Delaunay, cKDTree

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 5:
        raise ValueError(f"too few points for alpha shape: {len(pts)}")
    if alpha is None:
        dist, _ = cKDTree(pts).query(pts, k=min(9, len(pts)))
        alpha = float(k_scale * np.median(dist[:, 1:]))
    tri = Delaunay(pts)
    simp = tri.simplices  # (S, 4)
    p = pts[simp]
    edges = np.concatenate([
        np.linalg.norm(p[:, 0] - p[:, 1], axis=1),
        np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
        np.linalg.norm(p[:, 0] - p[:, 3], axis=1),
        np.linalg.norm(p[:, 1] - p[:, 2], axis=1),
        np.linalg.norm(p[:, 1] - p[:, 3], axis=1),
        np.linalg.norm(p[:, 2] - p[:, 3], axis=1),
    ]).reshape(6, -1)
    keep = edges.max(axis=0) <= alpha
    if not keep.any():
        raise ValueError(f"alpha={alpha:.4f} kept no tetrahedra")
    # alpha-complex surface = faces belonging to exactly ONE kept tet
    # (interior faces are shared by two tets and dropped)
    kept = simp[keep]
    all_faces = kept[:, [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2]].reshape(-1, 3)
    und = np.sort(all_faces, axis=1)
    _, inv, counts = np.unique(und, axis=0, return_inverse=True,
                               return_counts=True)
    boundary = counts[inv] == 1
    faces = all_faces[boundary]
    # drop unreferenced vertices, fix winding
    used = np.unique(faces)
    remap = np.full(len(pts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    mesh = trimesh.Trimesh(vertices=pts[used], faces=remap[faces], process=True)
    return mesh


def mirror_half_shell(xyz: np.ndarray) -> np.ndarray:
    """Mirror a half-shell across the mid-plane of its narrowest horizontal axis.

    Vehicles / wheeled devices are roughly left-right symmetric; mirroring
    doubles the angular coverage of a one-sided scan before surface fitting.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    # narrowest horizontal axis (x/y in canonical frame; z is height)
    axis = int(np.argmin(extent[:2])) + 0
    mid = 0.5 * (xyz[:, axis].min() + xyz[:, axis].max())
    mirrored = xyz.copy()
    mirrored[:, axis] = 2.0 * mid - mirrored[:, axis]
    both = np.vstack([xyz, mirrored])
    # merge seam duplicates (original + mirror coincide on the mid-plane)
    # so Delaunay does not see degenerate coincident vertices
    key = np.round(both, 3)
    _, idx = np.unique(key, axis=0, return_index=True)
    return both[np.sort(idx)]


def complete_realcar_canon(canon: dict, max_loop_len: int = 4096) -> tuple[dict, dict]:
    """Fan-fill boundary loops of a canonical mesh; returns (canon, stats).

    New centroid vertices inherit the loop's mean uv/color/normal so the
    completed mesh stays texture-ready for Gaussian initialization.
    """
    verts = np.asarray(canon["verts"], dtype=np.float64)
    faces = np.asarray(canon["faces"], dtype=np.int64)
    uv = np.asarray(canon["uv"], dtype=np.float64)
    colors = np.asarray(canon["colors"], dtype=np.float64)
    normals = np.asarray(canon["normals"], dtype=np.float64)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    filled, stats = fill_holes(mesh, max_loop_len=max_loop_len)

    new_verts = np.asarray(filled.vertices, dtype=np.float64)
    new_faces = np.asarray(filled.faces, dtype=np.int64)
    # fill_holes appends one centroid vertex per filled loop; recover their
    # attributes from the loop means recorded implicitly by vertex count
    n_old = len(verts)
    n_new = len(new_verts)
    if n_new > n_old:
        # centroid k = n_old + j corresponds to loop j; recover loop indices
        # by re-walking the original boundary loops in fill order
        loops = boundary_loops(faces)
        filled_loops = [lp for lp in loops if len(lp) <= max_loop_len]
        extra_uv = np.zeros((n_new - n_old, 2))
        extra_col = np.zeros((n_new - n_old, 3))
        extra_nrm = np.zeros((n_new - n_old, 3))
        for j, lp in enumerate(filled_loops):
            if j >= n_new - n_old:
                break
            extra_uv[j] = uv[lp].mean(axis=0)
            extra_col[j] = colors[lp].mean(axis=0)
            extra_nrm[j] = normals[lp].mean(axis=0)
        uv = np.vstack([uv, extra_uv])
        colors = np.vstack([colors, extra_col])
        normals = np.vstack([normals, extra_nrm])
    else:
        # nothing filled; arrays unchanged
        pass

    out = dict(canon)
    out["verts"] = new_verts.astype(np.float32)
    out["faces"] = new_faces.astype(np.int32)
    out["uv"] = uv.astype(np.float32)
    out["colors"] = colors.astype(np.float32)
    out["normals"] = normals.astype(np.float32)
    # hole fill does not move the original surface; bbox stays valid.
    # boundary_edges_after is the light-leak metric for 3DGS init: 0 means
    # every hole is closed. is_watertight can still be False when the source
    # mesh has non-manifold edges (edges shared by 3+ faces) — harmless for
    # rendering, recorded for honesty.
    from .normalize import boundary_edge_count

    stats = {
        **stats,
        "method": "fan_fill",
        "n_verts_before": int(n_old),
        "n_verts_after": int(n_new),
        "watertight_after": bool(filled.is_watertight),
        "boundary_edges_after": int(boundary_edge_count(filled)),
    }
    return out, stats


def complete_av2_canon(canon: dict, n_samples: int = 20000,
                       seed: int = 0) -> tuple[dict, dict]:
    """Mirror a half-shell and fit a closed alpha-shape proxy surface.

    Returns a completed canonical dict in the av2 schema (xyz/intensity/...)
    with ``xyz`` = area-weighted samples of the closed proxy — dense enough
    for Gaussian initialization without light leaks. Original shell points are
    kept in ``extra`` for provenance.
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(canon["xyz"], dtype=np.float64)
    intensity = np.asarray(canon.get("intensity",
                                    np.zeros(len(xyz))), dtype=np.float64)
    shell = mirror_half_shell(xyz)
    method = "alpha_shape_mirror"
    try:
        mesh = alpha_shape_mesh(shell)
    except ValueError:
        mesh = None
    if mesh is None or not mesh.is_watertight:
        # half-shells leave gaps the alpha filter cannot close; the convex
        # hull of the mirrored shell is the reliable dims-correct proxy
        mesh = trimesh.convex.convex_hull(shell)
        method = "convex_hull_mirror"

    pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    # color/intensity: nearest original shell point (keeps reflectance identity)
    dist, idx = cKDTree(xyz).query(pts, k=1)
    inten = intensity[idx]

    # observation stats recomputed from the *proxy* (closed => full coverage).
    # Canonical xyz is bbox-min translated to the origin, so azimuth must be
    # measured about the cloud center — raw atan2 would only see one quadrant.
    center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    theta = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    bins = np.floor((theta + np.pi) / (2 * np.pi) * 360).astype(int) % 360
    az_span = float(len(np.unique(bins)))
    area = float(mesh.area) if mesh.area > 0 else 1.0

    out = {
        "xyz": pts.astype(np.float32),
        "intensity": inten.astype(np.uint8),
        "raw_bbox_min": np.asarray(canon.get("raw_bbox_min", xyz.min(0)),
                                   dtype=np.float32),
        "raw_bbox_size": (pts.max(0) - pts.min(0)).astype(np.float32),
        "box_lwh": np.asarray(canon.get("box_lwh",
                                        pts.max(0) - pts.min(0)), dtype=np.float32),
        "cloud_extent": (pts.max(0) - pts.min(0)).astype(np.float32),
        "ego": np.asarray(canon.get("ego", np.zeros((0, 3))), dtype=np.float32),
        "cat": np.asarray(str(canon.get("cat", ""))),
        "shell_xyz": xyz.astype(np.float32),   # provenance: original half-shell
        "proxy_method": np.array(method),
    }
    stats = {
        "method": method,
        "n_proxy_samples": int(n_samples),
        "n_shell": int(len(xyz)),
        "n_mesh_faces": int(len(mesh.faces)),
        "watertight_after": bool(mesh.is_watertight),
        "az_span_deg": az_span,
        "pts_per_m2": float(n_samples / area),
        "surface_m2": area,
        "knn_median_m": float(np.median(
            cKDTree(pts).query(pts, k=min(9, len(pts)))[0][:, 1:])),
    }
    return out, stats
