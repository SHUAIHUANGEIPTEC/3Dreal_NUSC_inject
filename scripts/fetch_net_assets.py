#!/usr/bin/env python
"""Fetch public Objaverse vehicle meshes (hf-mirror) into the realcar_assets layout.

The 3DRealCar corpus is a proprietary iPhone-scan release (not downloadable);
this script substitutes **public CC vehicle meshes** from Objaverse v1 over the
hf-mirror mirror (the same channel ``nusc-inject/scripts/download_road_ood.py``
uses) and lays them out in the exact file contract of
``ingest_realcar.py`` / ``docs/ASSETS.md``::

    <assets.root>/realcar_assets/<scene>/mesh.obj|mesh.mtl|mesh.jpg|meta.json
    <assets.root>/realcar_assets_manifest.json
        [{scene, grade: "A", L, W, H, nv, nf}, ...]

``L/W/H`` are computed with the very functions ``canonicalize_realcar`` uses
(``yaw_align`` + ``yup_to_blender`` extents), so the manifest cross-check
(``bbox_manifest_tol``) is exact by construction. Gate window:
L 3.5-6.5, W 1.4-2.6, H 1.0-2.0, texture std > 5.

Usage:
    python scripts/fetch_net_assets.py --want 20 --classes car,sports_car,truck
    python scripts/fetch_net_assets.py --report-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realinject import paths  # noqa: E402
from realinject.assets.normalize import yaw_align, yup_to_blender  # noqa: E402

BASE = "https://hf-mirror.com/datasets/allenai/objaverse/resolve/main"
CACHE = Path.home() / ".cache" / "objaverse-mirror"
GLB_DIR = CACHE / "glbs"
ALLOWED_LICENSES = {"by", "cc0"}
# vehicle classes whose members plausibly sit inside the realcar gate window
DEFAULT_CLASSES = "car,sports_car,truck,cab_(taxi),van"

GATE_L = (3.5, 6.5)
GATE_W = (1.4, 2.6)
GATE_H = (1.0, 2.0)


def fetch(rel: str) -> Path:
    dst = CACHE / rel
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    cmd = ["curl", "-sSL", "--fail", "--retry", "3", "--max-time", "300",
           "-o", str(part), f"{BASE}/{rel}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not part.exists() or part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"download failed: {rel} :: {r.stderr.strip()[:200]}")
    part.replace(dst)
    return dst


def load_gz_json(rel: str):
    import gzip
    return json.load(gzip.open(fetch(rel), "rt", encoding="utf-8"))


def glb_images_count(path: Path) -> int | None:
    """Number of image resources in the glTF JSON chunk; None if not a glb."""
    try:
        with open(path, "rb") as f:
            head = f.read(20)
            if len(head) < 20 or head[:4] != b"glTF" or head[16:20] != b"JSON":
                return None
            g = json.loads(f.read(int.from_bytes(head[12:16], "little"))
                           .decode("utf-8").rstrip(" \t\r\n\0"))
        return len(g.get("images") or [])
    except Exception:
        return None


def download_glb(rel: str, stem: str) -> Path | None:
    GLB_DIR.mkdir(parents=True, exist_ok=True)
    dst = GLB_DIR / f"{stem}.glb"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    part = dst.with_suffix(".glb.part")
    r = subprocess.run(["curl", "-sSL", "--fail", "--retry", "2",
                        "--max-time", "180", "-o", str(part), f"{BASE}/{rel}"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not part.exists() or part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        return None
    part.replace(dst)
    return dst


def load_merged_mesh(glb_path: Path):
    """Scene -> single mesh with per-vertex UV (None when unusable)."""
    import trimesh
    scene = trimesh.load(str(glb_path), force="scene")
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    geoms = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh)
             and len(g.faces)]
    if not geoms:
        return None, None
    vs, fs, uvs = [], [], []
    off = 0
    for g in geoms:
        uv = getattr(g.visual, "uv", None)
        if uv is None or len(uv) != len(g.vertices):
            return None, None          # canonicalize_realcar hard-requires uv
        vs.append(np.asarray(g.vertices, dtype=float))
        uvs.append(np.asarray(uv, dtype=float))
        fs.append(np.asarray(g.faces, dtype=np.int32) + off)
        off += len(g.vertices)
    verts = np.vstack(vs)
    faces = np.vstack(fs)
    uv = np.vstack(uvs)

    # texture: the first baseColor image found among the parts
    tex = None
    for g in geoms:
        mat = getattr(g.visual, "material", None)
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if img is not None:
            tex = img
            break
    if tex is None:
        return None, None              # texture_flat gate would reject anyway

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=None)
    return mesh, tex


def gate_dims(mesh) -> tuple[float, float, float]:
    """Yaw-aligned blender-frame extents — the exact canonicalize path."""
    verts = np.asarray(mesh.vertices, dtype=float)
    t, _ = yaw_align(verts)
    b = yup_to_blender((t[:3, :3] @ verts.T).T)
    bsize = b.max(axis=0) - b.min(axis=0)
    return tuple(float(v) for v in bsize)


def write_realcar_layout(mesh, tex, out_dir: Path, meta: dict,
                         dims: tuple[float, float, float]) -> None:
    """mesh.obj + mesh.mtl + mesh.jpg + meta.json (trimesh-readable)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(mesh.vertices, dtype=float)
    uv = np.asarray(mesh.visual.uv, dtype=float)
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    tex_path = out_dir / "mesh.jpg"
    if getattr(tex, "mode", "RGB") != "RGB":
        tex = tex.convert("RGB")
    tex.save(tex_path, quality=92)

    mtl = out_dir / "mesh.mtl"
    mtl.write_text("newmtl material0\nKd 1.000 1.000 1.000\n"
                   "Ka 0.000 0.000 0.000\nd 1.0\nillum 1\nmap_Kd mesh.jpg\n",
                   encoding="utf-8")
    lines = ["mtllib mesh.mtl", "o mesh"]
    lines += [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in verts]
    lines += [f"vt {t[0]:.6f} {t[1]:.6f}" for t in uv]
    lines += [f"vn {n[0]:.5f} {n[1]:.5f} {n[2]:.5f}" for n in normals]
    lines.append("usemtl material0")
    for f in faces:
        a, b, c = (int(i) + 1 for i in f)
        lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
    (out_dir / "mesh.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {**meta, "dims_lwh_m": [round(v, 3) for v in dims],
            "nv": int(len(verts)), "nf": int(len(faces))}
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--classes", default=DEFAULT_CLASSES)
    ap.add_argument("--want", type=int, default=20,
                    help="number of cars passing the size gate")
    ap.add_argument("--max-try", type=int, default=150, help="download cap")
    ap.add_argument("--min-mb", type=float, default=1.0)
    ap.add_argument("--max-mb", type=float, default=30.0)
    ap.add_argument("--assets-root", default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)

    cfg = paths.load_config(ROOT / "configs" / "assets_ingest.yaml")
    assets_root = Path(args.assets_root or cfg["assets"]["root"])
    rc_dir = assets_root / cfg["assets"].get("realcar_dir", "realcar_assets")
    man_path = assets_root / cfg["assets"].get(
        "realcar_manifest", "realcar_assets_manifest.json")

    print(f"assets root: {assets_root}")
    print("fetching LVIS annotations + object paths ...")
    lvis = load_gz_json("lvis-annotations.json.gz")
    opaths = load_gz_json("object-paths.json.gz")

    wanted: dict[str, str] = {}
    for cls in (c.strip() for c in args.classes.split(",")):
        for uid in lvis.get(cls, []):
            wanted.setdefault(uid, cls)
    print(f"candidate uids: {len(wanted)} ({args.classes})")

    rows = []
    shard_of = {u: opaths[u].split("/")[1] for u in wanted}
    for shard in sorted(set(shard_of.values())):
        try:
            meta_all = load_gz_json(f"metadata/{shard}.json.gz")
        except Exception as exc:
            print(f"  ! {shard}: {exc}")
            continue
        for uid, cls in wanted.items():
            if shard_of[uid] != shard:
                continue
            m = meta_all.get(uid)
            if (not m or m.get("license") not in ALLOWED_LICENSES
                    or m.get("isDownloadable") is False):
                continue
            a = (m.get("archives") or {}).get("glb") or {}
            if (a.get("textureCount") or 0) <= 0:
                continue
            est_mb = (a.get("size") or 0) / 3.0 / 1e6
            if est_mb < args.min_mb or est_mb > args.max_mb:
                continue
            rows.append({"uid": uid, "cls": cls, "path": opaths[uid],
                         "est_mb": est_mb,
                         "vertexCount": m.get("vertexCount") or 0,
                         "license": m.get("license"),
                         "name": m.get("name") or uid})
    rows.sort(key=lambda r: -r["vertexCount"])
    print(f"metadata-filtered candidates: {len(rows)}")
    if args.report_only:
        return 0

    if man_path.exists():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    else:
        manifest = []
    have = {e["scene"] for e in manifest}
    print(f"manifest already holds {len(have)} scenes")

    n_ok, n_try = 0, 0
    for r in rows:
        if n_ok >= args.want or n_try >= args.max_try:
            break
        safe_cls = r["cls"].replace("(", "_").replace(")", "").replace(",", "_")
        scene = f"{safe_cls}_{r['uid'][:8]}"
        if scene in have:
            continue
        n_try += 1
        glb = download_glb(r["path"], f"{safe_cls}_{r['uid'][:8]}")
        if glb is None:
            print(f"  [{n_try}] {r['uid'][:8]} download failed")
            continue
        n_img = glb_images_count(glb)
        if not n_img:
            print(f"  [{n_try}] {r['uid'][:8]} not a textured glb")
            continue
        try:
            mesh, tex = load_merged_mesh(glb)
        except Exception as exc:
            print(f"  [{n_try}] {r['uid'][:8]} load failed: {type(exc).__name__}")
            continue
        if mesh is None:
            print(f"  [{n_try}] {r['uid'][:8]} no uv/single texture")
            continue
        L, W, H = gate_dims(mesh)
        if not (GATE_L[0] <= L <= GATE_L[1] and GATE_W[0] <= W <= GATE_W[1]
                and GATE_H[0] <= H <= GATE_H[1]):
            print(f"  [{n_try}] {r['uid'][:8]} size {L:.1f}x{W:.1f}x{H:.1f} m "
                  f"outside gate")
            continue
        write_realcar_layout(
            mesh, tex, rc_dir / scene,
            {"uid": r["uid"], "lvis_class": r["cls"], "name": r["name"],
             "license": r["license"], "source": "Objaverse v1 (hf-mirror)",
             "n_images": n_img}, (L, W, H))
        # manifest L/W/H with the SAME functions as canonicalize_realcar ->
        # bbox_manifest_tol cross-check is exact by construction
        manifest.append({"scene": scene, "grade": "A",
                         "L": round(L, 3), "W": round(W, 3), "H": round(H, 3),
                         "nv": int(len(mesh.vertices)),
                         "nf": int(len(mesh.faces)),
                         "uid": r["uid"], "lvis_class": r["cls"]})
        n_ok += 1
        print(f"  [{n_try}] PASS {scene}: {L:.2f}x{W:.2f}x{H:.2f} m "
              f"({len(mesh.vertices)} verts)")

    assets_root.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {man_path} ({len(manifest)} scenes), +{n_ok} new "
          f"from {n_try} tries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
