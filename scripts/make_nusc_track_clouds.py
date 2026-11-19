#!/usr/bin/env python
"""Build av2_assets track clouds from the LOCAL nuScenes mini (offline substitute).

The original Argoverse2 per-track clouds (``av2_assets/_merged/<CAT>_<id>.npz``)
were extracted on the old workstation and are not re-downloadable as an
intermediate product. This script rebuilds the SAME file contract from the
local nuScenes annotations + LiDAR: for every instance whose category maps to
an AV2 class, it aggregates all its cuboid points across keyframes into the
object-local frame (box-centred, Z-up) and writes:

    <assets.root>/av2_assets/_merged/<CAT>_<track8>.npz
        pts (N,3) float  — object-local, Z-up, bbox-centred
        intensity (N,) uint8
        dims (3,) box LWH  (median cuboid, nuScenes [w,l,h] -> (l,w,h))
        ego (F,3) per-frame lidar origin in the FIRST-frame lidar frame
        cat  — AV2 class name
    <assets.root>/av2_assets/_merged/manifest.json
        [{track, cat, n, az_span_deg, pts_per_m2, coverage, ...}]  (gate stats)

Gate window (configs/assets_ingest.yaml -> gates.av2):
    n >= 300, az_span >= 90 deg, density >= 150 pts/m^2, coverage >= 0.01.

Usage:
    python scripts/make_nusc_track_clouds.py --top-per-class 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realinject import paths  # noqa: E402

# nuScenes category name -> AV2 class (ingest_av2.AV2_CATEGORY_MAP keys)
NUSC2AV2 = {
    "vehicle.car": "REGULAR_VEHICLE",
    "vehicle.motorcycle": "MOTORCYCLE",
    "human.pedestrian.stroller": "STROLLER",
    "human.pedestrian.wheelchair": "WHEELCHAIR",
    "human.pedestrian.personal_mobility": "WHEELED_DEVICE",
    "animal": "DOG",
}


def quat_to_mat(q_xyzw) -> np.ndarray:
    x, y, z, w = q_xyzw
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def az_span_deg(pts_xy: np.ndarray) -> float:
    """Angular coverage of the cloud: 360 - largest empty sector (10 deg bins)."""
    if len(pts_xy) == 0:
        return 0.0
    ang = np.degrees(np.arctan2(pts_xy[:, 1], pts_xy[:, 0])) % 360.0
    bins = np.zeros(36, dtype=bool)
    bins[(ang // 10).astype(int) % 36] = True
    if bins.all():
        return 360.0
    # largest run of empty bins (circular)
    two = np.concatenate([bins, bins])
    best = run = 0
    for b in two:
        run = 0 if b else run + 1
        best = max(best, run)
    return float(360.0 - min(best, 36) * 10.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/inject_gauss.yaml")
    ap.add_argument("--top-per-class", type=int, default=6,
                    help="keep the N densest tracks per AV2 class")
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--pad-m", type=float, default=0.10,
                    help="cuboid clip padding (grow the box a little)")
    ap.add_argument("--assets-root", default=None)
    ap.add_argument("--limit-instances", type=int, default=0)
    args = ap.parse_args(argv)

    from nuscenes.nuscenes import NuScenes

    cfg = paths.load_config(ROOT / args.config)
    assets_root = Path(args.assets_root or cfg["assets"]["root"])
    out_dir = assets_root / "av2_assets" / "_merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=cfg["dataset"]["version"],
                    dataroot=str(cfg["dataset"]["root"]), verbose=True)

    # category name -> AV2 class
    cat2av2 = {c["name"]: NUSC2AV2[c["name"]]
               for c in nusc.category if c["name"] in NUSC2AV2}
    print(f"mapped categories: {cat2av2}")

    # ---- collect instances with >= min-frames annotations ----
    anns = {a["token"]: a for a in nusc.sample_annotation}
    inst_anns: dict[str, list] = {}
    for a in nusc.sample_annotation:
        inst_anns.setdefault(a["instance_token"], []).append(a)
    def _time_chain(inst) -> list:
        """Annotations in temporal order via the official first->next chain."""
        out, tok = [], inst.get("first_annotation_token") or ""
        while tok:
            a = anns[tok]
            out.append(a)
            tok = a.get("next") or ""
        return out

    candidates = []
    for inst in nusc.instance:
        cname = nusc.get("category", inst["category_token"])["name"]
        if cname not in cat2av2:
            continue
        chain = _time_chain(inst)
        if len(chain) < args.min_frames:
            continue
        candidates.append((inst, cname, chain))
    print(f"instances with >= {args.min_frames} frames in mapped classes: "
          f"{len(candidates)}")
    if args.limit_instances:
        candidates = candidates[:args.limit_instances]

    # ---- per instance: aggregate cuboid points across frames ----
    # keyframe LIDAR_TOP sample_data per sample (sensor_modality is NOT a raw
    # sample_data field — index by the samples/LIDAR_TOP filename instead)
    lidar_sd_of = {sd["sample_token"]: sd for sd in nusc.sample_data
                   if sd["filename"].startswith("samples/LIDAR_TOP")}
    sd_cache: dict[str, dict] = {}
    results = []  # (entry, pts, intensity, dims, ego, cat, track)
    for inst, cname, chain in candidates:
        av2_cat = cat2av2[cname]
        pts_all, inten_all, egos, sizes = [], [], [], []
        first_T_inv = None
        ok = True
        for a in chain:                  # already temporal (first->next chain)
            sd = lidar_sd_of.get(a["sample_token"])
            if sd is None:
                ok = False
                break
            key = sd["token"]
            if key not in sd_cache:
                calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                pose = nusc.get("ego_pose", sd["ego_pose_token"])
                T_l2g = np.eye(4)
                T_l2g[:3, :3] = quat_to_mat(pose["rotation"])
                T_l2g[:3, 3] = pose["translation"]
                T_c = np.eye(4)
                T_c[:3, :3] = quat_to_mat(calib["rotation"])
                T_c[:3, 3] = calib["translation"]
                sd_cache[key] = {"path": nusc.get_sample_data_path(sd["token"]),
                                 "T": T_l2g @ T_c}   # lidar -> global
            meta = sd_cache[key]
            pts = np.fromfile(meta["path"], dtype=np.float32).reshape(-1, 5)
            if first_T_inv is None:
                first_T_inv = np.linalg.inv(meta["T"])
            # ego (lidar origin) of this frame in FIRST-frame lidar coords
            egos.append((first_T_inv @ np.array([*meta["T"][:3, 3], 1.0]))[:3])

            # cuboid clip: lidar -> box-local (box frame: x=l, y=w, z=h, Z-up)
            T_b2g = np.eye(4)
            T_b2g[:3, :3] = quat_to_mat(a["rotation"])
            T_b2g[:3, 3] = a["translation"]
            T_b = np.linalg.inv(meta["T"]) @ T_b2g
            local = pts[:, :3] @ T_b[:3, :3].T + T_b[:3, 3]
            wid, lng, hgt = a["size"]     # nuScenes: [w, l, h]
            m = args.pad_m
            hit = ((np.abs(local[:, 0]) <= lng / 2 + m)
                   & (np.abs(local[:, 1]) <= wid / 2 + m)
                   & (np.abs(local[:, 2]) <= hgt / 2 + m))
            if hit.sum() < 20:
                continue
            pts_all.append(local[hit])
            inten_all.append(np.clip(pts[hit, 3], 0, 255).astype(np.uint8))
            sizes.append((lng, wid, hgt))

        if not ok or len(pts_all) < args.min_frames:
            continue
        pts = np.vstack(pts_all)
        inten = np.concatenate(inten_all)
        dims = np.median(np.asarray(sizes), axis=0)      # (L, W, H)
        track = inst["token"][:8]
        area = max(float(dims[0] * dims[1]), 1e-6)
        az = az_span_deg(pts[:, :2])
        entry = {"track": track, "cat": av2_cat, "nuc_category": cname,
                 "instance_token": inst["token"], "n_frames": len(pts_all),
                 "n": int(len(pts)), "az_span_deg": round(az, 1),
                 "pts_per_m2": round(len(pts) / area, 1),
                 "coverage": round(az / 360.0, 3),
                 "dims_lwh": [round(float(v), 2) for v in dims]}
        npz_path = out_dir / f"{av2_cat}_{track}.npz"
        np.savez_compressed(npz_path, pts=pts.astype(np.float32),
                            intensity=inten, dims=dims.astype(np.float32),
                            ego=np.asarray(egos, dtype=np.float32),
                            cat=np.array(av2_cat))
        results.append(entry)

    if not results:
        print("FATAL: no track produced (check min-frames / mappings)")
        return 2

    # ---- keep the top-N densest tracks per class (gate friendly) ----
    keep = []
    for cat in sorted({r["cat"] for r in results}):
        rows = [r for r in results if r["cat"] == cat]
        rows.sort(key=lambda r: -r["n"])
        keep += rows[:args.top_per_class]
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(keep, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {manifest}: {len(keep)} tracks "
          f"(from {len(results)} produced)")
    for r in keep:
        flag = "PASS" if (r["n"] >= 300 and r["az_span_deg"] >= 90
                          and r["pts_per_m2"] >= 150) else "weak"
        print(f"  {r['cat']:18s} {r['track']}  n={r['n']:6d} "
              f"az={r['az_span_deg']:5.1f} dens={r['pts_per_m2']:7.1f} "
              f"[{flag}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
