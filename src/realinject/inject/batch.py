"""Batch driver: cross-frame tracking over scenes, then one export.

Ports the ``run_batch.run_tracking`` loop semantics (frozen world pose per
track, prev/next via track keys, terminate on invisible/no-camera/validation
failure) onto ``pipeline_gauss.run_sample``.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from . import placement
from .pipeline_gauss import run_sample

log = logging.getLogger("batch")


def run_tracking(cfg: dict, meta, records: list, batch_dir: Path,
                 library_root: str, max_frames: int | None = None) -> tuple[list, dict]:
    """Run the tracking loop; returns (specs, summary)."""
    seed = int((cfg.get("objects") or {}).get("seed", 42))
    scene_filter = cfg["dataset"].get("scenes") or [cfg["dataset"].get("scene")]
    scenes = [s for s in meta.scene.values()
              if scene_filter is None or s["name"] in scene_filter]
    scenes.sort(key=lambda s: meta.sample_idx_by_token[s["first_sample_token"]])
    if not scenes:
        raise SystemExit(f"no scenes matched filter {scene_filter}")

    specs: list = []
    track_stats: list = []
    total_ok = 0
    for si, scene in enumerate(scenes):
        samples = meta.scene_samples(scene)
        if max_frames:
            samples = samples[:int(max_frames)]
        rng = np.random.default_rng([seed, si])
        tracks = placement.plan_tracks(cfg, scene["name"], len(samples), rng, records,
                                          library_root=library_root)
        log.info("=== scene %s (%d frames, %d tracks) ===",
                 scene["name"], len(samples), len(tracks))

        cloud_cache: dict = {}
        for fi, sample in enumerate(samples):
            active = [t for t in tracks
                      if not t["terminated"]
                      and t["start"] <= fi < t["start"] + t["length"]]
            if not active:
                continue
            objects, tkeys = [], []
            for t in active:
                if t["T"] is None:
                    objects.append({"model_cfg": t["model_cfg"],
                                    "placement_cfg": t["placement_cfg"]})
                else:
                    objects.append({
                        "model_cfg": t["model_cfg"],
                        "placement_cfg": {
                            "from_global": t["T"],
                            "ground": t["placement_cfg"].get("ground", {}),
                        },
                    })
                tkeys.append(t["key"])

            out_dir = batch_dir / scene["name"] / f"f{fi:03d}"
            idx = meta.sample_idx_by_token[sample["token"]]
            res = run_sample(cfg, meta, idx, objects, out_dir,
                             cloud_cache=cloud_cache, track_keys=tkeys,
                             drop_invisible=True)
            status = res.get("status", "ok")
            dropped_keys = {tkeys[i] for i in res.get("dropped", [])}

            if status == "no_visible":
                for t in active:
                    t["terminated"] = True
                    t["end_reason"] = "invisible"
                log.info("  frame %d: all tracks invisible -> close", fi)
                continue
            if status in ("no_camera", "validation_failed"):
                for t in active:
                    t["terminated"] = True
                    t["end_reason"] = status
                log.info("  frame %d: %s -> close %d tracks",
                         fi, status, len(active))
                continue

            for t in active:
                if t["key"] in dropped_keys:
                    t["terminated"] = True
                    t["end_reason"] = "invisible"

            ann_keys = [o["track_key"] for o in res["spec"]["objects"]]
            for t in active:
                if t["T"] is None and t["key"] in ann_keys:
                    gi = ann_keys.index(t["key"])
                    t["T"] = res["gts"][gi]["T_model2global"]
                if t["key"] not in ann_keys and t["T"] is None:
                    t["terminated"] = True
                    t["end_reason"] = "invisible"

            specs.append(res["spec"])
            total_ok += 1
            log.info("  frame %d annotated: %d objects (%s)",
                     fi, len(ann_keys), ann_keys)

        for t in tracks:
            if not t["terminated"]:
                t["end_reason"] = "window_end"
            track_stats.append({
                "key": t["key"], "asset_id": t["asset_id"],
                "start": t["start"], "planned_length": t["length"],
                "end_reason": t["end_reason"], "T_resolved": t["T"] is not None,
            })

    lengths = Counter(o["track_key"] for s in specs for o in s["objects"])
    summary = {
        "scenes": [s["name"] for s in scenes],
        "frames_annotated": total_ok,
        "n_tracks_planned": len(track_stats),
        "n_tracks_realized": len(lengths),
        "track_length_hist": dict(Counter(lengths.values())),
        "end_reasons": dict(Counter(t["end_reason"] for t in track_stats)),
        "tracks": track_stats,
    }
    return specs, summary


def write_batch(batch_dir: Path, specs: list, summary: dict, run_id: str) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "specs.json").write_text(json.dumps(
        {"run_id": run_id, "n_samples": len(specs), "specs": specs}, indent=2))
    (batch_dir / "summary.json").write_text(json.dumps(summary, indent=2))
