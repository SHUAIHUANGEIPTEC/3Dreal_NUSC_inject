#!/usr/bin/env python
"""Inject real extracted assets into one nuScenes scene, harmonize, export.

    conda activate nusc-inject
    python scripts/run_inject_gauss.py --run-id gs01 --scene scene-0061 --samples 5
    python scripts/run_inject_gauss.py --run-id gs01 --skip-export   # batch only
    python scripts/run_inject_gauss.py --run-id gs01 --skip-harmonize

The e2e flow is batch (inject) -> harmonize (exposure match / contact shadow,
writes refined/<cam>.jpg + harmonize_report.json) -> export.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realinject import paths  # noqa: E402
from realinject.assets.manifest import AssetLibrary  # noqa: E402
from realinject.inject import batch, bridge  # noqa: E402
from realinject.inject.export import export as export_dataset  # noqa: E402


def _run_harmonize(cfg: dict, batch_dir: Path, scenes: list[str],
                   log: logging.Logger) -> None:
    """P4 two-pass harmonization over the batch just written (e2e step)."""
    from realinject.harmonize import get_refiner, run_session

    hcfg = dict(paths.load_config("configs/harmonize.yaml").get("harmonize") or {})
    refiner = get_refiner(str(hcfg.get("refiner", "cpu")), cfg=hcfg)
    cams = cfg.get("cam_channels") or [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    merged: dict = {"scenes": {}, "refiner": None}
    for scene in scenes:
        report = run_session(batch_dir, scene, cams, hcfg, refiner,
                             write_report=False)
        merged["refiner"] = report.get("refiner", merged["refiner"])
        merged["scenes"].update(report.get("scenes") or {})
        for cam, m in sorted((report.get("scenes") or {}).get(scene, {}).items()):
            log.info("  harmonize %s: dE(target,ring) %.1f -> %.1f, "
                     "dL %.1f -> %.1f", cam,
                     m.get("dE_target_ring_before", float("nan")),
                     m.get("dE_target_ring_after", float("nan")),
                     m.get("lightness_gap_before", float("nan")),
                     m.get("lightness_gap_after", float("nan")))
    report_path = batch_dir / "harmonize_report.json"
    report_path.write_text(json.dumps(merged, indent=2))
    log.info("harmonize: report at %s", report_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/inject_gauss.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--scene", default=None, help="override dataset.scene")
    ap.add_argument("--samples", type=int, default=None,
                    help="max frames per scene (default: whole scene)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--assets", default=None,
                    help="comma-separated asset_ids to place "
                         "(e.g. av2/DOG_03ce7747,av2/STROLLER_e55913c8)")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-harmonize", action="store_true",
                    help="skip the P4 harmonization step (refined/ output)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("run_inject_gauss")

    cfg = paths.load_config(args.config)
    if args.run_id:
        cfg["run_id"] = args.run_id
    if args.scene:
        cfg["dataset"]["scene"] = args.scene
        cfg["dataset"].pop("scenes", None)
    if args.seed is not None:
        cfg.setdefault("objects", {})["seed"] = int(args.seed)
    run_id = str(cfg.get("run_id", "gs01"))

    bridge.bootstrap(cfg)

    from nusc_inject import nusc_io

    meta = nusc_io.NuScenesMini(cfg["dataset"]["root"], cfg["dataset"]["version"])
    library_root = Path(str((cfg.get("library") or {}).get("path", "data/library")))
    manifest = library_root / "assets_manifest.json"
    if not manifest.exists():
        print(f"FATAL: {manifest} missing — run scripts/build_asset_library.py first",
              file=sys.stderr)
        return 2
    lib = AssetLibrary.load(manifest)
    records = lib.passed()
    if args.assets:
        wanted = {a.strip() for a in args.assets.split(",") if a.strip()}
        records = [r for r in records if r.asset_id in wanted]
        missing = wanted - {r.asset_id for r in records}
        if missing:
            print(f"FATAL: asset_ids not in the passed library: {sorted(missing)}",
                  file=sys.stderr)
            return 2
        if not records:
            print("FATAL: --assets matched no passed records", file=sys.stderr)
            return 2
    else:
        # Default deal: only textured realcar meshes. av2_cloud records are
        # sparse untextured half-shells (see quality gates); --assets can
        # still request them explicitly.
        records = [r for r in records if r.kind == "realcar_mesh"]
    if not records:
        print("FATAL: no passed assets in the library", file=sys.stderr)
        return 2
    log.info("library: %d passed assets (%d kinds)%s", len(records),
             len({r.kind for r in records}),
             f" (filtered: {sorted({r.asset_id for r in records})})"
             if args.assets else "")

    batch_dir = Path(str(cfg.get("output_dir", "data/outputs"))) / run_id
    specs, summary = batch.run_tracking(
        cfg, meta, records, batch_dir, str(library_root),
        max_frames=args.samples)
    batch.write_batch(batch_dir, specs, summary, run_id)
    log.info("batch: %d frames annotated, %d tracks realized",
             summary["frames_annotated"], summary["n_tracks_realized"])
    for length, n in sorted(summary.get("track_length_hist", {}).items()):
        log.info("  %d track(s) with %s annotated frames", n, length)

    if not args.skip_harmonize and summary.get("frames_annotated"):
        log.info("harmonize: exposure match + contact shadow -> refined/")
        _run_harmonize(cfg, batch_dir, summary.get("scenes") or [], log)

    if args.skip_export:
        log.info("skip export; specs at %s", batch_dir / "specs.json")
        return 0

    export_root = Path(str(cfg.get("export_dir", "data/exports"))) / run_id
    result = export_dataset(cfg, specs, export_root)
    verify = result.get("verify", {})
    log.info("export: %s", result["export_root"])
    log.info("  new annotations: %d across %d samples",
             result["n_new_annotations"], result["n_samples_injected"])
    log.info("  categories: %s", result.get("category_names"))
    log.info("  devkit verify: %s", verify.get("problems") or "ok")
    if verify.get("problems"):
        for p in verify["problems"]:
            log.error("  verify problem: %s", p)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
