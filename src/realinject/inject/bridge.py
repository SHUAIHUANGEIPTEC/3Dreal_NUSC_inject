"""Read-only bootstrap of the sibling nusc-inject library.

The upstream repo is never modified; we only import it. Contract asserts fail
fast with an actionable message if someone links an older revision (the
``composite_onto`` lidar->global fix and the export helpers are load-bearing).
"""
from __future__ import annotations

import inspect
from pathlib import Path

from ..paths import add_inject_nusc_path

#: private upstream helpers composite_ext reuses (depth hole/gap filling)
_COMPOSITE_HELPERS = ("_fill_depth_holes", "_dilate_scene_depth",
                      "_fill_radius_px", "_zbuffer_depth")
_ANNOTATION_HELPERS = ("build_sample_spec", "export_dataset", "verify_export",
                       "patch_devkit_colormap")


def bootstrap(cfg: dict) -> Path:
    """Put nusc-inject on sys.path and verify the contract; returns its root."""
    root = add_inject_nusc_path(cfg)
    assert_upstream_contract()
    return root


def assert_upstream_contract() -> None:
    """Fail fast if the linked nusc-inject revision lacks what we rely on."""
    import nusc_inject.annotation as annotation
    import nusc_inject.composite as composite
    import nusc_inject.lidar_channel as lidar_channel
    import nusc_inject.validate as validate

    sig = inspect.signature(composite.composite_onto)
    if "T_lidar2global" not in sig.parameters:
        raise RuntimeError(
            "linked nusc-inject composite.composite_onto lacks T_lidar2global "
            "(the lidar->global scene-depth fix). Point inject_nusc.root at a "
            "revision that includes it (nusc-inject >= the 2026-11 fix).")
    problems = []
    for mod, names in ((composite, _COMPOSITE_HELPERS),
                       (annotation, _ANNOTATION_HELPERS)):
        missing = [n for n in names if not hasattr(mod, n)]
        if missing:
            problems.append(f"{mod.__name__}: missing {missing}")
    if not hasattr(lidar_channel, "inject") or not hasattr(validate, "run"):
        problems.append("lidar_channel.inject / validate.run missing")
    if problems:
        raise RuntimeError("upstream contract broken: " + "; ".join(problems))
