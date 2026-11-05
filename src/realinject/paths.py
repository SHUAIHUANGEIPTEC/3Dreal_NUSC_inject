"""Path resolution and config loading for portable configs.

Ported from ``stag1_inject/src/paths.py``. Rules:

- ``~`` and ``${ENV_VAR}`` are expanded in path values.
- Relative paths resolve against the project root (repo top level), NOT the
  process cwd, so any script can be launched from anywhere.
- Absolute paths pass through unchanged.
- Machine-specific overrides live in ``<project root>/local.yaml`` (not in
  configs/), deep-merged over the base config. See ``local.yaml.example``.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

# src/realinject/paths.py -> parents[2] is the repo root.
# (A paths.py directly under src/ would need parents[1]; see stag1_inject.)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: config fields that hold filesystem paths (dotted notation)
PATH_FIELDS = (
    "dataset.root",
    "assets.root",
    "library.path",
    "inject_nusc.root",
    "model.path",
    "output_dir",
    "export_dir",
    "injection.batch_dir",
)


def resolve(path_like, root: Path | str | None = None) -> Path:
    """Resolve a config path value to an absolute Path.

    root: base for relative paths; defaults to PROJECT_ROOT.
    """
    if isinstance(path_like, Path):
        p = path_like
    else:
        p = Path(os.path.expandvars(os.path.expanduser(str(path_like))))
    if not p.is_absolute():
        p = Path(root or PROJECT_ROOT) / p
    return p


def _get_field(cfg: dict, dotted: str):
    node = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _set_field(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def _deep_merge(base: dict, override: dict) -> dict:
    """Return base updated by override (dicts merged recursively)."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def resolve_cfg_paths(cfg: dict) -> dict:
    """Resolve known path fields in place (env expansion + project-root anchor)."""
    for dotted in PATH_FIELDS:
        val = _get_field(cfg, dotted)
        if val is not None:
            _set_field(cfg, dotted, str(resolve(val)))
    for m in cfg.get("models") or []:
        if isinstance(m, dict) and "path" in m:
            m["path"] = str(resolve(m["path"]))
    return cfg


def load_config(path_like, root: Path | str | None = None) -> dict:
    """Load a YAML config, merge ``<project root>/local.yaml`` overrides, resolve paths."""
    root = Path(root or PROJECT_ROOT)
    path = resolve(path_like, root)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    local = root / "local.yaml"
    if local.exists():
        with open(local) as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)
    return resolve_cfg_paths(cfg)


#: package names the sibling repo has used for its injection library
INJECT_PKG_CANDIDATES = ("nusc_inject", "inject_nusc")
#: submodules this project imports from the sibling package
INJECT_SUBMODULES = (
    "nusc_io",
    "transforms",
    "composite",
    "mesh_points",
    "annotation",
    "pipeline",
    "anchor",
    "lidar_channel",
    "camera_channel",
    "validate",
    "visualize",
    "occluders",
    "range_image",
    "blender_render",
    "config",
)


def add_inject_nusc_path(cfg: dict) -> Path:
    """Put the sibling injection package on sys.path (read-only dependency).

    Auto-detects whichever package directory exists under ``<root>/src`` and
    aliases it in ``sys.modules`` under BOTH historical names (``nusc_inject``
    / ``inject_nusc``) so call sites keep working across the upstream rename.
    """
    import importlib
    import sys

    root = None
    for key in ("inject_nusc", "nusc_inject"):
        node = _get_field(cfg, f"{key}.root")
        if node:
            root = resolve(node)
            break
    if root is None:
        raise KeyError("config needs inject_nusc.root (or nusc_inject.root)")
    src = root / "src"
    pkg = next((n for n in INJECT_PKG_CANDIDATES if (src / n).is_dir()), None)
    if pkg is None:
        raise FileNotFoundError(
            f"no injection package ({' / '.join(INJECT_PKG_CANDIDATES)}) "
            f"under {src} — set inject_nusc.root in local.yaml "
            f"(see local.yaml.example)")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    pkg_mod = importlib.import_module(pkg)
    for alias in INJECT_PKG_CANDIDATES:
        if alias == pkg:
            continue
        sys.modules.setdefault(alias, pkg_mod)
        for sub in INJECT_SUBMODULES:
            try:
                real = importlib.import_module(f"{pkg}.{sub}")
            except ImportError:
                continue
            sys.modules.setdefault(f"{alias}.{sub}", real)
    return root
