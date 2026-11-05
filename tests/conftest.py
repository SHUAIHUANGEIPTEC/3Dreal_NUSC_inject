"""Shared pytest fixtures.

All tests are CPU-only and must pass on a machine without the datasets: the
``assets_root``/``dataset_root``/``inject_root`` fixtures auto-skip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cfg_assets() -> dict:
    from realinject import paths

    return paths.load_config("configs/assets_ingest.yaml")


@pytest.fixture(scope="session")
def assets_root(cfg_assets) -> Path:
    root = Path(str((cfg_assets.get("assets") or {}).get("root", "")))
    if not root.is_dir() or "${" in str(root):
        pytest.skip("3D asset corpus not configured (local.yaml / ASSETS_ROOT)")
    return root


@pytest.fixture(scope="session")
def library_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("library")


@pytest.fixture(scope="session")
def dataset_root(cfg_assets) -> Path:
    from realinject import paths

    cfg = paths.load_config("configs/base.yaml")
    root = Path(str((cfg.get("dataset") or {}).get("root", "")))
    if not root.is_dir() or "${" in str(root):
        pytest.skip("nuScenes dataset not configured (local.yaml / NUSCENES_ROOT)")
    return root


@pytest.fixture(scope="session")
def inject_root(cfg_assets) -> Path:
    from realinject import paths

    cfg = paths.load_config("configs/base.yaml")
    root = paths.resolve((cfg.get("inject_nusc") or {}).get("root", ""))
    if not (root / "src").is_dir():
        pytest.skip("sibling nusc-inject repo not available")
    return root
