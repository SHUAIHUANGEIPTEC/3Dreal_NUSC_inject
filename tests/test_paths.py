"""Path/config resolution tests (portability contract)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from realinject import paths


def test_project_root_is_repo_root():
    # src/realinject/paths.py must use parents[2]; parents[1] would be src/
    assert (paths.PROJECT_ROOT / "pyproject.toml").exists()
    assert paths.PROJECT_ROOT.name == "3Dreal-nusc-inject"


def test_resolve_relative_is_project_root_anchored():
    assert paths.resolve("configs/base.yaml") == paths.PROJECT_ROOT / "configs/base.yaml"
    assert paths.resolve("data/library").is_absolute()


def test_resolve_absolute_passthrough(tmp_path):
    p = paths.resolve(str(tmp_path / "x"))
    assert p == tmp_path / "x"


def test_resolve_env_expansion(monkeypatch):
    monkeypatch.setenv("SOME_ROOT", "/tmp/abc")
    assert paths.resolve("${SOME_ROOT}/lib") == Path("/tmp/abc/lib")


def test_deep_merge_nested():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = paths._deep_merge(base, {"a": {"c": 9}})
    assert out == {"a": {"b": 1, "c": 9}, "d": 3}
    assert base["a"]["c"] == 2  # base untouched


def test_load_config_merges_local_yaml(tmp_path, monkeypatch):
    (tmp_path / "cfg.yaml").write_text("a:\n  b: 1\n")
    (tmp_path / "local.yaml").write_text("a:\n  b: 2\n")
    cfg = paths.load_config("cfg.yaml", root=tmp_path)
    assert cfg["a"]["b"] == 2


def test_resolve_cfg_paths_resolves_known_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("NUSC", str(tmp_path / "nusc"))
    cfg = paths.resolve_cfg_paths({"dataset": {"root": "${NUSC}"}, "unknown": "rel/x"})
    assert Path(cfg["dataset"]["root"]) == tmp_path / "nusc"
    assert cfg["unknown"] == "rel/x"


@pytest.mark.skipif(os.environ.get("CI") == "1", reason="local repo probe")
def test_inject_package_detection(inject_root):
    from realinject import paths as p

    cfg = {"inject_nusc": {"root": str(inject_root)}}
    root = p.add_inject_nusc_path(cfg)
    assert (root / "src").is_dir()
    import importlib

    for alias in ("nusc_inject", "inject_nusc"):
        assert importlib.import_module(alias).__name__ in ("nusc_inject", "inject_nusc")
