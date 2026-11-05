"""Diffusion package contract: schema, stub idempotence, gauss trace."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from realinject.diffusion import FRAME_ROUTES, get_backend, read_package, write_package
from realinject.diffusion.backend import assert_gauss_trace, placeholder_run


def test_write_package_schema(tmp_path):
    pkg = write_package(
        tmp_path / "p", backend="viewcrafter", role="realism",
        command="python scripts/gpu/run_viewcrafter.py --package p",
        inputs={"traj": "0 15 30\n0 0 0\n0 0 0\n",
                "renderings": np.zeros((4, 8, 8, 3), np.float32)},
        meta={"condition_sources": ["/x/gauss/CAM_FRONT.png"],
              "renderer": "realinject.gauss.raster_cpu"})
    assert (pkg.path / "inputs" / "traj").exists()
    assert (pkg.path / "inputs" / "renderings.npy").exists()
    assert (pkg.path / "command.sh").exists()
    info = json.loads((pkg.path / "backend.json").read_text())
    assert info["protocol"] == "3dreal-diffusion/1"
    assert info["backend"] == "viewcrafter"
    loaded = read_package(pkg.path)
    assert loaded.backend == "viewcrafter"
    assert loaded.meta["renderer"].startswith("realinject.gauss")


def test_write_package_copies_file_inputs(tmp_path):
    # keys ARE the target filenames inside inputs/ (GPU drivers rely on them)
    src = tmp_path / "_ref_src.jpg"
    src.write_bytes(b"\x89PNG fake")
    pkg = write_package(tmp_path / "p2", backend="x", role="realism",
                        command="true", inputs={"ref.png": src})
    assert (pkg.path / "inputs" / "ref.png").read_bytes() == b"\x89PNG fake"


def test_gauss_trace_rejects_blender_era():
    with pytest.raises(RuntimeError, match="Gaussian"):
        assert_gauss_trace({"condition_sources": ["/x/blender/CAM_FRONT.png"],
                            "renderer": "blender"})


def test_gauss_trace_requires_renderer_field():
    with pytest.raises(RuntimeError, match="renderer"):
        assert_gauss_trace({"condition_sources": ["/x/gauss/CAM_FRONT.png"]})


def test_placeholder_run_is_stable():
    a = placeholder_run("viewcrafter")
    b = placeholder_run("viewcrafter")
    assert a == b
    assert a["status"] == "placeholder"


def test_registry_routes():
    for name in (*FRAME_ROUTES, "complete", "magicdrive"):
        b = get_backend(name)
        assert b.name == name
        ok, why = b.available()
        assert ok is False  # all routes need the GPU server for real inference
        assert why
    with pytest.raises(ValueError):
        get_backend("nope")


@pytest.mark.parametrize("route", list(FRAME_ROUTES))
def test_frame_route_prepare_requires_gauss_frames(route, tmp_path, cfg_stub):
    """prepare() must refuse a batch that has no Gaussian frames."""
    backend = get_backend(route)
    with pytest.raises(FileNotFoundError, match="Gaussian"):
        backend.prepare(cfg_stub, tmp_path / route, run_id="missing",
                        scene="scene-x", frame=0, cam="CAM_FRONT")


@pytest.fixture
def cfg_stub():
    return {"output_dir": str(_tmp_root()), "library": {"path": "data/library"},
            "dataset": {"root": ".", "version": "v1.0-mini"},
            "assets": {"root": "."}}


def _tmp_root():
    import tempfile

    return Path(tempfile.mkdtemp()) / "outputs"


def test_gpu_scripts_compile():
    import py_compile
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    scripts = sorted((root / "scripts" / "gpu").glob("*.py"))
    assert scripts, "no GPU scripts found"
    for s in scripts:
        py_compile.compile(str(s), doraise=True)
