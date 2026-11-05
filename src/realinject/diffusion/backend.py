"""Pluggable diffusion backends (complete / realism / harmonize / generate).

Execution model (ported from stag1_inject): every backend has three CPU-safe
verbs — ``available`` / ``prepare`` / ``collect`` — and one GPU verb
``run_local`` (real inference only on the server). ``prepare`` writes a
condition package:

    <package>/
      inputs/          ref.png, traj.txt, renderings.npy, mask.png, ...
      command.sh       exact command for the GPU server
      backend.json     provenance: backend, role, condition_sources, params
      output/          filled by collect() after the GPU run

CPU stubs keep the whole pipeline runnable without a GPU: ``run_local``
returns ``{"status": "placeholder"}`` and videos fall back to refined/composite.

Condition-source contract: every realism/harmonize package must trace back to
the **Gaussian renderer's** frames (``condition_sources`` contains a ``/gauss/``
path); asset-completion packages trace to the canonical asset library. Violations
raise at prepare() time so a Blender-era batch can never silently feed the
diffusion routes.
"""
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROTOCOL = "3dreal-diffusion/1"


@dataclass
class Package:
    """Handle to a prepared condition package."""

    path: Path
    backend: str
    role: str
    meta: dict = field(default_factory=dict)

    @property
    def inputs(self) -> Path:
        return self.path / "inputs"

    @property
    def output(self) -> Path:
        return self.path / "output"

    @property
    def command(self) -> Path:
        return self.path / "command.sh"

    @property
    def info(self) -> Path:
        return self.path / "backend.json"


def write_package(out_dir: str | Path, *, backend: str, role: str,
                  command: str, inputs: dict, meta: dict | None = None) -> Package:
    """Materialize a condition package.

    inputs: {filename: Path (copied) | np.ndarray (saved) | str (written text)}
            — keys ARE the target filenames inside inputs/ (e.g. "ref.png"),
            so GPU drivers can rely on stable names.
    meta:   extra provenance merged into backend.json.
    """
    out = Path(out_dir)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "output").mkdir(exist_ok=True)
    written = {}
    for name, value in inputs.items():
        if isinstance(value, np.ndarray):
            fname = name if name.endswith(".npy") else f"{name}.npy"
            p = out / "inputs" / fname
            np.save(p, value)
            written[fname] = str(p.relative_to(out))
        elif isinstance(value, Path):
            if not value.exists():
                raise FileNotFoundError(f"input {name} missing: {value}")
            dst = out / "inputs" / name
            if value.resolve() != dst.resolve():
                dst.write_bytes(value.read_bytes())
            written[name] = str(dst.relative_to(out))
        elif isinstance(value, str):
            (out / "inputs" / name).write_text(value)
            written[name] = f"inputs/{name}"
        else:
            raise TypeError(f"input {name}: unsupported type {type(value)}")
    (out / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                                    f"{command}\n")
    (out / "command.sh").chmod(0o755)
    info = {
        "protocol": PROTOCOL,
        "backend": backend,
        "role": role,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "inputs": written,
        **(meta or {}),
    }
    (out / "backend.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return Package(path=out, backend=backend, role=role, meta=info)


def read_package(package: str | Path) -> Package:
    path = Path(package)
    info = json.loads((path / "backend.json").read_text())
    return Package(path=path, backend=info["backend"], role=info["role"], meta=info)


def assert_gauss_trace(meta: dict, *, require_gauss_frame: bool = True) -> None:
    """Fail fast when a frame-derived condition did not come from the Gaussian renderer."""
    sources = [str(s) for s in meta.get("condition_sources", [])]
    if require_gauss_frame:
        if not any("/gauss/" in s for s in sources):
            raise RuntimeError(
                "diffusion condition is not traced to Gaussian frames "
                f"(condition_sources={sources}); regenerate the batch with "
                "scripts/run_inject_gauss.py")
        if not meta.get("renderer", "").startswith("realinject.gauss"):
            raise RuntimeError(
                f"condition renderer {meta.get('renderer')!r} is not the "
                "Gaussian renderer")


def placeholder_run(backend: str) -> dict:
    """CPU stub result — the real inference runs on the GPU server."""
    return {"status": "placeholder", "backend": backend,
            "note": "run command.sh on the GPU server, then collect()"}
