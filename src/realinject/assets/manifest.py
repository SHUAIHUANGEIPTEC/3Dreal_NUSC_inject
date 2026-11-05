"""Asset library manifest: one record per injectable asset.

``assets_manifest.json`` (written under ``library.path``) is the single index
consumed by the placement/injection stages. Schema version ``3dreal-assets/1``:

    {schema, generated_at, assets_root, gates, counts, assets: [AssetRecord...]}

Every path inside a record is relative to ``assets.root`` (source) or
``library.path`` (canonical npz) so the manifest survives machine moves.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA = "3dreal-assets/1"


@dataclass
class AssetRecord:
    """One normalized, injectable asset."""

    asset_id: str  # "realcar/2024_06_04_13_44_39" | "av2/DOG_03ce7747"
    kind: str  # "realcar_mesh" | "av2_cloud"
    source_path: str  # relative to assets.root
    canonical_path: str  # relative to library.path
    category: str  # nuScenes category string for the annotation export
    dims_lwh_m: tuple[float, float, float]
    frame: str = "blender_zup"
    raw_bbox_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    raw_bbox_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    grade: str | None = None
    quality: dict = field(default_factory=dict)
    completion: dict = field(default_factory=dict)
    gaussian: dict = field(default_factory=dict)
    conditioning: dict = field(default_factory=dict)
    tracks: list = field(default_factory=list)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["dims_lwh_m"] = list(self.dims_lwh_m)
        out["raw_bbox_min"] = list(self.raw_bbox_min)
        out["raw_bbox_size"] = list(self.raw_bbox_size)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> AssetRecord:
        known = {f for f in cls.__dataclass_fields__}
        out = {k: v for k, v in d.items() if k in known}
        for key in ("dims_lwh_m", "raw_bbox_min", "raw_bbox_size"):
            if key in out and out[key] is not None:
                out[key] = tuple(float(v) for v in out[key])
        return cls(**out)


class AssetLibrary:
    """In-memory view over the manifest with gate-aware selection."""

    def __init__(self, records: list[AssetRecord], gates: dict | None = None,
                 assets_root: str = "", library_root: str = ""):
        self.records = list(records)
        self.gates = dict(gates or {})
        self.assets_root = assets_root
        self.library_root = library_root

    def __len__(self) -> int:
        return len(self.records)

    def passed(self) -> list[AssetRecord]:
        return [r for r in self.records if r.quality.get("passed", False)]

    def by_category(self) -> dict[str, list[AssetRecord]]:
        out: dict[str, list[AssetRecord]] = {}
        for r in self.passed():
            out.setdefault(r.category, []).append(r)
        return out

    def by_kind(self) -> dict[str, list[AssetRecord]]:
        out: dict[str, list[AssetRecord]] = {}
        for r in self.passed():
            out.setdefault(r.kind, []).append(r)
        return out

    def select(self, kind: str | None = None, category: str | None = None,
               asset_ids: list[str] | None = None) -> list[AssetRecord]:
        out = self.passed()
        if kind:
            out = [r for r in out if r.kind == kind]
        if category:
            out = [r for r in out if r.category == category]
        if asset_ids:
            wanted = set(asset_ids)
            out = [r for r in out if r.asset_id in wanted]
        return out

    def counts(self) -> dict:
        passed = self.passed()
        return {
            "total": len(self.records),
            "passed": len(passed),
            "rejected": len(self.records) - len(passed),
            "by_kind": _count_by(passed, "kind"),
            "by_category": _count_by(passed, "category"),
        }

    def to_dict(self, generated_at: str | None = None) -> dict:
        return {
            "schema": SCHEMA,
            "generated_at": generated_at,
            "assets_root": self.assets_root,
            "library_root": self.library_root,
            "gates": self.gates,
            "counts": self.counts(),
            "assets": [r.to_dict() for r in self.records],
        }

    def save(self, path: str | Path, generated_at: str | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(generated_at), indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> AssetLibrary:
        data = json.loads(Path(path).read_text())
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported manifest schema: {data.get('schema')!r}")
        return cls(
            records=[AssetRecord.from_dict(d) for d in data["assets"]],
            gates=data.get("gates"),
            assets_root=data.get("assets_root", ""),
            library_root=data.get("library_root", ""),
        )

    def gates_report(self) -> dict:
        """Per-asset gate outcome plus rejection reason histogram."""
        reasons: dict[str, int] = {}
        assets = []
        for r in self.records:
            reasons_list = list(r.quality.get("reasons", []))
            for reason in reasons_list:
                key = reason.split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
            assets.append({
                "asset_id": r.asset_id,
                "kind": r.kind,
                "category": r.category,
                "grade": r.grade,
                "passed": bool(r.quality.get("passed", False)),
                "reasons": reasons_list,
                "dims_lwh_m": list(r.dims_lwh_m),
            })
        return {"counts": self.counts(), "reason_histogram": reasons, "assets": assets}


def _count_by(records: list[AssetRecord], attr: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        key = getattr(r, attr)
        out[key] = out.get(key, 0) + 1
    return out
