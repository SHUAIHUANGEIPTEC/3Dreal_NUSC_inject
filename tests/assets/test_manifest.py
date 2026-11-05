"""Manifest round-trip and selection semantics."""
from __future__ import annotations

from realinject.assets.manifest import AssetLibrary, AssetRecord


def _rec(asset_id="realcar/a", kind="realcar_mesh", category="vehicle.car", passed=True):
    return AssetRecord(
        asset_id=asset_id, kind=kind, source_path="realcar_assets/a",
        canonical_path="canonical/realcar/a.npz", category=category,
        dims_lwh_m=(4.9, 1.9, 1.3), grade="A",
        quality={"passed": passed, "reasons": [] if passed else ["grade:C"]},
    )


def test_roundtrip(tmp_path):
    lib = AssetLibrary([_rec(), _rec("av2/DOG_1", "av2_cloud",
                                      "human.pedestrian.animal")],
                       gates={"av2": {"min_points": 300}},
                       assets_root="/a", library_root="/l")
    path = lib.save(tmp_path / "m.json", generated_at="2026-01-01T00:00:00")
    loaded = AssetLibrary.load(path)
    assert len(loaded) == 2
    assert loaded.records[0].dims_lwh_m == (4.9, 1.9, 1.3)
    assert loaded.gates["av2"]["min_points"] == 300
    assert loaded.records[1].category == "human.pedestrian.animal"


def test_select_filters_passed_only():
    lib = AssetLibrary([_rec(), _rec("realcar/b", passed=False),
                        _rec("av2/DOG_1", "av2_cloud", "human.pedestrian.animal")])
    assert {r.asset_id for r in lib.select()} == {"realcar/a", "av2/DOG_1"}
    assert [r.asset_id for r in lib.select(kind="av2_cloud")] == ["av2/DOG_1"]
    assert [r.asset_id for r in lib.select(category="vehicle.car")] == ["realcar/a"]
    assert [r.asset_id for r in lib.select(asset_ids=["realcar/a"])] == ["realcar/a"]


def test_counts_and_gates_report():
    lib = AssetLibrary([_rec(), _rec("realcar/b", passed=False)])
    counts = lib.counts()
    assert counts["total"] == 2 and counts["passed"] == 1 and counts["rejected"] == 1
    report = lib.gates_report()
    assert report["counts"]["passed"] == 1
    assert report["reason_histogram"] == {"grade": 1}
