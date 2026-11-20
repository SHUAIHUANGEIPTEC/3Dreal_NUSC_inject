"""run_completion eligibility: av2 az-candidates vs realcar pass-gate."""
from __future__ import annotations

from realinject.assets import complete_runner
from realinject.assets.manifest import AssetRecord


def _rec(asset_id: str, kind: str, passed: bool, candidate: bool = False) -> AssetRecord:
    return AssetRecord(
        asset_id=asset_id, kind=kind, source_path="src", canonical_path="can.npz",
        category="vehicle.car", dims_lwh_m=(4.0, 2.0, 1.5),
        quality={"passed": passed},
        completion={"candidate": candidate},
    )


def test_run_completion_eligibility(monkeypatch, tmp_path):
    calls: dict[str, list[str]] = {"realcar": [], "av2": []}

    def fake_realcar(rec, root):
        calls["realcar"].append(rec.asset_id)
        return {"status": "cpu_poisson"}

    def fake_av2(rec, root, n_samples: int = 20000):
        calls["av2"].append(rec.asset_id)
        return {"status": "cpu_proxy"}

    monkeypatch.setattr(complete_runner, "complete_realcar", fake_realcar)
    monkeypatch.setattr(complete_runner, "complete_av2", fake_av2)

    records = [
        _rec("realcar/ok", "realcar_mesh", True),
        _rec("realcar/rejected", "realcar_mesh", False),
        _rec("av2/weak_wide", "av2_cloud", False, candidate=True),
        _rec("av2/narrow", "av2_cloud", True, candidate=False),
    ]
    report = complete_runner.run_completion(records, tmp_path)

    # passed mesh completes; gate-rejected mesh is skipped (fill won't help)
    assert calls["realcar"] == ["realcar/ok"]
    # az-wide candidate completes even though the original gates fail —
    # the proxy's coverage/density stats are what the rebuild re-checks
    assert calls["av2"] == ["av2/weak_wide"]
    assert report["realcar_completed"] == 1
    assert report["av2_completed"] == 1
    assert report["skipped"] == 2
    assert report["failed"] == 0
