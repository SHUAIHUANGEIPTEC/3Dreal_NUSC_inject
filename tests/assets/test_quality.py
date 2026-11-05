"""Quality gate logic."""
from __future__ import annotations

from realinject.assets import quality


def _manifest(length=4.9, w=1.9, h=1.3, grade="A"):
    return {"L": length, "W": w, "H": h, "grade": grade}


def test_realcar_pass():
    ok, reasons = quality.gate_realcar((4.9, 1.9, 1.3), 30.0, _manifest())
    assert ok and not reasons


def test_realcar_grade_c_rejected():
    ok, reasons = quality.gate_realcar((4.9, 1.9, 1.3), 30.0, _manifest(grade="C"))
    assert not ok and any(r.startswith("grade:") for r in reasons)


def test_realcar_dims_out_of_range():
    ok, reasons = quality.gate_realcar((7.5, 1.9, 1.3), 30.0, _manifest(length=7.5))
    assert not ok
    assert any(r.startswith("l_out_of_range") for r in reasons)


def test_realcar_bbox_disagreement_flagged():
    ok, reasons = quality.gate_realcar((4.9, 1.9, 1.3), 30.0, _manifest(length=6.5))
    assert not ok
    assert any(r.startswith("bbox_disagrees_l") for r in reasons)


def test_realcar_flat_texture_rejected():
    ok, reasons = quality.gate_realcar((4.9, 1.9, 1.3), 1.0, _manifest())
    assert not ok
    assert any(r.startswith("texture_flat") for r in reasons)


def test_av2_pass_and_fail():
    good = {"n": 4479, "az_span_deg": 347, "pts_per_m2": 1409, "coverage": 0.174}
    ok, reasons = quality.gate_av2(good)
    assert ok and not reasons

    bad = {"n": 19, "az_span_deg": 4, "pts_per_m2": 10, "coverage": 0.009}
    ok, reasons = quality.gate_av2(bad)
    assert not ok
    assert {"too_few_points:19", "az_span_low:4"}.issubset(set(reasons))


def test_completion_candidate_only_wide_dogs():
    assert quality.completion_candidate({"cat": "DOG", "az_span_deg": 347})
    assert not quality.completion_candidate({"cat": "DOG", "az_span_deg": 134})
    assert not quality.completion_candidate({"cat": "STROLLER", "az_span_deg": 300})


def test_gate_override_from_cfg():
    gates = {"realcar": {"l_range": [3.0, 8.0]}}
    ok, _ = quality.gate_realcar((7.5, 1.9, 1.3), 30.0, _manifest(length=7.5), gates)
    assert ok
