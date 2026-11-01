"""Tests for registry reconciliation (detector/inventory.py).

Dimensions come from real detections on the demo airfield, so the size-check verdicts
here are the same as on the map.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from detector.inventory import (
    EXTRA, MATCH, MISSING, NOT_IN_REGISTRY, UNRECOGNIZABLE, load_registry, parse_registry, reconcile,
)

AREA = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
C17_OK = (53.0, 51.8)       # matches the reference
C17_BORDER = (41.5, 39.0)   # 24.7%: borderline
C17_MISS = (28.2, 26.4)     # 49.0%: type not confirmed
C5_MISS = (38.8, 35.6)      # 48.6%: type not confirmed
E8 = (47.3, 41.4)           # 6.8%
KC135 = (41.5, 39.9)        # reference


def plane(cls, size=(None, None), lon=0.5, lat=0.5, **extra):
    p = {"class_name": cls, "confidence": 0.8, "center_lon": lon, "center_lat": lat,
         "length_m": size[0], "width_m": size[1]}
    p.update(extra)
    return {"properties": p}


def registry(**expected):
    return parse_registry({"airfields": [{"id": "a", "name": "A", "area": AREA, "expected": expected}]})


def lines(report):
    return {line.type: line for line in report.airfields[0].lines}


def test_statuses():
    feats = [plane("C-17", C17_OK), plane("C-17", C17_OK), plane("E-8", E8), plane("E-8", E8),
             plane("KC-135", KC135)]
    got = lines(reconcile(feats, registry(**{"C-17": 2, "E-8": 1, "C-130": 1})))
    assert got["C-17"].status == MATCH
    assert got["E-8"].status == EXTRA
    assert got["C-130"].status == MISSING
    assert got["KC-135"].status == NOT_IN_REGISTRY


def test_size_check_prevents_false_alarm():
    # the model confidently called something a C-5 that cannot be a C-5 by size
    feats = [plane("C-5", C5_MISS)]
    with_check = reconcile(feats, registry(**{"C-17": 1}))
    assert "C-5" not in lines(with_check)
    assert [u["class"] for u in with_check.airfields[0].unconfirmed] == ["C-5"]

    without_check = reconcile(feats, registry(**{"C-17": 1}), trust_size_check=False)
    assert lines(without_check)["C-5"].status == NOT_IN_REGISTRY  # false alarm


def test_mismatch_not_counted_toward_its_type():
    feats = [plane("C-17", C17_OK), plane("C-17", C17_MISS)]
    assert lines(reconcile(feats, registry(**{"C-17": 2})))["C-17"].status == MISSING
    assert lines(reconcile(feats, registry(**{"C-17": 2}), trust_size_check=False))["C-17"].status == MATCH


def test_borderline_counted_and_flagged():
    line = lines(reconcile([plane("C-17", C17_BORDER)], registry(**{"C-17": 1})))["C-17"]
    assert line.status == MATCH and line.observed == 1 and line.borderline == 1


def test_unknown_type_is_not_reported_as_missing():
    line = lines(reconcile([], registry(**{"SU-27": 2})))["SU-27"]
    assert line.status == UNRECOGNIZABLE


def test_untyped_plane_and_missing_note():
    report = reconcile([plane("plane")], registry(**{"C-130": 1}))
    assert len(report.airfields[0].untyped) == 1
    assert lines(report)["C-130"].status == MISSING
    assert "may include the ones not seen" in report.to_text()


def test_aircraft_outside_airfields():
    report = reconcile([plane("C-17", C17_OK, lon=5, lat=5)], registry(**{"C-17": 1}))
    assert len(report.outside) == 1 and lines(report)["C-17"].status == MISSING


def test_airfield_outside_scene_is_not_reported_missing():
    report = reconcile([], registry(**{"C-17": 2}), scene_bounds=[10, 10, 11, 11])
    assert report.airfields[0].coverage == "none"
    assert report.airfields[0].lines == [] and report.airfields[0].missing == 0
    assert "outside the image" in report.to_text()


def test_partial_coverage_warns():
    report = reconcile([], registry(**{"C-17": 1}), scene_bounds=[0.5, 0.5, 2, 2])
    assert report.airfields[0].coverage == "partial"
    assert any("partially" in w for w in report.warnings)
    full = reconcile([], registry(**{"C-17": 1}), scene_bounds=[-1, -1, 2, 2])
    assert full.airfields[0].coverage == "full" and not full.warnings


def test_resolution_warning():
    assert reconcile([], registry(**{"C-17": 1}), gsd_m=10.0).warnings
    assert not reconcile([], registry(**{"C-17": 1}), gsd_m=0.3).warnings


def test_missing_coordinates_warning():
    feat = plane("C-17", C17_OK, lon=None, lat=None)
    report = reconcile([feat], registry(**{"C-17": 1}))
    assert any("without coordinates" in w for w in report.warnings)


def test_operator_review_is_respected():
    feats = [plane("C-5", C5_MISS, review_class="plane"), plane("C-17", C17_OK, review_status="rejected")]
    report = reconcile(feats, registry(**{"C-17": 1}))
    assert len(report.airfields[0].untyped) == 1 and lines(report)["C-17"].observed == 0

    raw = reconcile(feats, registry(**{"C-17": 1}), use_review=False)
    assert lines(raw)["C-17"].observed == 1


def test_size_check_follows_corrected_class():
    # the model said C-5, the operator corrected it to C-17: these dimensions are plausible for a C-17
    feat = plane("C-5", C17_OK, review_class="C-17")
    assert lines(reconcile([feat], registry(**{"C-17": 1})))["C-17"].status == MATCH


def test_non_aircraft_ignored():
    feats = [plane("ship"), plane("storage tank")]
    report = reconcile(feats, registry(**{"C-17": 1}))
    assert not report.airfields[0].untyped and not report.outside


@pytest.mark.parametrize("bad", [
    {}, {"airfields": []}, {"airfields": [{"expected": {"C-17": 1}}]},
    {"airfields": [{"area": AREA, "expected": {"C-17": -1}}]},
])
def test_registry_validation(bad):
    with pytest.raises(ValueError):
        parse_registry(bad)


def test_demo_registry_parses():
    airfields = load_registry(Path(__file__).resolve().parents[1] / "samples" / "registry_demo.json")
    assert airfields[0].expected["C-17"] == 2
