"""Расчёт полноты по сцене (api/recall.py)."""
from __future__ import annotations

from api.recall import scene_recall


def det(cls, conf, status="confirmed", review_class=None):
    return {"class_name": cls, "confidence": conf, "review_status": status, "review_class": review_class}


def test_recall_at_two_thresholds():
    dets = [det("ship", 0.9), det("ship", 0.5), det("ship", 0.3)]
    r = scene_recall(dets, [{"class_name": "ship"}], complete=True)
    assert r["total"]["exist"] == 4 and r["total"]["missed"] == 1
    assert r["total"]["recall"] == {"0.25": 0.75, "0.4": 0.5}
    assert not r["warnings"]


def test_rejected_are_not_objects_and_pending_is_warned():
    dets = [det("ship", 0.9), det("ship", 0.9, status="rejected"), det("ship", 0.9, status="pending")]
    r = scene_recall(dets, [], complete=True)
    assert r["total"]["exist"] == 1
    assert any("не проверены" in w for w in r["warnings"])


def test_by_class_uses_operator_class():
    dets = [det("C-17", 0.8, review_class="plane"), det("small vehicle", 0.3)]
    r = scene_recall(dets, [{"class_name": "small vehicle"}], complete=True)
    assert set(r["by_class"]) == {"plane", "small vehicle"}
    sv = r["by_class"]["small vehicle"]
    # мелкая техника с уверенностью 0.3 теряется при пороге карты 0.4
    assert sv["recall"] == {"0.25": 0.5, "0.4": 0.0}


def test_incomplete_marking_is_only_an_upper_bound():
    r = scene_recall([det("ship", 0.9)], [], complete=False)
    assert r["total"]["recall"]["0.25"] == 1.0
    assert any("верхняя оценка" in w for w in r["warnings"])


def test_empty_scene():
    r = scene_recall([], [], complete=True)
    assert r["total"]["exist"] == 0 and r["total"]["recall"]["0.25"] is None
