"""Тесты сверки габаритов (detector/physical.py).

Числа взяты из реальных детекций демо-сцены 7, аэродром King Shaka.
"""
from __future__ import annotations

import pytest

from detector.physical import BORDERLINE, MISMATCH, OK, REFERENCE, UNKNOWN, annotate, check_size


def test_c5_twice_too_small_is_flagged():
    # C-5 Galaxy по паспорту 75.5 x 67.9 м, измерено 38.8 x 35.6 — промах вдвое
    check = check_size("C-5", 38.8, 35.6)
    assert check.verdict == MISMATCH
    assert check.deviation == pytest.approx(0.486, abs=0.005)
    assert not check.confirmed


def test_c17_half_size_is_flagged():
    # C-17 по паспорту 53.0 x 51.8 м, измерено 28.2 x 26.4
    assert check_size("C-17", 28.2, 26.4).verdict == MISMATCH


def test_e8_matching_size_passes():
    # E-8 (на базе Boeing 707) 46.6 x 44.4 м, измерено 47.3 x 41.4 — расхождение в пределах допуска
    check = check_size("E-8", 47.3, 41.4)
    assert check.verdict == OK
    assert check.confirmed


def test_borderline_is_its_own_verdict():
    # C-17 41.5 x 39.0 м: расхождение 24.7% — ни совпадение, ни промах
    check = check_size("C-17", 41.5, 39.0)
    assert check.verdict == BORDERLINE
    assert not check.confirmed
    assert check.deviation == pytest.approx(0.247, abs=0.005)


def test_verdict_boundaries_are_configurable():
    assert check_size("C-17", 41.5, 39.0, borderline=0.30).verdict == OK
    assert check_size("C-17", 41.5, 39.0, tolerance=0.20).verdict == MISMATCH


def test_swing_wing_accepts_both_positions():
    # Су-24: размах 17.6 м в развёрнутом положении и 10.4 м в сложенном — верны оба
    assert check_size("SU-24", 22.7, 17.6).verdict == OK
    assert check_size("SU-24", 22.7, 10.4).verdict == OK
    assert check_size("SU-24", 22.7, 4.0).verdict == MISMATCH


def test_long_axis_compared_regardless_of_order():
    """Длинная ось сверяется с длинной независимо от того, что пришло первым."""
    assert check_size("C-130", 40.4, 29.8).verdict == OK
    assert check_size("C-130", 29.8, 40.4).verdict == OK


def test_unknown_without_reference_or_size():
    assert check_size("ship", 30.0, 10.0).verdict == UNKNOWN          # нет эталона для класса
    assert check_size("C-17", None, None).verdict == UNKNOWN          # нет геопривязки
    assert check_size("C-17", 53.0, None).verdict == UNKNOWN


def test_all_reference_types_are_self_consistent():
    """Каждый эталон должен проходить собственную проверку."""
    for name, (length, span, _) in REFERENCE.items():
        assert check_size(name, max(length, span), min(length, span)).verdict == OK, name


def test_annotate_writes_properties():
    props = {"class_name": "C-5", "length_m": 38.8, "width_m": 35.6}
    annotate(props)
    assert props["size_check"] == MISMATCH
    assert props["size_deviation"] == pytest.approx(0.486, abs=0.005)

    clean = {"class_name": "ship", "length_m": 30.0, "width_m": 10.0}
    annotate(clean)
    assert clean["size_check"] == UNKNOWN
    assert "size_deviation" not in clean


def test_describe_explains_the_verdict():
    assert "не подтверждён" in check_size("C-5", 38.8, 35.6).describe()
    assert "соответствуют" in check_size("E-8", 47.3, 41.4).describe()
    assert "не проверялся" in check_size("ship", 30.0, 10.0).describe()
