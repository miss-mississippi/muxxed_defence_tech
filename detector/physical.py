"""Checking the predicted type against the measured size.

The scene is georeferenced and the pixel size is known, so object dimensions are measured in metres.
An aircraft of the claimed type cannot be half its reference size; the check is built on that.
This is how a "C-5 Galaxy" measuring 38.8 x 35.6 m against the reference 75.5 x 67.9 m was caught.

Matching dimensions do not prove the type: a Boeing 767 (48.5 x 47.6 m) is indistinguishable by size
from an E-8 built on the Boeing 707 airframe (46.6 x 44.4 m). Only gross errors are caught, hence
three verdicts: size matches, borderline, type not confirmed.

Reference values come from public sources: fuselage length and wingspan.
"""
from __future__ import annotations

from dataclasses import dataclass

# fuselage length and wingspan in metres; span_swept is the span with wings fully swept
# for variable-sweep aircraft (Tu-160, B-1B, Tu-22M, Su-24)
REFERENCE: dict[str, tuple[float, float, float | None]] = {
    "SU-35":  (21.9, 15.3, None),
    "SU-34":  (23.3, 14.7, None),
    "SU-24":  (22.7, 17.6, 10.4),
    "F-16":   (15.1, 10.0, None),
    "F-15":   (19.4, 13.1, None),
    "F-22":   (18.9, 13.6, None),
    "FA-18":  (17.1, 12.3, None),
    "C-130":  (29.8, 40.4, None),
    "C-17":   (53.0, 51.8, None),
    "C-5":    (75.5, 67.9, None),
    "E-3":    (46.6, 44.4, None),
    "E-8":    (46.6, 44.4, None),
    "P-3C":   (35.6, 30.4, None),
    "B-52":   (48.5, 56.4, None),
    "B-1B":   (44.5, 41.8, 24.1),
    "TU-160": (54.1, 55.7, 35.6),
    "TU-95":  (46.2, 50.1, None),
    "TU-22":  (42.5, 34.3, 23.3),
    "KC-135": (41.5, 39.9, None),
    "KC-10":  (55.4, 50.4, None),
}

OK = "ok"                    # dimensions match the reference
BORDERLINE = "borderline"    # noticeable deviation, still explainable by imaging conditions
MISMATCH = "mismatch"        # dimensions do not match: the type is not confirmed by size
UNKNOWN = "unknown"          # no reference for the class, or no georeferencing


@dataclass(frozen=True)
class SizeCheck:
    verdict: str
    deviation: float | None = None       # largest relative deviation, 0.48 = 48%
    expected_long: tuple[float, float] | None = None
    expected_short: tuple[float, float] | None = None

    @property
    def confirmed(self) -> bool:
        return self.verdict == OK

    def describe(self) -> str:
        if self.verdict == UNKNOWN:
            return "size not checked"
        assert self.expected_long and self.expected_short
        span = f"{self.expected_long[0]:.0f}–{self.expected_long[1]:.0f}" \
            if self.expected_long[0] != self.expected_long[1] else f"{self.expected_long[0]:.0f}"
        if self.verdict == OK:
            return f"size matches the type (expected ~{span} m along the long axis)"
        if self.verdict == BORDERLINE:
            return (f"borderline: {self.deviation:.0%} off the reference "
                    f"(expected ~{span} m along the long axis)")
        return (f"type not confirmed by size: {self.deviation:.0%} off "
                f"the reference (expected ~{span} m along the long axis)")


def _bounds(length: float, span: float, span_swept: float | None) -> tuple[tuple[float, float], tuple[float, float]]:
    """Allowed long- and short-axis ranges across all wing positions."""
    configs = [(length, span)] + ([(length, span_swept)] if span_swept else [])
    longs = [max(a, b) for a, b in configs]
    shorts = [min(a, b) for a, b in configs]
    return (min(longs), max(longs)), (min(shorts), max(shorts))


def _deviation(value: float, lo: float, hi: float) -> float:
    if lo <= value <= hi:
        return 0.0
    reference = lo if value < lo else hi
    return abs(value - reference) / reference


def check_size(
    class_name: str,
    length_m: float | None,
    width_m: float | None,
    tolerance: float = 0.25,
    borderline: float = 0.15,
) -> SizeCheck:
    """Compares the measured dimensions with the reference dimensions for the type.

    Three grades instead of a single threshold: a binary answer right at the boundary misleads
    in both directions. Up to `borderline` (15%) the deviation is explained by box error, shadows
    and georeferencing accuracy; between `borderline` and `tolerance` (25%) it is noticeable but
    not by itself a type error; beyond that the type is considered not confirmed.
    """
    reference = REFERENCE.get(class_name)
    if reference is None or length_m is None or width_m is None:
        return SizeCheck(UNKNOWN)
    long_range, short_range = _bounds(*reference)
    measured_long, measured_short = max(length_m, width_m), min(length_m, width_m)
    deviation = max(_deviation(measured_long, *long_range), _deviation(measured_short, *short_range))
    if deviation <= borderline:
        verdict = OK
    elif deviation <= tolerance:
        verdict = BORDERLINE
    else:
        verdict = MISMATCH
    return SizeCheck(verdict, round(deviation, 3), long_range, short_range)


def annotate(properties: dict, tolerance: float = 0.25, borderline: float = 0.15) -> dict:
    """Adds the size-check result to the detection properties (in place)."""
    check = check_size(
        properties.get("class_name", ""), properties.get("length_m"), properties.get("width_m"),
        tolerance, borderline,
    )
    properties["size_check"] = check.verdict
    if check.verdict != UNKNOWN:
        properties["size_deviation"] = check.deviation
    return properties
