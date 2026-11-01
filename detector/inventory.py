"""Reconciling detected aircraft against an equipment registry.

The registry is a list of airfields: a WGS84 polygon and how many aircraft of each type are
listed there. A detection is assigned to an airfield by its centre point, then "listed" and
"detected" counts are compared type by type. The registry is customer data: it comes in as input
and is never stored. The repository only contains a mock example, samples/registry_demo.json.

What keeps the reconciliation honest:
- a type is counted only if the measured size does not contradict it; otherwise the detection goes
  to "type not confirmed" and does not raise a false alarm about an unlisted aircraft;
- "not seen in the image" does not mean "absent": the aircraft may be in a hangar or in the air;
- a type the model does not know cannot be reconciled, so it gets its own status;
- an airfield outside the image means "cannot reconcile", not "every aircraft is gone";
- aircraft types do not survive resolution coarser than ~1 m, and the report warns about it.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import Point, box, shape
from shapely.geometry.base import BaseGeometry

from .physical import BORDERLINE, MISMATCH, REFERENCE, check_size

# degrading the demo airfield: at 1 m the type was found for 3 of 4 aircraft, at 2 m for only one
MAX_TYPE_GSD_M = 1.0
UNTYPED = frozenset({"plane"})  # the general model finds an aircraft but not its type

MATCH = "match"
MISSING = "missing"
EXTRA = "extra"
NOT_IN_REGISTRY = "not_in_registry"
UNRECOGNIZABLE = "unrecognizable"

STATUS_TEXT = {
    MATCH: "matches",
    MISSING: "not seen in the image",
    EXTRA: "more than listed",
    NOT_IN_REGISTRY: "type not listed at this airfield",
    UNRECOGNIZABLE: "type not recognised by the model",
}


@dataclass
class Airfield:
    id: str
    name: str
    area: BaseGeometry
    expected: dict[str, int]


def parse_registry(data: dict) -> list[Airfield]:
    if not isinstance(data, dict) or not isinstance(data.get("airfields"), list) or not data["airfields"]:
        raise ValueError("registry: expected an object with a non-empty airfields list")
    airfields = []
    for i, a in enumerate(data["airfields"]):
        try:
            area = shape(a["area"])
            expected = {str(t): int(n) for t, n in a["expected"].items()}
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"registry: airfield #{i} needs area (GeoJSON) and expected {{type: count}}") from exc
        if area.is_empty or not area.is_valid:
            raise ValueError(f"registry: airfield #{i} has an empty or invalid polygon")
        if any(n < 0 for n in expected.values()):
            raise ValueError(f"registry: airfield #{i} has a negative aircraft count")
        airfields.append(Airfield(str(a.get("id", i)), str(a.get("name", a.get("id", i))), area, expected))
    return airfields


def load_registry(path: str | Path) -> list[Airfield]:
    return parse_registry(json.loads(Path(path).read_text()))


@dataclass
class TypeLine:
    type: str
    expected: int
    observed: int
    borderline: int
    status: str


@dataclass
class AirfieldReport:
    id: str
    name: str
    lines: list[TypeLine]
    unconfirmed: list[dict] = field(default_factory=list)  # type named, but the size contradicts it
    untyped: list[dict] = field(default_factory=list)      # aircraft without a type
    coverage: str = "unknown"                               # full | partial | none | unknown

    @property
    def missing(self) -> int:
        if self.coverage == "none":
            return 0
        return sum(line.expected - line.observed for line in self.lines if line.status == MISSING)


@dataclass
class InventoryReport:
    airfields: list[AirfieldReport]
    outside: list[dict]
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "warnings": self.warnings,
            "airfields": [
                {
                    "id": a.id,
                    "name": a.name,
                    "types": [
                        {"type": line.type, "expected": line.expected, "observed": line.observed,
                         "borderline": line.borderline, "status": line.status,
                         "status_text": STATUS_TEXT[line.status]}
                        for line in a.lines
                    ],
                    "coverage": a.coverage,
                    "unconfirmed": a.unconfirmed,
                    "untyped": a.untyped,
                }
                for a in self.airfields
            ],
            "outside": self.outside,
        }

    def to_text(self) -> str:
        out = [f"! {w}" for w in self.warnings]
        for a in self.airfields:
            out.append(f'Airfield "{a.name}"')
            if a.coverage == "none":
                out.append("  airfield is outside the image: cannot reconcile")
                continue
            for line in a.lines:
                seen = str(line.observed)
                if line.borderline:
                    seen += f" (borderline: {line.borderline})"
                verdict = STATUS_TEXT[line.status]
                if line.status in (MISSING, EXTRA):
                    verdict += f": {abs(line.observed - line.expected)}"
                out.append(f"  {line.type:<7} listed {line.expected:>2}   detected {seen:<24} {verdict}")
            if a.unconfirmed:
                items = ", ".join(
                    f"{u['class']} ({u['size_deviation']:.1%})" if u.get("size_deviation") is not None else u["class"]
                    for u in a.unconfirmed
                )
                out.append(f"  type named by the model but contradicted by size: {len(a.unconfirmed)}: {items}")
            if a.untyped:
                out.append(f"  aircraft without a determined type: {len(a.untyped)}")
            unresolved = len(a.unconfirmed) + len(a.untyped)
            if a.missing and unresolved:
                out.append(f"  the {unresolved} aircraft without a confirmed type may include "
                           f"the ones not seen ({a.missing})")
        if self.outside:
            out.append(f"Aircraft outside registry airfields: {len(self.outside)}")
        return "\n".join(out)


def _size_verdict(p: dict, cls: str) -> tuple[str | None, float | None]:
    # if the operator corrected the class, the stored verdict was computed for the old one, so recompute
    length, width = p.get("length_m"), p.get("width_m")
    if length is not None and width is not None:
        check = check_size(cls, length, width)
        return check.verdict, check.deviation
    return p.get("size_check"), p.get("size_deviation")


def reconcile(
    features: list[dict],
    airfields: list[Airfield],
    gsd_m: float | None = None,
    scene_bounds: list[float] | None = None,
    known_types: set[str] | None = None,
    use_review: bool = True,
    trust_size_check: bool = True,
    max_type_gsd: float = MAX_TYPE_GSD_M,
) -> InventoryReport:
    """features -- GeoJSON features or their properties, from detect_scene or from the catalogue.

    scene_bounds -- image footprint [west, south, east, north] in WGS84; without it, whether
    an airfield is covered by the image is not checked.
    known_types -- the types the model can name; defaults to the 20 MAR20 types.
    use_review -- respect operator decisions: corrected classes and rejected objects.
    trust_size_check=False is only for comparison: without the size check, wrong types are
    counted and produce false discrepancies.
    """
    known = set(REFERENCE) if known_types is None else set(known_types)
    warnings = []
    if gsd_m is not None and gsd_m > max_type_gsd:
        warnings.append(f"resolution {gsd_m:.2f} m/px is coarser than {max_type_gsd:g} m: aircraft types are "
                        f"unreliable at this resolution, so the per-type reconciliation is indicative only")

    buckets = {a.id: {"observed": Counter(), "borderline": Counter(), "unconfirmed": [], "untyped": []}
               for a in airfields}
    outside, no_coords = [], 0
    for f in features:
        p = f.get("properties", f)
        if use_review and p.get("review_status") == "rejected":
            continue
        cls = (p.get("review_class") if use_review else None) or p["class_name"]
        if cls not in UNTYPED and cls not in known:
            continue  # not an aircraft
        lon, lat = p.get("center_lon"), p.get("center_lat")
        if lon is None or lat is None:
            no_coords += 1
            continue
        verdict, deviation = (None, None) if cls in UNTYPED else _size_verdict(p, cls)
        item = {"id": p.get("id"), "class": cls, "confidence": p.get("confidence"),
                "size_check": verdict, "size_deviation": deviation}
        point = Point(lon, lat)
        home = next((a for a in airfields if a.area.covers(point)), None)
        if home is None:
            outside.append(item)
            continue
        b = buckets[home.id]
        if cls in UNTYPED:
            b["untyped"].append(item)
        elif trust_size_check and verdict == MISMATCH:
            b["unconfirmed"].append(item)
        else:
            b["observed"][cls] += 1
            if verdict == BORDERLINE:
                b["borderline"][cls] += 1
    if no_coords:
        warnings.append(f"{no_coords} aircraft without coordinates (image not georeferenced) were not assigned to airfields")

    scene_area = box(*scene_bounds) if scene_bounds else None
    reports = []
    for a in airfields:
        b = buckets[a.id]
        if scene_area is None:
            coverage = "unknown"
        elif not a.area.intersects(scene_area):
            reports.append(AirfieldReport(a.id, a.name, [], coverage="none"))
            continue
        elif scene_area.covers(a.area):
            coverage = "full"
        else:
            coverage = "partial"
            warnings.append(f'airfield "{a.name}" is only partially covered by the image, '
                            "so counts may be incomplete")
        lines = []
        for t in sorted(set(a.expected) | set(b["observed"])):
            exp, obs = a.expected.get(t, 0), b["observed"].get(t, 0)
            if t not in known and exp > 0:
                status = UNRECOGNIZABLE
            elif obs == exp:
                status = MATCH
            elif exp == 0:
                status = NOT_IN_REGISTRY
            elif obs < exp:
                status = MISSING
            else:
                status = EXTRA
            lines.append(TypeLine(t, exp, obs, b["borderline"].get(t, 0), status))
        reports.append(AirfieldReport(a.id, a.name, lines, b["unconfirmed"], b["untyped"], coverage))
    return InventoryReport(reports, outside, warnings)
