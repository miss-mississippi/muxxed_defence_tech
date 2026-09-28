"""Сверка обнаруженных самолётов с реестром техники.

Реестр — список аэродромов: полигон в WGS84 и сколько бортов каждого типа там числится.
Обнаружение относится к аэродрому по координатам центра, дальше по типам сравнивается
«числится» и «обнаружено». Реестр — данные заказчика: сюда он приходит на вход и нигде
не сохраняется. В репозитории только условный пример, samples/registry_demo.json.

Что учтено, чтобы сверка не врала:
- тип засчитывается, только если ему не противоречат габариты; иначе обнаружение уходит
  в «тип не подтверждён» и не создаёт ложной тревоги о постороннем борте;
- «не обнаружено на снимке» не значит «отсутствует»: борт может быть в ангаре или в воздухе;
- тип, которого модель не знает, сверить нельзя — это отдельный статус;
- аэродром вне снимка — это «сверка невозможна», а не «все борта пропали»;
- грубее ~1 м тип самолёта не держится, отчёт об этом предупреждает.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import Point, box, shape
from shapely.geometry.base import BaseGeometry

from .physical import BORDERLINE, MISMATCH, REFERENCE, check_size

# огрубление демо-аэродрома: при 1 м тип определился у 3 бортов из 4, при 2 м — у одного
MAX_TYPE_GSD_M = 1.0
UNTYPED = frozenset({"plane"})  # общая модель находит самолёт, но не его тип

MATCH = "match"
MISSING = "missing"
EXTRA = "extra"
NOT_IN_REGISTRY = "not_in_registry"
UNRECOGNIZABLE = "unrecognizable"

STATUS_RU = {
    MATCH: "совпадает",
    MISSING: "не обнаружено на снимке",
    EXTRA: "сверх реестра",
    NOT_IN_REGISTRY: "тип не числится на аэродроме",
    UNRECOGNIZABLE: "модель этот тип не распознаёт",
}


@dataclass
class Airfield:
    id: str
    name: str
    area: BaseGeometry
    expected: dict[str, int]


def parse_registry(data: dict) -> list[Airfield]:
    if not isinstance(data, dict) or not isinstance(data.get("airfields"), list) or not data["airfields"]:
        raise ValueError("реестр: ожидается объект с непустым списком airfields")
    airfields = []
    for i, a in enumerate(data["airfields"]):
        try:
            area = shape(a["area"])
            expected = {str(t): int(n) for t, n in a["expected"].items()}
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"реестр: аэродром #{i} — нужны area (GeoJSON) и expected {{тип: число}}") from exc
        if area.is_empty or not area.is_valid:
            raise ValueError(f"реестр: у аэродрома #{i} пустой или некорректный полигон")
        if any(n < 0 for n in expected.values()):
            raise ValueError(f"реестр: у аэродрома #{i} отрицательное число бортов")
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
    unconfirmed: list[dict] = field(default_factory=list)  # тип назван, габариты против
    untyped: list[dict] = field(default_factory=list)      # самолёт без типа
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
                         "status_ru": STATUS_RU[line.status]}
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
            out.append(f"Аэродром «{a.name}»")
            if a.coverage == "none":
                out.append("  аэродром вне снимка — сверка невозможна")
                continue
            for line in a.lines:
                seen = str(line.observed)
                if line.borderline:
                    seen += f" (на границе допуска: {line.borderline})"
                verdict = STATUS_RU[line.status]
                if line.status in (MISSING, EXTRA):
                    verdict += f": {abs(line.observed - line.expected)}"
                out.append(f"  {line.type:<7} числится {line.expected:>2}   обнаружено {seen:<28} {verdict}")
            if a.unconfirmed:
                items = ", ".join(
                    f"{u['class']} ({u['size_deviation']:.1%})" if u.get("size_deviation") is not None else u["class"]
                    for u in a.unconfirmed
                )
                out.append(f"  тип назван моделью, но габариты против — {len(a.unconfirmed)}: {items}")
            if a.untyped:
                out.append(f"  самолёт без определённого типа — {len(a.untyped)}")
            unresolved = len(a.unconfirmed) + len(a.untyped)
            if a.missing and unresolved:
                out.append(f"  среди {unresolved} бортов без подтверждённого типа могут быть "
                           f"не обнаруженные ({a.missing})")
        if self.outside:
            out.append(f"Самолёты вне аэродромов реестра: {len(self.outside)}")
        return "\n".join(out)


def _size_verdict(p: dict, cls: str) -> tuple[str | None, float | None]:
    # если оператор исправил класс, сохранённый вердикт считался для старого — пересчитываем
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
    """features — GeoJSON-объекты или их properties, из detect_scene или из каталога.

    scene_bounds — контур снимка [west, south, east, north] в WGS84; без него покрытие
    аэродромов снимком не проверяется.
    known_types — какие типы умеет называть модель; по умолчанию 20 типов MAR20.
    use_review — учитывать решение оператора: исправленный класс и отклонённые объекты.
    trust_size_check=False нужен только для сравнения: без сверки габаритов ошибочные
    типы попадают в счёт и дают ложные расхождения.
    """
    known = set(REFERENCE) if known_types is None else set(known_types)
    warnings = []
    if gsd_m is not None and gsd_m > max_type_gsd:
        warnings.append(f"разрешение {gsd_m:.2f} м/px грубее {max_type_gsd:g} м: тип самолёта при таком "
                        f"разрешении не держится, сверка по типам ориентировочная")

    buckets = {a.id: {"observed": Counter(), "borderline": Counter(), "unconfirmed": [], "untyped": []}
               for a in airfields}
    outside, no_coords = [], 0
    for f in features:
        p = f.get("properties", f)
        if use_review and p.get("review_status") == "rejected":
            continue
        cls = (p.get("review_class") if use_review else None) or p["class_name"]
        if cls not in UNTYPED and cls not in known:
            continue  # не самолёт
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
        warnings.append(f"{no_coords} самолётов без координат (снимок без геопривязки) к аэродромам не отнесены")

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
            warnings.append(f"аэродром «{a.name}» покрыт снимком частично — счёт может быть неполным")
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
