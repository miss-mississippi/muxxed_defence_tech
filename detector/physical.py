"""Сверка типа с измеренными габаритами.

Сцена геопривязана, масштаб пикселя известен, поэтому размеры объекта считаются в метрах.
Самолёт заявленного типа не может быть вдвое меньше паспортного — на этом и строится проверка.
Так отсеялся C-5 Galaxy с габаритами 38.8 x 35.6 м при паспортных 75.5 x 67.9 м.

Совпадение размеров типа не доказывает: Boeing 767 (48.5 x 47.6 м) неотличим по габаритам
от E-8 на базе Boeing 707 (46.6 x 44.4 м). Ловятся только грубые промахи, поэтому
вердикта три: габариты сходятся, на границе допуска, тип не подтверждён.

Эталоны — открытые справочники: длина фюзеляжа и размах крыла.
"""
from __future__ import annotations

from dataclasses import dataclass

# длина фюзеляжа и размах крыла, метры; span_swept — размах со сложенным крылом
# для машин с изменяемой геометрией (Ту-160, B-1B, Ту-22М, Су-24)
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

OK = "ok"                    # габариты совпадают с паспортными
BORDERLINE = "borderline"    # расхождение заметное, но объяснимое условиями съёмки
MISMATCH = "mismatch"        # габариты не сходятся — тип не подтверждён геометрией
UNKNOWN = "unknown"          # нет эталона для класса или нет геопривязки


@dataclass(frozen=True)
class SizeCheck:
    verdict: str
    deviation: float | None = None       # наибольшее относительное отклонение, 0.48 = 48%
    expected_long: tuple[float, float] | None = None
    expected_short: tuple[float, float] | None = None

    @property
    def confirmed(self) -> bool:
        return self.verdict == OK

    def describe(self) -> str:
        if self.verdict == UNKNOWN:
            return "размер не проверялся"
        assert self.expected_long and self.expected_short
        span = f"{self.expected_long[0]:.0f}–{self.expected_long[1]:.0f}" \
            if self.expected_long[0] != self.expected_long[1] else f"{self.expected_long[0]:.0f}"
        if self.verdict == OK:
            return f"габариты соответствуют типу (ожидается ~{span} м по длинной оси)"
        if self.verdict == BORDERLINE:
            return (f"на границе допуска: расхождение {self.deviation:.0%} "
                    f"(ожидается ~{span} м по длинной оси)")
        return (f"тип не подтверждён геометрией: расхождение {self.deviation:.0%} "
                f"с паспортным размером (ожидается ~{span} м по длинной оси)")


def _bounds(length: float, span: float, span_swept: float | None) -> tuple[tuple[float, float], tuple[float, float]]:
    """Допустимые длинная и короткая оси с учётом всех положений крыла."""
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
    """Сверяет измеренные габариты с паспортными для типа.

    Градаций три, а не порог: двоичный ответ у самой границы вводит в заблуждение
    в обе стороны. До borderline (15%) расхождение объясняется ошибкой бокса, тенью
    и точностью геопривязки; от borderline до tolerance (25%) оно заметное, но само
    по себе ошибкой типа не является; дальше тип считается неподтверждённым.
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
    """Добавляет к свойствам детекции результат проверки габаритами (на месте)."""
    check = check_size(
        properties.get("class_name", ""), properties.get("length_m"), properties.get("width_m"),
        tolerance, borderline,
    )
    properties["size_check"] = check.verdict
    if check.verdict != UNKNOWN:
        properties["size_deviation"] = check.deviation
    return properties
