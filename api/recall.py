"""Полнота обнаружения по сцене.

Точность известна из проверки: оператор подтверждает или отклоняет найденное. Полноту так
не узнать — пропущенного объекта в списке нет. Поэтому оператор отмечает пропуски на карте
отдельно, и полнота считается как доля найденного среди всего, что реально есть на снимке:

    существует  = подтверждённые обнаружения + отмеченные пропуски
    полнота(t)  = подтверждённые с уверенностью ≥ t / существует

Пропуск — объект, которого нет в каталоге вовсе, то есть не найденный даже при пороге
каталога 0.25. Поэтому один набор отметок даёт полноту сразу при нескольких порогах.
"""
from __future__ import annotations

THRESHOLDS = (0.25, 0.4)


def _block(confidences: list[float], n_missed: int, thresholds) -> dict:
    exist = len(confidences) + n_missed
    found = {f"{t:g}": sum(1 for c in confidences if c >= t) for t in thresholds}
    recall = {k: (round(v / exist, 3) if exist else None) for k, v in found.items()}
    return {"exist": exist, "missed": n_missed, "found": found, "recall": recall}


def scene_recall(detections: list[dict], missed: list[dict], complete: bool, thresholds=THRESHOLDS) -> dict:
    confirmed = [d for d in detections if d.get("review_status") == "confirmed"]
    pending = sum(1 for d in detections if d.get("review_status") == "pending")
    warnings = []
    if not complete:
        warnings.append("пропуски размечены не полностью — цифры ниже только верхняя оценка полноты")
    if pending:
        warnings.append(f"{pending} обнаружений не проверены — они не учтены ни как найденные, ни как существующие")

    def cls_of(d: dict) -> str:
        return d.get("review_class") or d["class_name"]

    classes = sorted({cls_of(d) for d in confirmed} | {m["class_name"] for m in missed})
    by_class = {
        c: _block([d["confidence"] for d in confirmed if cls_of(d) == c],
                  sum(1 for m in missed if m["class_name"] == c), thresholds)
        for c in classes
    }
    return {
        "complete": complete,
        "warnings": warnings,
        "thresholds": list(thresholds),
        "total": _block([d["confidence"] for d in confirmed], len(missed), thresholds),
        "by_class": by_class,
    }
