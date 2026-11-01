"""Per-scene detection recall.

Precision comes from review: the operator confirms or rejects what was found. Recall cannot be
learned that way, because a missed object is not in the list. So the operator marks misses on the
map separately, and recall is the share of what was found among everything actually in the image:

    exist      = confirmed detections + marked misses
    recall(t)  = confirmed with confidence >= t / exist

A miss is an object absent from the catalogue altogether, i.e. not found even at the catalogue
threshold of 0.25. That is why one set of marks gives recall at several thresholds at once.
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
        warnings.append("misses are not fully marked: the numbers below are only an upper bound on recall")
    if pending:
        warnings.append(f"{pending} detections are not reviewed yet: they count neither as found nor as existing")

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
