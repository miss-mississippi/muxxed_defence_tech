"""Оценка модели YOLO-OBB: mAP50, mAP50-95, precision/recall по классам, скорость инференса.

    python scripts/eval.py --data data/mar20_yolo/mar20.yaml --weights weights/mar20_s.pt --split test
    python scripts/eval.py --data dota8.yaml     # smoke-тест на встроенном мини-датасете ultralytics

Отчёт: outputs/eval/<веса>_<датасет>_<сплит>.json и .md (таблица для NOTES и презентации).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector.model import DEFAULT_WEIGHTS, pick_device  # noqa: E402

EVAL_DIR = ROOT / "outputs" / "eval"


def evaluate(
    weights: str | Path,
    data: str | Path,
    split: str = "val",
    imgsz: int = 1024,
    batch: int = 8,
    device: str | None = None,
    out_dir: Path = EVAL_DIR,
) -> dict:
    weights, device = Path(weights), device or pick_device()
    tag = f"{weights.stem}_{Path(str(data)).stem}_{split}"
    model = YOLO(str(weights), task="obb")
    metrics = model.val(
        data=str(data), split=split, imgsz=imgsz, batch=batch, device=device,
        plots=True, project=str(out_dir), name=tag, exist_ok=True, verbose=False,
    )
    box = metrics.box
    nt_class = getattr(metrics, "nt_per_class", None)
    nt_image = getattr(metrics, "nt_per_image", None)
    per_class = []
    for i, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class.append({
            "class": metrics.names[int(c)],
            "images": int(nt_image[c]) if nt_image is not None else None,
            "instances": int(nt_class[c]) if nt_class is not None else None,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "map50": round(float(ap50), 4),
            "map50_95": round(float(ap), 4),
        })
    report = {
        "weights": str(weights),
        "data": str(data),
        "split": split,
        "imgsz": imgsz,
        "device": device,
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in metrics.speed.items()},
        "per_class": per_class,
        "plots_dir": str(out_dir / tag),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out_dir / f"{tag}.md").write_text(to_markdown(report))
    return report


def to_markdown(r: dict) -> str:
    speed = r["speed_ms_per_image"]
    lines = [
        f"**{Path(r['weights']).name}** · `{Path(r['data']).name}` / {r['split']} · imgsz {r['imgsz']} · {r['device']}",
        "",
        f"mAP50 **{r['map50']:.3f}** · mAP50-95 **{r['map50_95']:.3f}** · P {r['precision']:.3f} · R {r['recall']:.3f} · "
        f"инференс {speed.get('inference', 0):.1f} мс/изобр. (с пре/постобработкой {sum(speed.values()):.1f} мс)",
        "",
        "| Класс | Изобр. | Объектов | P | R | mAP50 | mAP50-95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in r["per_class"]:
        lines.append(
            f"| {c['class']} | {c['images']} | {c['instances']} | {c['precision']:.3f} | {c['recall']:.3f} "
            f"| {c['map50']:.3f} | {c['map50_95']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="yaml датасета YOLO-OBB")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="cpu | mps | 0 (по умолчанию — автовыбор)")
    args = p.parse_args()

    report = evaluate(args.weights, args.data, args.split, args.imgsz, args.batch, args.device)
    print(to_markdown(report))
    print(f"-> {EVAL_DIR}")


if __name__ == "__main__":
    main()
