"""Дообучение YOLO-OBB с весов DOTA (трансферное обучение) и оценка лучшей эпохи.

    python scripts/train.py --data data/mar20_yolo/mar20.yaml --name mar20_s_800 --epochs 60 --imgsz 800 --batch 16
    python scripts/train.py --data dota8.yaml --name smoke --epochs 1 --imgsz 640 --batch 4 --device cpu

Лучшие веса копируются в weights/<name>.pt, метрики — в outputs/eval/ (test, если он есть в yaml, иначе val).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector.model import DEFAULT_WEIGHTS, pick_device  # noqa: E402
from scripts.eval import evaluate, to_markdown  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="yaml датасета YOLO-OBB")
    p.add_argument("--name", required=True, help="имя прогона → runs/obb/<name>, weights/<name>.pt")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="стартовые веса (по умолчанию DOTA yolo11s-obb)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="cpu | mps | 0 | 0,1 (по умолчанию — автовыбор)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=15, help="ранняя остановка без улучшения val")
    p.add_argument("--degrees", type=float, default=0.0, help="аугментация поворотом ±градусы")
    p.add_argument("--freeze", type=int, default=0, help="заморозить первые N слоёв")
    p.add_argument("--fraction", type=float, default=1.0, help="доля train для быстрых прогонов")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device or pick_device()
    model = YOLO(args.weights, task="obb")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=str(ROOT / "runs" / "obb"),
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        deterministic=True,
        fraction=args.fraction,
        freeze=args.freeze or None,
        cos_lr=True,
        close_mosaic=min(10, args.epochs),
        # у снимка сверху нет «верха»: вертикальный флип и поворот не искажают сцену
        flipud=0.5,
        fliplr=0.5,
        degrees=args.degrees,
        plots=True,
    )

    best = Path(model.trainer.best)
    target = ROOT / "weights" / f"{args.name}.pt"
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(best, target)
    print(f"лучшие веса: {best} -> {target}")

    split = "test" if check_det_dataset(args.data).get("test") else "val"
    report = evaluate(target, args.data, split=split, imgsz=args.imgsz, batch=args.batch, device=device.split(",")[0])
    print(to_markdown(report))


if __name__ == "__main__":
    main()
