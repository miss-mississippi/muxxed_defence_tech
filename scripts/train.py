"""Дообучение YOLO-OBB с весов DOTA (трансферное обучение) и оценка лучшей эпохи.

    python scripts/train.py --data data/mar20_yolo/mar20.yaml --name mar20_s_800 --epochs 60 --imgsz 800 --batch 16
    python scripts/train.py --data dota8.yaml --name smoke --epochs 1 --imgsz 640 --batch 4 --device cpu

Лучшие веса копируются в weights/<name>.pt, метрики — в outputs/eval/ (test, если он есть в yaml, иначе val).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector.model import DEFAULT_WEIGHTS, pick_device  # noqa: E402
from scripts.eval import evaluate, to_markdown  # noqa: E402


def noise_augmentations(probability: float) -> list | None:
    """Набор аугментаций с гауссовым шумом (штатный параметр ultralytics `augmentations`).

    Замер устойчивости показал единственную реальную слабость модели — сенсорный шум
    (mAP50 0.900 → 0.371 при σ=15): в стандартном наборе аугментаций ultralytics шума нет.
    Повторяем стандартный набор и добавляем к нему GaussNoise.
    """
    try:
        import albumentations as A
    except ImportError:  # необязательная зависимость: без неё обучение идёт, просто без шума
        LOGGER.warning("albumentations не установлена, обучение пойдёт без шумовой аугментации")
        return None
    return [
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.GaussNoise(p=probability),
    ]


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
    p.add_argument("--noise", type=float, default=0.0, help="вероятность гауссова шума в аугментациях")
    p.add_argument("--robustness", type=int, default=400,
                   help="после обучения замерить устойчивость на N снимках (0 — не мерить)")
    args = p.parse_args()

    device = args.device or pick_device()
    extra = {}
    if args.noise > 0:
        augmentations = noise_augmentations(args.noise)
        if augmentations:
            extra["augmentations"] = augmentations
            LOGGER.info(f"аугментации: GaussNoise(p={args.noise}) добавлен к стандартному набору")
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
        **extra,
    )

    best = Path(model.trainer.best)
    target = ROOT / "weights" / f"{args.name}.pt"
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(best, target)
    print(f"лучшие веса: {best} -> {target}")

    split = "test" if check_det_dataset(args.data).get("test") else "val"
    report = evaluate(target, args.data, split=split, imgsz=args.imgsz, batch=args.batch, device=device.split(",")[0])
    print(to_markdown(report))

    if args.robustness:
        # меряем здесь, а не отдельной ячейкой ноутбука: так замер не зависит от того,
        # какая версия ноутбука оказалась у запускающего
        command = [
            sys.executable, str(ROOT / "scripts" / "robustness.py"),
            "--data", str(args.data), "--weights", str(target), "--split", split,
            "--limit", str(args.robustness), "--imgsz", str(args.imgsz),
            "--batch", str(args.batch), "--device", device.split(",")[0],
        ]
        print(f"\nзамер устойчивости: {' '.join(command[1:])}")
        subprocess.run(command, check=False)  # неудача замера не должна ронять результат обучения


if __name__ == "__main__":
    main()
