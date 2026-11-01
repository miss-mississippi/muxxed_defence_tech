"""Fine-tune YOLO-OBB from the DOTA weights (transfer learning) and evaluate the best epoch.

    python scripts/train.py --data data/mar20_yolo/mar20.yaml --name mar20_s_800 --epochs 60 --imgsz 800 --batch 16
    python scripts/train.py --data dota8.yaml --name smoke --epochs 1 --imgsz 640 --batch 4 --device cpu

The best weights are copied to weights/<name>.pt, metrics go to outputs/eval/ (test if the yaml has it, else val).
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
    """The standard ultralytics augmentation set plus Gaussian noise.

    The default set has no noise, which is why the model collapsed on noisy images:
    mAP50 0.900 -> 0.371 at sigma=15.
    """
    try:
        import albumentations as A
    except ImportError:  # optional dependency: training still runs, just without noise
        LOGGER.warning("albumentations is not installed, training will run without noise augmentation")
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
    p.add_argument("--data", required=True, help="YOLO-OBB dataset yaml")
    p.add_argument("--name", required=True, help="run name -> runs/obb/<name>, weights/<name>.pt")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="starting weights (DOTA yolo11s-obb by default)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="cpu | mps | 0 | 0,1 (auto-selected by default)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=15, help="early stopping patience without val improvement")
    p.add_argument("--degrees", type=float, default=0.0, help="rotation augmentation, +/- degrees")
    p.add_argument("--freeze", type=int, default=0, help="freeze the first N layers")
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of train used, for quick runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.0, help="probability of Gaussian noise in augmentations")
    p.add_argument("--robustness", type=int, default=400,
                   help="after training, measure robustness on N images (0 to skip)")
    args = p.parse_args()

    device = args.device or pick_device()
    extra = {}
    if args.noise > 0:
        augmentations = noise_augmentations(args.noise)
        if augmentations:
            extra["augmentations"] = augmentations
            LOGGER.info(f"augmentations: GaussNoise(p={args.noise}) added to the standard set")
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
        # an overhead image has no "up": vertical flips and rotation do not distort the scene
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
    print(f"best weights: {best} -> {target}")

    split = "test" if check_det_dataset(args.data).get("test") else "val"
    report = evaluate(target, args.data, split=split, imgsz=args.imgsz, batch=args.batch, device=device.split(",")[0])
    print(to_markdown(report))

    if args.robustness:
        # measured here rather than in a separate notebook cell, so the measurement does not
        # depend on which version of the notebook the runner happens to have
        command = [
            sys.executable, str(ROOT / "scripts" / "robustness.py"),
            "--data", str(args.data), "--weights", str(target), "--split", split,
            "--limit", str(args.robustness), "--imgsz", str(args.imgsz),
            "--batch", str(args.batch), "--device", device.split(",")[0],
        ]
        print(f"\nrobustness run: {' '.join(command[1:])}")
        subprocess.run(command, check=False)  # a failed measurement must not break the training result


if __name__ == "__main__":
    main()
