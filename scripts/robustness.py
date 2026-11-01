"""Model robustness to image quality: noise, blur, compression, lower resolution, low light, haze.

    python scripts/robustness.py --data data/mar20_yolo/mar20.yaml --weights weights/mar20_s_800.pt --split test --limit 400

Takes a dataset split, makes a degraded copy of the images (labels unchanged) and computes the usual
metrics on it. The result is a "degradation -> mAP50 / mAP50-95 / drop" table.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.utils import check_det_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval import evaluate  # noqa: E402

OUT_DIR = ROOT / "outputs" / "robustness"


def _noise(image: np.ndarray, sigma: float) -> np.ndarray:
    noisy = image.astype(np.float32) + np.random.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _blur(image: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else image


def _downscale(image: np.ndarray, factor: int) -> np.ndarray:
    """Simulate coarser resolution: downscale, then upscale back."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _brightness(image: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _haze(image: np.ndarray, strength: float) -> np.ndarray:
    """Haze / thin cloud: blend in white."""
    white = np.full_like(image, 255)
    return cv2.addWeighted(image, 1 - strength, white, strength, 0)


DEGRADATIONS = {
    "original": lambda im: im,
    "noise sigma=15": lambda im: _noise(im, 15),
    "noise sigma=30": lambda im: _noise(im, 30),
    "blur 5 px": lambda im: _blur(im, 5),
    "blur 9 px": lambda im: _blur(im, 9),
    "JPEG q=30": lambda im: _jpeg(im, 30),
    "resolution /2": lambda im: _downscale(im, 2),
    "resolution /4": lambda im: _downscale(im, 4),
    "brightness x0.5": lambda im: _brightness(im, 0.5),
    "haze 30%": lambda im: _haze(im, 0.3),
}


def build_variant(images: list[Path], labels_dir: Path, out: Path, transform, names: dict) -> Path:
    """Copy of the split with degraded images and the original labels -> path to the yaml."""
    img_dir, lbl_dir = out / "images" / "val", out / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for src in images:
        image = cv2.imread(str(src))
        if image is None:
            continue
        cv2.imwrite(str(img_dir / f"{src.stem}.jpg"), transform(image))
        label = labels_dir / f"{src.stem}.txt"
        if label.exists():
            shutil.copy2(label, lbl_dir / label.name)
    yaml_path = out / "data.yaml"
    lines = [f"path: {out.resolve()}", "train: images/val", "val: images/val"]
    lines += ["names:"] + [f"  {i}: {n}" for i, n in sorted(names.items())]
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="source dataset yaml")
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=400, help="how many images to use (for speed)")
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    data = check_det_dataset(args.data)
    split_dir = Path(data[args.split] if args.split in data and data[args.split] else data["val"])
    images = sorted(p for p in split_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.limit]
    if not images:
        sys.exit(f"no images in {split_dir}")
    labels_dir = Path(str(split_dir).replace("/images/", "/labels/"))
    print(f"images under test: {len(images)}, labels: {labels_dir}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, transform in DEGRADATIONS.items():
        tag = name.replace(" ", "_").replace("/", "").replace("%", "").replace("=", "")
        variant = OUT_DIR / "data" / tag
        shutil.rmtree(variant, ignore_errors=True)
        yaml_path = build_variant(images, labels_dir, variant, transform, data["names"])
        report = evaluate(args.weights, yaml_path, split="val", imgsz=args.imgsz, batch=args.batch,
                          device=args.device, out_dir=OUT_DIR / "eval")
        rows.append({"degradation": name, "map50": report["map50"], "map50_95": report["map50_95"],
                     "precision": report["precision"], "recall": report["recall"]})
        print(f"  {name:16} mAP50={report['map50']:.3f} mAP50-95={report['map50_95']:.3f}")
        shutil.rmtree(variant, ignore_errors=True)

    base = rows[0]["map50"] or 1e-9
    table = ["| Degradation | mAP50 | mAP50-95 | P | R | mAP50 change |", "|---|---|---|---|---|---|"]
    for r in rows:
        drop = (r["map50"] - base) / base * 100
        table.append(f"| {r['degradation']} | {r['map50']:.3f} | {r['map50_95']:.3f} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} | {drop:+.1f}% |")
    markdown = (f"**{Path(args.weights).name}** · {Path(args.data).name} / {args.split} · "
                f"{len(images)} images · imgsz {args.imgsz}\n\n" + "\n".join(table) + "\n")
    (OUT_DIR / "robustness.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (OUT_DIR / "robustness.md").write_text(markdown)
    print("\n" + markdown)
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
