"""MAR20 (военные самолёты, 20 типов, OBB) → датасет YOLO-OBB для ultralytics.

    python scripts/prepare_mar20.py --src /path/to/MAR20 --out data/mar20_yolo

Ожидаемая структура исходника (официальный архив и зеркало на Hugging Face):
    MAR20/Annotations/Oriented Bounding Boxes/*.xml
    MAR20/ImageSets/Main/train.txt, test.txt
    MAR20/JPEGImages/*.jpg

Сплиты: официальный train делится на train/val (--val-frac), официальный test остаётся test.
Итоговые метрики считаем на test, val — только для выбора эпохи.
Лицензия MAR20: CC BY-NC 4.0 (некоммерческое использование).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# порядок A1..A20 из статьи MAR20 (Yu et al., National Remote Sensing Bulletin, 2023)
MAR20_NAMES = [
    "SU-35", "C-130", "C-17", "C-5", "F-16", "TU-160", "E-3", "B-52", "P-3C", "B-1B",
    "E-8", "TU-22", "F-15", "KC-135", "F-22", "FA-18", "TU-95", "KC-10", "SU-34", "SU-24",
]
CORNERS = ("left_top", "right_top", "right_bottom", "left_bottom")


def parse_obb_xml(path: Path) -> tuple[int, int, list[tuple[int, list[float]]]]:
    """→ (width, height, [(class_index, [x1, y1, ..., x4, y4])]) в пикселях."""
    root = ET.parse(path).getroot()
    width = int(float(root.findtext("size/width")))
    height = int(float(root.findtext("size/height")))
    objects = []
    for obj in root.iter("object"):
        box = obj.find("robndbox")
        if box is None:
            continue
        cls = int(obj.findtext("name").strip().lstrip("Aa")) - 1
        if not 0 <= cls < len(MAR20_NAMES):
            raise ValueError(f"{path.name}: неизвестный класс {obj.findtext('name')}")
        pts = []
        for corner in CORNERS:
            pts += [float(box.findtext(f"x_{corner}")), float(box.findtext(f"y_{corner}"))]
        objects.append((cls, pts))
    return width, height, objects


def yolo_obb_line(cls: int, pts: list[float], width: int, height: int) -> str:
    norm = [min(max(v / (width if i % 2 == 0 else height), 0.0), 1.0) for i, v in enumerate(pts)]
    return f"{cls} " + " ".join(f"{v:.6f}" for v in norm)


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def convert(src: Path, out: Path, val_frac: float = 0.15, seed: int = 0, copy: bool = False) -> dict:
    ann_dir = src / "Annotations" / "Oriented Bounding Boxes"
    img_dir = src / "JPEGImages"
    sets = src / "ImageSets" / "Main"
    rng = random.Random(seed)
    if (sets / "train.txt").exists() and (sets / "test.txt").exists():
        train_ids, test_ids = sorted(read_ids(sets / "train.txt")), read_ids(sets / "test.txt")
    else:
        print("! ImageSets/Main не найден — случайный сплит 35/65 как в оригинале")
        ids = sorted(p.stem for p in ann_dir.glob("*.xml"))
        rng.shuffle(ids)
        n_train = round(len(ids) * 0.35)
        train_ids, test_ids = sorted(ids[:n_train]), ids[n_train:]
    rng.shuffle(train_ids)
    n_val = round(len(train_ids) * val_frac)
    splits = {"train": train_ids[n_val:], "val": train_ids[:n_val], "test": test_ids}

    out = out.resolve()
    stats: dict = {"images": {}, "instances": {}, "missing": 0}
    for split, ids in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        counts: Counter = Counter()
        n_images = 0
        for image_id in ids:
            xml, img = ann_dir / f"{image_id}.xml", img_dir / f"{image_id}.jpg"
            if not xml.exists() or not img.exists():
                stats["missing"] += 1
                continue
            width, height, objects = parse_obb_xml(xml)
            dst = out / "images" / split / img.name
            if not dst.exists():
                shutil.copy2(img, dst) if copy else os.symlink(img.resolve(), dst)
            lines = [yolo_obb_line(c, p, width, height) for c, p in objects]
            (out / "labels" / split / f"{image_id}.txt").write_text("".join(f"{line}\n" for line in lines))
            counts.update(MAR20_NAMES[c] for c, _ in objects)
            n_images += 1
        stats["images"][split] = n_images
        stats["instances"][split] = {name: counts.get(name, 0) for name in MAR20_NAMES}

    yaml_lines = [f"path: {out}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml_lines += [f"  {i}: {name}" for i, name in enumerate(MAR20_NAMES)]
    (out / "mar20.yaml").write_text("\n".join(yaml_lines) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="корень MAR20")
    p.add_argument("--out", type=Path, default=Path("data/mar20_yolo"))
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--copy", action="store_true", help="копировать картинки вместо симлинков")
    args = p.parse_args()

    stats = convert(args.src, args.out, args.val_frac, args.seed, args.copy)
    print(f"изображений: {stats['images']}, пропущено без пары xml/jpg: {stats['missing']}")
    print(f"{'класс':<8}" + "".join(f"{s:>8}" for s in stats["instances"]))
    for name in MAR20_NAMES:
        print(f"{name:<8}" + "".join(f"{stats['instances'][s][name]:>8}" for s in stats["instances"]))
    print(f"-> {args.out.resolve() / 'mar20.yaml'}")


if __name__ == "__main__":
    main()
