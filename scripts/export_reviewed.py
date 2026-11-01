"""Operator-reviewed detections -> YOLO-OBB dataset for fine-tuning (the improvement loop).

    python scripts/export_reviewed.py --out data/reviewed_yolo
    python scripts/export_reviewed.py --scene-id 3 --scene-id 4 --val-frac 0.5

Catalogue scenes are cut into the same tiles as during detection. A tile enters the dataset only
if ALL detections intersecting it have been reviewed: confirmed ones become labels, rejected ones
become background (hard negatives). Otherwise an unreviewed object would become "background" and
teach the model to ignore it.
Limitation: objects the model missed cannot be labelled this way; they need a tool such as CVAT.
Validation uses separate scenes (tiles of one scene overlap, so a random split would leak).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
from rasterio.windows import Window
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import db  # noqa: E402
from detector.model import DEFAULT_WEIGHTS  # noqa: E402
from detector.scene import open_raster, rgb_bands, stretch_limits, tile_windows, to_uint8  # noqa: E402


def export_reviewed(
    db_path: Path,
    out: Path,
    names: list[str],
    tile: int = 1024,
    overlap: int = 200,
    scene_ids: list[int] | None = None,
    val_frac: float = 0.2,
    min_visible: float = 0.5,
) -> dict:
    index = {name: i for i, name in enumerate(names)}
    with db.session(db_path) as conn:
        scenes = conn.execute("SELECT id, path, width, height FROM scenes WHERE status = 'done' ORDER BY id").fetchall()
        if scene_ids:
            scenes = [s for s in scenes if s["id"] in set(scene_ids)]
        detections = {
            s["id"]: conn.execute(
                "SELECT polygon_px, class_name, review_class, review_status FROM detections WHERE scene_id = ?",
                (s["id"],),
            ).fetchall()
            for s in scenes
        }

    n_val = round(len(scenes) * val_frac) if len(scenes) > 1 else 0
    val_ids = {s["id"] for s in scenes[len(scenes) - n_val :]} if n_val else set()
    out = out.resolve()
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"tiles": {"train": 0, "val": 0}, "skipped_pending": 0, "unknown_class": 0, "labels": Counter()}
    for scene in scenes:
        split = "val" if scene["id"] in val_ids else "train"
        polys = [(Polygon(json.loads(r["polygon_px"])).buffer(0), r) for r in detections[scene["id"]]]
        with open_raster(scene["path"]) as src:
            bands = rgb_bands(src)
            limits = stretch_limits(src, bands)
            for x0, y0, w, h in tile_windows(scene["width"], scene["height"], tile, overlap):
                tile_box = box(x0, y0, x0 + w, y0 + h)
                inside = [(p, r) for p, r in polys if p.intersects(tile_box)]
                if not inside:
                    continue  # nobody reviewed a tile without detections: it may contain misses
                if any(r["review_status"] == "pending" for _, r in inside):
                    stats["skipped_pending"] += 1
                    continue
                lines = []
                for poly, r in inside:
                    if r["review_status"] != "confirmed":
                        continue
                    if poly.area == 0 or poly.intersection(tile_box).area / poly.area < min_visible:
                        continue
                    name = r["review_class"] or r["class_name"]
                    if name not in index:
                        stats["unknown_class"] += 1
                        continue
                    coords = []
                    for x, y in json.loads(r["polygon_px"]):
                        coords += [min(max((x - x0) / w, 0.0), 1.0), min(max((y - y0) / h, 0.0), 1.0)]
                    lines.append(f"{index[name]} " + " ".join(f"{v:.6f}" for v in coords))
                    stats["labels"][name] += 1

                stem = f"scene{scene['id']}_{x0}_{y0}"
                image = to_uint8(src.read(bands, window=Window(x0, y0, w, h)), limits)
                cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                (out / "labels" / split / f"{stem}.txt").write_text("".join(f"{line}\n" for line in lines))
                stats["tiles"][split] += 1

    val_dir = "images/val" if stats["tiles"]["val"] else "images/train"
    yaml_lines = [f"path: {out}", "train: images/train", f"val: {val_dir}", "names:"]
    yaml_lines += [f"  {i}: {name}" for i, name in enumerate(names)]
    (out / "reviewed.yaml").write_text("\n".join(yaml_lines) + "\n")
    stats["labels"] = dict(stats["labels"])
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, default=ROOT / "data" / "catalog.sqlite")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "reviewed_yolo")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="where to take the class list from")
    p.add_argument("--scene-id", type=int, action="append", help="only these scenes (may be repeated)")
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--val-frac", type=float, default=0.2, help="share of scenes used for val")
    args = p.parse_args()

    from ultralytics import YOLO

    model_names = YOLO(str(args.weights), task="obb").names
    names = [model_names[i] for i in sorted(model_names)]
    stats = export_reviewed(args.db, args.out, names, args.tile, args.overlap, args.scene_id, args.val_frac)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["skipped_pending"]:
        print(f"! {stats['skipped_pending']} tiles skipped: they contain unreviewed detections")
    print(f"-> {args.out.resolve() / 'reviewed.yaml'}")


if __name__ == "__main__":
    main()
