"""Производительность детекции на большой сцене: тайлы/с, Мпикс/с, км²/ч.

    python scripts/benchmark.py                               # синтетическая сцена 8192×8192, GSD 0.5 м
    python scripts/benchmark.py --scene path/to/scene.tif     # реальная сцена (GSD берётся из геопривязки)

Синтетическая сцена — мозаика из samples/boats.jpg: число объектов не показательно, скорость — да.
Отчёт: outputs/benchmark/<сцена>_<устройство>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector import DEFAULT_WEIGHTS, Detector, detect_scene  # noqa: E402


def synthetic_scene(path: Path, size: int, gsd: float) -> None:
    sample = cv2.imread(str(ROOT / "samples" / "boats.jpg"))[..., ::-1]
    reps = (size // sample.shape[0] + 1, size // sample.shape[1] + 1, 1)
    mosaic = np.tile(sample, reps)[:size, :size]
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8", tiled=True,
        compress="deflate", crs="EPSG:32642", transform=from_origin(500000.0, 5670000.0, gsd, gsd),
    ) as dst:
        dst.write(np.ascontiguousarray(mosaic.transpose(2, 0, 1)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, default=None, help="сцена; по умолчанию синтетическая")
    p.add_argument("--size", type=int, default=8192, help="сторона синтетической сцены, px")
    p.add_argument("--gsd", type=float, default=0.5, help="м/px синтетической сцены")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--device", default=None)
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()

    detector = Detector(args.weights, device=args.device)
    detector.predict(np.zeros((args.tile, args.tile, 3), np.uint8))  # прогрев

    with tempfile.TemporaryDirectory() as tmp:
        scene = args.scene
        if scene is None:
            scene = Path(tmp) / f"synthetic_{args.size}.tif"
            synthetic_scene(scene, args.size, args.gsd)
        result = detect_scene(detector, scene, tile=args.tile, overlap=args.overlap, batch=args.batch)

    megapixels = result.width * result.height / 1e6
    gsd = result.gsd_m or args.gsd
    area_km2 = megapixels * gsd**2
    tiles = result.tiles_total - result.tiles_skipped
    report = {
        "scene": scene.name,
        "size_px": [result.width, result.height],
        "gsd_m": gsd,
        "area_km2": round(area_km2, 2),
        "device": detector.device,
        "weights": args.weights.name,
        "tile": args.tile,
        "overlap": args.overlap,
        "batch": args.batch,
        "tiles": tiles,
        "elapsed_s": result.elapsed_s,
        "tiles_per_s": round(tiles / result.elapsed_s, 2),
        "megapixels_per_s": round(megapixels / result.elapsed_s, 2),
        "km2_per_hour": round(area_km2 / result.elapsed_s * 3600, 1),
        "detections": len(result.detections),
    }
    out = ROOT / "outputs" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{Path(report['scene']).stem}_{detector.device}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
