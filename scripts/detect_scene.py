"""Детекция на целой сцене (GeoTIFF/JPG/PNG): тайлы → YOLO-OBB → склейка → GeoJSON.

    python scripts/detect_scene.py data/scenes/airport.tif
    python scripts/detect_scene.py scene.tif --tile 1024 --overlap 200 --conf 0.3

Результат: outputs/<имя>.geojson (WGS84, если снимок геопривязан) и
outputs/<имя>_overview.jpg (уменьшенная сцена с боксами для быстрой проверки).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils.plotting import colors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector import DEFAULT_WEIGHTS, Detector, SceneResult, detect_scene  # noqa: E402
from detector.scene import _read_scaled, open_raster, rgb_bands, stretch_limits, to_uint8  # noqa: E402


def overview(result: SceneResult, max_size: int = 3000) -> np.ndarray:
    with open_raster(result.path) as src:
        bands = rgb_bands(src)
        rgb = to_uint8(_read_scaled(src, bands, max_size), stretch_limits(src, bands))
    scale = rgb.shape[1] / result.width
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for d in result.detections:
        pts = np.round(np.array(d.polygon) * scale).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=colors(d.class_id, bgr=True), thickness=2)
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--device", default=None, help="cpu | mps | 0 (по умолчанию — автовыбор)")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--ios", type=float, default=0.6, help="порог IoS склейки дублей (>1 — не склеивать)")
    p.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = p.parse_args()

    detector = Detector(args.weights, device=args.device, imgsz=args.imgsz, conf=args.conf)

    def progress(done: int, total: int) -> None:
        print(f"\rтайлы: {done}/{total}", end="", flush=True)

    result = detect_scene(
        detector,
        args.source,
        tile=args.tile,
        overlap=args.overlap,
        batch=args.batch,
        ios_threshold=args.ios,
        progress=progress,
    )
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    geojson_path = args.out / f"{args.source.stem}.geojson"
    overview_path = args.out / f"{args.source.stem}_overview.jpg"
    geojson_path.write_text(json.dumps(result.to_geojson(), ensure_ascii=False))
    cv2.imwrite(str(overview_path), overview(result))

    s = result.summary()
    print(f"{args.source.name}: {s['width']}x{s['height']}, crs={s['crs']}, gsd={s['gsd_m']} м/px")
    print(f"тайлов: {s['tiles_total']} (пустых {s['tiles_skipped']}), время {s['elapsed_s']} с, device={detector.device}")
    print(f"детекций: {s['detections']}  {s['counts']}")
    for w in s["warnings"]:
        print(f"! {w}")
    print(f"-> {geojson_path}\n-> {overview_path}")


if __name__ == "__main__":
    main()
