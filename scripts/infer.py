"""Инференс YOLO-OBB на одном снимке: JSON с детекциями + картинка с боксами.

    python scripts/infer.py samples/boats.jpg
    python scripts/infer.py path/to/image.jpg --device cpu --conf 0.3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils.plotting import colors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector import DEFAULT_WEIGHTS, Detection, Detector  # noqa: E402


def draw(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    # рисуем именно из Detection, чтобы картинка проверяла экспортируемые координаты
    canvas = image.copy()
    for d in detections:
        color = colors(d.class_id, bgr=True)
        pts = np.round(np.array(d.polygon)).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        cv2.putText(
            canvas,
            f"{d.class_name} {d.confidence:.2f}",
            (x, max(y - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="путь к снимку")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--device", default=None, help="cpu | mps | 0 (по умолчанию — автовыбор)")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--runs", type=int, default=5, help="прогонов для замера времени после прогрева")
    p.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = p.parse_args()

    image = cv2.imread(str(args.source))
    if image is None:
        sys.exit(f"не удалось прочитать {args.source}")

    detector = Detector(args.weights, device=args.device, imgsz=args.imgsz, conf=args.conf)
    detector.predict(image)  # прогрев: первый прогон на MPS/CUDA заметно медленнее

    wall_ms, model_ms = [], []
    for _ in range(max(args.runs, 1)):
        t0 = time.perf_counter()
        detections = detector.predict(image)
        wall_ms.append((time.perf_counter() - t0) * 1000)
        model_ms.append(detector.last_speed["inference"])

    h, w = image.shape[:2]
    counts = Counter(d.class_name for d in detections)
    report = {
        "image": str(args.source),
        "width": w,
        "height": h,
        "weights": args.weights.name,
        "device": detector.device,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "timing_ms": {
            "end_to_end_median": round(statistics.median(wall_ms), 1),
            "model_inference_median": round(statistics.median(model_ms), 1),
            "runs": len(wall_ms),
        },
        "counts": dict(counts.most_common()),
        "detections": [d.to_dict() for d in detections],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"{args.source.stem}.json"
    img_path = args.out / f"{args.source.stem}_obb.jpg"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    cv2.imwrite(str(img_path), draw(image, detections))

    print(f"{args.source.name}: {w}x{h}, device={detector.device}, imgsz={args.imgsz}")
    print(f"детекций: {len(detections)}  {dict(counts.most_common())}")
    print(
        f"время (медиана из {len(wall_ms)}): end-to-end {report['timing_ms']['end_to_end_median']} мс, "
        f"модель {report['timing_ms']['model_inference_median']} мс"
    )
    print(f"-> {json_path}\n-> {img_path}")


if __name__ == "__main__":
    main()
