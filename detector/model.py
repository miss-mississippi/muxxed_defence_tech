"""YOLO-OBB wrapper: loads weights and returns predictions as Detection objects."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from .schema import Detection

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weights" / "yolo11s-obb.pt"

ImageInput = Union[str, Path, np.ndarray]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        device: str | None = None,
        imgsz: int = 1024,
        conf: float = 0.25,
        iou: float = 0.7,
        name: str | None = None,
        agnostic_nms: bool = False,
    ):
        weights = Path(weights)
        if not weights.exists():
            # official ultralytics weights are downloaded by file name from GitHub releases
            weights.parent.mkdir(parents=True, exist_ok=True)
            attempt_download_asset(weights)
        self.weights = weights
        self.name = name or weights.stem  # stored in the model field of every detection
        self.model = YOLO(str(weights), task="obb")
        self.names: dict[int, str] = self.model.names
        self.device = device or pick_device()
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.agnostic_nms = agnostic_nms  # for type models: an object cannot be two types at once
        self.last_speed: dict[str, float] = {}

    def predict(
        self, image: ImageInput, offset: tuple[float, float] = (0, 0), rgb: bool = False
    ) -> list[Detection]:
        """Detections on a single image or tile.

        image  -- file path or an HxWx3 uint8 np.ndarray (BGR by default, as from cv2.imread;
                  pass rgb=True for arrays coming from rasterio/PIL).
        offset -- (x0, y0) of the tile's top-left corner in scene pixels.
        """
        return self.predict_batch([image], [offset], rgb=rgb)[0]

    def predict_batch(
        self,
        images: Sequence[ImageInput],
        offsets: Sequence[tuple[float, float]] | None = None,
        rgb: bool = False,
    ) -> list[list[Detection]]:
        offsets = list(offsets) if offsets is not None else [(0, 0)] * len(images)
        if len(offsets) != len(images):
            raise ValueError("the number of offsets must match the number of images")
        prepared = [self._prepare(im, rgb) for im in images]
        results = self.model.predict(
            prepared,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            agnostic_nms=self.agnostic_nms,
            device=self.device,
            verbose=False,
        )
        self.last_speed = results[0].speed
        return [self._to_detections(r, off) for r, off in zip(results, offsets)]

    @staticmethod
    def _prepare(image: ImageInput, rgb: bool) -> str | np.ndarray:
        if isinstance(image, (str, Path)):
            return str(image)
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected an np.ndarray of shape HxWx3")
        if image.dtype != np.uint8:
            raise ValueError(
                f"expected uint8, got {image.dtype}: stretch 16-bit imagery to 0..255 first"
            )
        return np.ascontiguousarray(image[..., ::-1]) if rgb else image

    def _to_detections(self, result, offset: tuple[float, float]) -> list[Detection]:
        obb = result.obb
        if obb is None or len(obb) == 0:
            return []
        dx, dy = float(offset[0]), float(offset[1])
        polys = obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
        xywhr = obb.xywhr.cpu().numpy()  # (N, 5)
        confs = obb.conf.cpu().numpy()
        classes = obb.cls.cpu().numpy().astype(int)

        detections = []
        for poly, (cx, cy, w, h, angle), conf, cls in zip(polys, xywhr, confs, classes):
            detections.append(
                Detection(
                    class_id=int(cls),
                    class_name=self.names[int(cls)],
                    confidence=round(float(conf), 4),
                    polygon=[[round(float(x) + dx, 2), round(float(y) + dy, 2)] for x, y in poly],
                    cx=round(float(cx) + dx, 2),
                    cy=round(float(cy) + dy, 2),
                    w=round(float(w), 2),
                    h=round(float(h), 2),
                    angle=round(float(angle), 4),
                    model=self.name,
                )
            )
        return detections
