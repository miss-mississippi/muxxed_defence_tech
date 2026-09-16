"""Несколько моделей одновременно: общая (DOTA: суда, резервуары, техника)
и дообученная по типам (MAR20: типы военных самолётов).

Каждая детекция помечается именем модели. Если один объект нашли обе модели,
побеждает более приоритетная — специализированная важнее общей (`merge.suppress_cross_model`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .model import DEFAULT_WEIGHTS, Detector, ImageInput
from .schema import Detection

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ModelSpec:
    name: str
    weights: Path
    priority: int = 0  # при наложении объектов выигрывает модель с большим приоритетом


class ModelRegistry:
    """Прогоняет изображение через все модели и складывает детекции в один список."""

    def __init__(
        self,
        specs: Sequence[ModelSpec],
        device: str | None = None,
        imgsz: int = 1024,
        conf: float = 0.25,
        iou: float = 0.7,
    ):
        specs = list(specs)
        if not specs:
            raise ValueError("нужна хотя бы одна модель")
        self.specs = {s.name: s for s in specs}
        self.detectors = {
            s.name: Detector(s.weights, device=device, imgsz=imgsz, conf=conf, iou=iou, name=s.name)
            for s in specs
        }

    @classmethod
    def single(cls, detector: Detector) -> "ModelRegistry":
        """Обёртка вокруг уже загруженной модели — чтобы старый код работал без изменений."""
        registry = cls.__new__(cls)
        registry.specs = {detector.name: ModelSpec(detector.name, detector.weights)}
        registry.detectors = {detector.name: detector}
        return registry

    @classmethod
    def from_env(
        cls, value: str | None = None, device: str | None = None, imgsz: int = 1024, conf: float = 0.25
    ) -> "ModelRegistry":
        """value: "dota=weights/yolo11s-obb.pt,mar20=weights/mar20_s_800.pt".
        Порядок задаёт приоритет: следующая модель важнее предыдущей.
        Пусто — одна предобученная модель DOTA."""
        specs = []
        for priority, item in enumerate(v.strip() for v in (value or "").split(",") if v.strip()):
            name, _, path = item.partition("=")
            if not path:
                raise ValueError(f"ожидается имя=путь, получено {item!r}")
            weights = Path(path.strip())
            specs.append(ModelSpec(name.strip(), weights if weights.is_absolute() else ROOT / weights, priority))
        if not specs:
            specs = [ModelSpec("dota", DEFAULT_WEIGHTS)]
        return cls(specs, device=device, imgsz=imgsz, conf=conf)

    @property
    def priorities(self) -> dict[str, int]:
        return {name: spec.priority for name, spec in self.specs.items()}

    @property
    def device(self) -> str:
        return next(iter(self.detectors.values())).device

    @property
    def imgsz(self) -> int:
        return next(iter(self.detectors.values())).imgsz

    @property
    def conf(self) -> float:
        return next(iter(self.detectors.values())).conf

    @property
    def class_names(self) -> dict[str, dict[int, str]]:
        return {name: det.names for name, det in self.detectors.items()}

    def all_class_names(self) -> set[str]:
        return {n for det in self.detectors.values() for n in det.names.values()}

    def predict(self, image: ImageInput, offset: tuple[float, float] = (0, 0), rgb: bool = False) -> list[Detection]:
        return self.predict_batch([image], [offset], rgb=rgb)[0]

    def predict_batch(
        self,
        images: Sequence[ImageInput],
        offsets: Sequence[tuple[float, float]] | None = None,
        rgb: bool = False,
    ) -> list[list[Detection]]:
        merged: list[list[Detection]] = [[] for _ in images]
        for detector in self.detectors.values():
            for i, detections in enumerate(detector.predict_batch(images, offsets, rgb=rgb)):
                merged[i].extend(detections)
        return merged
