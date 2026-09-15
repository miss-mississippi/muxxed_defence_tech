"""Формат детекции — контракт между ML-модулем и геосервисом.

Координаты — в пикселях всей сцены (не тайла): начало в левом верхнем углу
левого верхнего пикселя, x вправо (столбцы), y вниз (строки). Это та же
конвенция, что у affine-трансформа rasterio, поэтому перевод в CRS снимка:

    x_geo, y_geo = dataset.transform * (x, y)

Если CRS снимка не EPSG:4326 — после этого перепроецировать в WGS84 (pyproj).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    # 4 угла повёрнутого бокса [[x, y], ...] в порядке ultralytics (xyxyxyxy)
    polygon: list[list[float]]
    # центр, ширина, высота — пиксели; угол — радианы (как xywhr в ultralytics)
    cx: float
    cy: float
    w: float
    h: float
    angle: float

    def to_dict(self) -> dict:
        return asdict(self)
