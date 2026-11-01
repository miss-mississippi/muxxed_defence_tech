"""Detection format: the contract between the ML module and the geo service.

Coordinates are in pixels of the whole scene (not the tile): the origin is the top-left corner
of the top-left pixel, x grows to the right (columns), y grows down (rows). This is the same
convention as the rasterio affine transform, so converting to the image CRS is just:

    x_geo, y_geo = dataset.transform * (x, y)

If the image CRS is not EPSG:4326, reproject to WGS84 afterwards (pyproj).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    # 4 corners of the rotated box [[x, y], ...] in ultralytics order (xyxyxyxy)
    polygon: list[list[float]]
    # centre, width, height in pixels; angle in radians (as in ultralytics xywhr)
    cx: float
    cy: float
    w: float
    h: float
    angle: float
    # model name: the general DOTA model or the fine-tuned type model (see detector/registry.py)
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
