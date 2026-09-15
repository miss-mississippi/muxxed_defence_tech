"""Детекция на целой сцене: GeoTIFF/JPG/PNG → тайлы → YOLO-OBB → склейка → GeoJSON (WGS84)."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from pyproj import Geod, Transformer
from rasterio.enums import ColorInterp, Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from .merge import merge_tiles
from .model import Detector
from .schema import Detection

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

GEOD = Geod(ellps="WGS84")
# веса DOTA обучены на снимках ~0.1–1 м/px; на более грубых самолёты и суда теряются
MAX_RECOMMENDED_GSD_M = 2.0


@dataclass
class SceneResult:
    path: str
    width: int
    height: int
    crs: str | None
    georeferenced: bool
    tiles_total: int
    tiles_skipped: int
    detections: list[Detection]
    features: list[dict] = field(default_factory=list)
    bounds_wgs84: list[float] | None = None  # [west, south, east, north]
    gsd_m: float | None = None
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def summary(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "georeferenced": self.georeferenced,
            "bounds_wgs84": self.bounds_wgs84,
            "gsd_m": self.gsd_m,
            "tiles_total": self.tiles_total,
            "tiles_skipped": self.tiles_skipped,
            "elapsed_s": self.elapsed_s,
            "detections": len(self.detections),
            "counts": self.counts(),
            "warnings": self.warnings,
        }

    def to_geojson(self) -> dict:
        return {"type": "FeatureCollection", "scene": self.summary(), "features": self.features}


def tile_windows(width: int, height: int, tile: int = 1024, overlap: int = 200) -> list[tuple[int, int, int, int]]:
    """Окна (x0, y0, w, h), покрывающие сцену. Последний ряд/столбец прижат к краю сцены,
    поэтому все тайлы полного размера и модель их не растягивает."""
    if not 0 <= overlap < tile:
        raise ValueError("нужно 0 <= overlap < tile")

    def starts(size: int) -> list[int]:
        if size <= tile:
            return [0]
        return list(range(0, size - tile, tile - overlap)) + [size - tile]

    return [(x, y, min(tile, width), min(tile, height)) for y in starts(height) for x in starts(width)]


def open_raster(path: str | Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        return rasterio.open(path)


def is_georeferenced(src) -> bool:
    return src.crs is not None and not src.transform.is_identity


def rgb_bands(src) -> list[int]:
    """Номера каналов R, G, B (1-based) по цветовой интерпретации, иначе первые три / один."""
    idx = {ci: i + 1 for i, ci in enumerate(src.colorinterp)}
    if all(c in idx for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
        return [idx[ColorInterp.red], idx[ColorInterp.green], idx[ColorInterp.blue]]
    return [1, 2, 3] if src.count >= 3 else [1]


def _read_scaled(ds, bands: list[int], max_size: int) -> np.ndarray:
    scale = max(ds.width, ds.height) / max_size
    if scale <= 1:
        return ds.read(bands)
    shape = (len(bands), max(1, round(ds.height / scale)), max(1, round(ds.width / scale)))
    return ds.read(bands, out_shape=shape, resampling=Resampling.bilinear)


def stretch_limits(src, bands: list[int], sample_size: int = 2048) -> tuple[np.ndarray, np.ndarray] | None:
    """Растяжка 2–98 перцентиль по уменьшенной копии всей сцены — одна на все тайлы,
    иначе контраст скачет от тайла к тайлу. uint8-снимки не трогаем."""
    if all(src.dtypes[b - 1] == "uint8" for b in bands):
        return None
    data = _read_scaled(src, bands, sample_size).astype(np.float32)
    lo = np.zeros(len(bands), np.float32)
    hi = np.ones(len(bands), np.float32)
    for k, band in enumerate(data):
        valid = band[band > 0]
        if valid.size:
            lo[k], hi[k] = np.percentile(valid, (2, 98))
    return lo, hi


def to_uint8(data: np.ndarray, limits: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """(C, H, W) из rasterio → HxWx3 uint8 RGB."""
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    img = np.transpose(data, (1, 2, 0))
    if limits is None:
        return np.ascontiguousarray(img)
    lo, hi = (np.resize(v, 3) for v in limits)
    scaled = (img.astype(np.float32) - lo) / np.maximum(hi - lo, 1e-6) * 255
    return np.clip(scaled, 0, 255).astype(np.uint8)


def detect_scene(
    detector: Detector,
    path: str | Path,
    tile: int = 1024,
    overlap: int = 200,
    batch: int = 4,
    ios_threshold: float = 0.6,
    bands: list[int] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SceneResult:
    t0 = time.perf_counter()
    with open_raster(path) as src:
        georef = is_georeferenced(src)
        bands = bands or rgb_bands(src)
        limits = stretch_limits(src, bands)
        windows = tile_windows(src.width, src.height, tile, overlap)

        tiles: list[tuple[tuple[int, int, int, int], list[Detection]]] = []
        skipped = 0
        for start in range(0, len(windows), batch):
            images, kept = [], []
            for x, y, w, h in windows[start : start + batch]:
                window = Window(x, y, w, h)
                raw = src.read(bands, window=window)
                # пустоту считаем по маске и сырым данным: тёмная вода после растяжки тоже 0, но это не пусто
                valid = (src.dataset_mask(window=window) > 0) & (raw.max(axis=0) > 0)
                if valid.mean() < 0.05:
                    skipped += 1
                    continue
                images.append(to_uint8(raw, limits))
                kept.append((x, y, w, h))
            if images:
                results = detector.predict_batch(images, [(x, y) for x, y, _, _ in kept], rgb=True)
                tiles.extend(zip(kept, results))
            if progress:
                progress(min(start + batch, len(windows)), len(windows))

        detections = merge_tiles(tiles, (src.width, src.height), ios_threshold)
        detections.sort(key=lambda d: -d.confidence)
        result = SceneResult(
            path=str(path),
            width=src.width,
            height=src.height,
            crs=src.crs.to_string() if src.crs else None,
            georeferenced=georef,
            tiles_total=len(windows),
            tiles_skipped=skipped,
            detections=detections,
        )

        to_wgs = None
        if georef:
            to_wgs = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            result.bounds_wgs84 = [round(v, 7) for v in transform_bounds(src.crs, "EPSG:4326", *src.bounds)]
            result.gsd_m = round(_gsd_m(src, to_wgs), 3)
            if result.gsd_m > MAX_RECOMMENDED_GSD_M:
                result.warnings.append(
                    f"разрешение {result.gsd_m} м/px грубее {MAX_RECOMMENDED_GSD_M} м/px — "
                    "модель рассчитана на детальные снимки, возможны пропуски"
                )
        else:
            result.warnings.append("снимок без геопривязки — координаты только в пикселях")
        result.features = [_feature(i, d, src.transform, to_wgs) for i, d in enumerate(detections)]

    result.elapsed_s = round(time.perf_counter() - t0, 2)
    return result


def _gsd_m(src, to_wgs: Transformer) -> float:
    cx, cy = src.width / 2, src.height / 2
    pts = [src.transform @ p for p in ((cx, cy), (cx + 1, cy), (cx, cy + 1))]
    lons, lats = to_wgs.transform([p[0] for p in pts], [p[1] for p in pts])
    dx = GEOD.inv(lons[0], lats[0], lons[1], lats[1])[2]
    dy = GEOD.inv(lons[0], lats[0], lons[2], lats[2])[2]
    return (dx + dy) / 2


def _feature(idx: int, det: Detection, transform, to_wgs: Transformer | None) -> dict:
    props = det.to_dict()
    props["polygon_px"] = props.pop("polygon")
    props["id"] = idx
    geometry = None
    if to_wgs is not None:
        px = np.asarray(det.polygon)
        lon, lat = to_wgs.transform(*(transform @ (px[:, 0], px[:, 1])))
        ring = orient(Polygon(list(zip(lon, lat))), sign=1.0).exterior.coords
        geometry = {"type": "Polygon", "coordinates": [[[round(x, 7), round(y, 7)] for x, y in ring]]}
        clon, clat = to_wgs.transform(*(transform @ (det.cx, det.cy)))
        az01, _, len01 = GEOD.inv(lon[0], lat[0], lon[1], lat[1])
        az12, _, len12 = GEOD.inv(lon[1], lat[1], lon[2], lat[2])
        long_az = az01 if len01 >= len12 else az12
        props.update(
            center_lon=round(float(clon), 7),
            center_lat=round(float(clat), 7),
            length_m=round(max(len01, len12), 1),
            width_m=round(min(len01, len12), 1),
            orientation_deg=round(long_az % 180, 1),  # азимут длинной оси, 0–180
        )
    return {"type": "Feature", "id": idx, "geometry": geometry, "properties": props}


def render_preview(
    path: str | Path, max_size: int = 2048, bands: list[int] | None = None
) -> tuple[np.ndarray, list[list[float]] | None]:
    """Превью для карты: RGBA uint8 и границы Leaflet [[south, west], [north, east]].
    Геопривязанная сцена перепроецируется в EPSG:3857 — так она точно ложится на подложку.
    Без геопривязки границы None (карта работает в пикселях)."""
    with open_raster(path) as src:
        bands = bands or rgb_bands(src)
        limits = stretch_limits(src, bands)
        if is_georeferenced(src):
            # без nodata и альфа-канала GDAL не знает, где кончается снимок, — просим добавить альфу
            add_alpha = src.nodata is None and ColorInterp.alpha not in src.colorinterp
            with WarpedVRT(src, crs="EPSG:3857", resampling=Resampling.bilinear, add_alpha=add_alpha) as vrt:
                data = _read_scaled(vrt, bands, max_size)
                # прозрачность — по маске валидных данных (за контуром снимка), а не по чёрным пикселям
                alpha = vrt.dataset_mask(out_shape=data.shape[1:])
                west, south, east, north = transform_bounds("EPSG:3857", "EPSG:4326", *vrt.bounds)
            bounds = [[south, west], [north, east]]
        else:
            data, bounds = _read_scaled(src, bands, max_size), None
            alpha = src.dataset_mask(out_shape=data.shape[1:])
    return np.dstack([to_uint8(data, limits), alpha.astype(np.uint8)]), bounds
