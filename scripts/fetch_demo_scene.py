"""Вырезать демо-сцену из облачного GeoTIFF (COG) Maxar Open Data без скачивания всего тайла.

    python scripts/fetch_demo_scene.py durban_port
    python scripts/fetch_demo_scene.py --url <visual.tif> --lon 31.03 --lat -29.87 --size 4096 --out data/demo/x.tif

Читаются только нужные байты по HTTP range. Результат — GeoTIFF в исходной UTM-проекции.
Снимки Maxar Open Data — CC BY-NC 4.0 (© Maxar Technologies), в git не кладём.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rasterio
from pyproj import Transformer
from rasterio.enums import ColorInterp
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION = "Maxar Open Data Program, CC BY-NC 4.0, (c) Maxar Technologies"

# центр окна (lon, lat) и размер в пикселях
DURBAN_PORT = "https://maxar-opendata.s3.amazonaws.com/events/southafrica-flooding22/ard/36/213131031203/2022-04-20/10400100770EF000-visual.tif"
DURBAN_AIRPORT = "https://maxar-opendata.s3.amazonaws.com/events/southafrica-flooding22/ard/36/213131013233/2022-04-20/10400100770EF000-visual.tif"
# Sentinel-2 L2A хранит каждую полосу отдельным COG; R, G, B — это B04, B03, B02 по 10 м/px.
# Снимок над Атырау: S2B, 14.09.2026, облачность 0.0%, тайл 39TWN (EPSG:32639).
S2_ATYRAU = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/39/T/WN/2026/9/S2B_39TWN_20260914_1_L2A"
ATTRIBUTION_S2 = "Copernicus Sentinel data 2026, processed by ESA; AWS Open Data (sentinel-cogs)"

PRESETS = {
    # Дурбан, ЮАР, WorldView 2022-04-20, 0.305 м/px: контейнерный терминал и суда у причалов
    "durban_port_terminal": {"url": DURBAN_PORT, "lon": 31.01872, "lat": -29.88028, "size": 4096},
    # нефтебаза с резервуарами и суда в бассейне
    "durban_port_tanks": {"url": DURBAN_PORT, "lon": 31.02752, "lat": -29.89060, "size": 4096},
    # аэропорт King Shaka: терминал, перрон с самолётами
    "durban_airport": {"url": DURBAN_AIRPORT, "lon": 31.10619, "lat": -29.61988, "size": 4096},
    # Атырау, Казахстан, 10 м/px: замер нижней границы разрешения на отечественной территории.
    # 2048 px при 10 м — это 20.5 км: город, порт на Урале и нефтяная инфраструктура.
    "atyrau_s2": {
        "urls": [f"{S2_ATYRAU}/B04.tif", f"{S2_ATYRAU}/B03.tif", f"{S2_ATYRAU}/B02.tif"],
        "lon": 51.8833, "lat": 47.1167, "size": 2048, "attribution": ATTRIBUTION_S2,
    },
}
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif", "GDAL_HTTP_MULTIRANGE": "YES"}


def fetch(url: str, lon: float, lat: float, size: int, out: Path) -> Path:
    with rasterio.Env(**GDAL_ENV), rasterio.open(url) as src:
        x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
        col, row = ~src.transform * (x, y)
        col0 = int(min(max(col - size / 2, 0), src.width - size))
        row0 = int(min(max(row - size / 2, 0), src.height - size))
        window = Window(col0, row0, min(size, src.width), min(size, src.height))
        data = src.read(window=window)
        # профиль задаём явно: исходный COG сжат JPEG в YCbCr, с deflate такое не пишется
        profile = {
            "driver": "GTiff", "width": window.width, "height": window.height, "count": src.count,
            "dtype": src.dtypes[0], "crs": src.crs, "transform": src.window_transform(window),
            "nodata": src.nodata, "photometric": "RGB" if src.count >= 3 else "MINISBLACK",
            "compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512,
        }
        colorinterp = src.colorinterp
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)  # недописанный файл от прерванного запуска GDAL перезаписать не может
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data)
        dst.colorinterp = colorinterp
        dst.update_tags(SOURCE=url, ATTRIBUTION=ATTRIBUTION)
    return out


def fetch_bands(urls: list[str], lon: float, lat: float, size: int, out: Path, attribution: str) -> Path:
    """Собрать RGB-сцену из отдельных однополосных COG.

    Sentinel-2 хранит каждую полосу в своём файле, поэтому одно и то же окно читается из
    нескольких источников и складывается в один GeoTIFF. Полосы 10 м лежат на общей сетке,
    так что окно считается один раз по первому файлу.
    """
    planes: list = []
    profile: dict | None = None
    window: Window | None = None
    with rasterio.Env(**GDAL_ENV):
        for url in urls:
            with rasterio.open(url) as src:
                if profile is None:
                    x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
                    col, row = ~src.transform @ (x, y)
                    col0 = int(min(max(col - size / 2, 0), src.width - size))
                    row0 = int(min(max(row - size / 2, 0), src.height - size))
                    window = Window(col0, row0, min(size, src.width), min(size, src.height))
                    profile = {
                        "driver": "GTiff", "width": window.width, "height": window.height, "count": len(urls),
                        "dtype": src.dtypes[0], "crs": src.crs, "transform": src.window_transform(window),
                        "nodata": src.nodata, "photometric": "RGB",
                        "compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512,
                    }
                planes.append(src.read(1, window=window))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        for i, plane in enumerate(planes, start=1):
            dst.write(plane, i)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue][: len(planes)]
        dst.update_tags(SOURCE=" ".join(urls), ATTRIBUTION=attribution)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("preset", nargs="?", choices=sorted(PRESETS), help="готовая сцена")
    p.add_argument("--url")
    p.add_argument("--lon", type=float)
    p.add_argument("--lat", type=float)
    p.add_argument("--size", type=int, default=4096, help="сторона окна, px")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    if args.preset:
        cfg = PRESETS[args.preset] | {k: v for k, v in (("url", args.url), ("lon", args.lon), ("lat", args.lat)) if v is not None}
        out = args.out or ROOT / "data" / "demo" / f"{args.preset}.tif"
    elif args.url and args.lon is not None and args.lat is not None:
        cfg, out = {"url": args.url, "lon": args.lon, "lat": args.lat, "size": args.size}, args.out
        if out is None:
            sys.exit("--out обязателен без пресета")
    else:
        sys.exit("укажите пресет или --url --lon --lat")
    size = args.size if args.size != 4096 or not args.preset else cfg["size"]

    attribution = cfg.get("attribution", ATTRIBUTION)
    if cfg.get("urls"):  # Sentinel-2: полосы лежат отдельными файлами
        path = fetch_bands(cfg["urls"], cfg["lon"], cfg["lat"], size, out, attribution)
    else:
        path = fetch(cfg["url"], cfg["lon"], cfg["lat"], size, out)
    with rasterio.open(path) as ds:
        print(f"{path}: {ds.width}x{ds.height}, {ds.crs}, {ds.res[0]:.2f} м/px, {path.stat().st_size / 1e6:.1f} МБ")
    print(f"атрибуция: {attribution}")


if __name__ == "__main__":
    main()
