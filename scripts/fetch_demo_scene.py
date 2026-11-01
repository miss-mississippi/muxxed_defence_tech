"""Cut a demo scene out of a cloud-optimised GeoTIFF (COG) without downloading the whole tile.

    python scripts/fetch_demo_scene.py durban_port_tanks
    python scripts/fetch_demo_scene.py --url <visual.tif> --lon 31.03 --lat -29.87 --size 4096 --out data/demo/x.tif

Only the needed bytes are read via HTTP range requests. The output is a GeoTIFF in the source UTM projection.
Maxar Open Data imagery is CC BY-NC 4.0 (© Maxar Technologies) and is not committed to git.
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

# presets: window centre (lon, lat) and size in pixels
DURBAN_PORT = "https://maxar-opendata.s3.amazonaws.com/events/southafrica-flooding22/ard/36/213131031203/2022-04-20/10400100770EF000-visual.tif"
DURBAN_AIRPORT = "https://maxar-opendata.s3.amazonaws.com/events/southafrica-flooding22/ard/36/213131013233/2022-04-20/10400100770EF000-visual.tif"
# Sentinel-2 L2A stores each band as a separate COG; R, G, B are B04, B03, B02 at 10 m/px.
# Scene over Atyrau: S2B, 2026-09-14, 0.0% cloud cover, tile 39TWN (EPSG:32639).
S2_ATYRAU = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/39/T/WN/2026/9/S2B_39TWN_20260914_1_L2A"
ATTRIBUTION_S2 = "Copernicus Sentinel data 2026, processed by ESA; AWS Open Data (sentinel-cogs)"

PRESETS = {
    # Durban, South Africa, WorldView 2022-04-20, 0.305 m/px: container terminal and ships at berth
    "durban_port_terminal": {"url": DURBAN_PORT, "lon": 31.01872, "lat": -29.88028, "size": 4096},
    # oil terminal with storage tanks and ships in the basin
    "durban_port_tanks": {"url": DURBAN_PORT, "lon": 31.02752, "lat": -29.89060, "size": 4096},
    # King Shaka airport: terminal and apron with aircraft
    "durban_airport": {"url": DURBAN_AIRPORT, "lon": 31.10619, "lat": -29.61988, "size": 4096},
    # Atyrau, Kazakhstan, 10 m/px: measuring the resolution limit on Kazakhstan territory.
    # 2048 px at 10 m is 20.5 km: the city, the port on the Ural river and oil infrastructure.
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
        # the profile is set explicitly: the source COG is JPEG-compressed in YCbCr, which cannot be written with deflate
        profile = {
            "driver": "GTiff", "width": window.width, "height": window.height, "count": src.count,
            "dtype": src.dtypes[0], "crs": src.crs, "transform": src.window_transform(window),
            "nodata": src.nodata, "photometric": "RGB" if src.count >= 3 else "MINISBLACK",
            "compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512,
        }
        colorinterp = src.colorinterp
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)  # GDAL cannot overwrite a partial file left by an interrupted run
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data)
        dst.colorinterp = colorinterp
        dst.update_tags(SOURCE=url, ATTRIBUTION=ATTRIBUTION)
    return out


def fetch_bands(urls: list[str], lon: float, lat: float, size: int, out: Path, attribution: str) -> Path:
    """Assemble an RGB scene from separate single-band COGs.

    Sentinel-2 stores every band in its own file, so the same window is read from several sources
    and stacked into one GeoTIFF. The 10 m bands share one grid, so the window is computed once
    from the first file.
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
    p.add_argument("preset", nargs="?", choices=sorted(PRESETS), help="ready-made scene")
    p.add_argument("--url")
    p.add_argument("--lon", type=float)
    p.add_argument("--lat", type=float)
    p.add_argument("--size", type=int, default=4096, help="window side, px")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    if args.preset:
        cfg = PRESETS[args.preset] | {k: v for k, v in (("url", args.url), ("lon", args.lon), ("lat", args.lat)) if v is not None}
        out = args.out or ROOT / "data" / "demo" / f"{args.preset}.tif"
    elif args.url and args.lon is not None and args.lat is not None:
        cfg, out = {"url": args.url, "lon": args.lon, "lat": args.lat, "size": args.size}, args.out
        if out is None:
            sys.exit("--out is required without a preset")
    else:
        sys.exit("specify a preset or --url --lon --lat")
    size = args.size if args.size != 4096 or not args.preset else cfg["size"]

    attribution = cfg.get("attribution", ATTRIBUTION)
    if cfg.get("urls"):  # Sentinel-2: bands are separate files
        path = fetch_bands(cfg["urls"], cfg["lon"], cfg["lat"], size, out, attribution)
    else:
        path = fetch(cfg["url"], cfg["lon"], cfg["lat"], size, out)
    with rasterio.open(path) as ds:
        print(f"{path}: {ds.width}x{ds.height}, {ds.crs}, {ds.res[0]:.2f} m/px, {path.stat().st_size / 1e6:.1f} MB")
    print(f"attribution: {attribution}")


if __name__ == "__main__":
    main()
