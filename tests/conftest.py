import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector import Detector  # noqa: E402

SAMPLE = ROOT / "samples" / "boats.jpg"
# синтетическая геопривязка: UTM 42N (район Астаны), пиксель 0.1 м
UTM_CRS = "EPSG:32642"
UTM_ORIGIN = (500000.0, 5670000.0)
UTM_PIXEL_M = 0.1


@pytest.fixture(scope="session")
def detector() -> Detector:
    return Detector()


@pytest.fixture(scope="session")
def boats_rgb() -> np.ndarray:
    return cv2.imread(str(SAMPLE))[..., ::-1].copy()


@pytest.fixture(scope="session")
def utm_uint16_tif(tmp_path_factory, boats_rgb) -> Path:
    """boats.jpg как 16-битный GeoTIFF в UTM — проверяет растяжку, тайлинг и геопривязку."""
    path = tmp_path_factory.mktemp("scenes") / "boats_utm_u16.tif"
    h, w = boats_rgb.shape[:2]
    data = (boats_rgb.astype(np.uint16) * 40 + 300).transpose(2, 0, 1)
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint16",
        crs=UTM_CRS, transform=from_origin(*UTM_ORIGIN, UTM_PIXEL_M, UTM_PIXEL_M),
    ) as dst:
        dst.write(data)
    return path
