import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from conftest import SAMPLE, UTM_CRS, UTM_ORIGIN, UTM_PIXEL_M
from detector import (
    DEFAULT_WEIGHTS,
    Detection,
    ModelRegistry,
    ModelSpec,
    apply_refinement,
    detect_scene,
    merge_tiles,
    render_preview,
    suppress_cross_model,
    tile_windows,
)


def box(x0, y0, x1, y1, cls=1, conf=0.8, model="") -> Detection:
    return Detection(
        class_id=cls, class_name="ship", confidence=conf,
        polygon=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        cx=(x0 + x1) / 2, cy=(y0 + y1) / 2, w=x1 - x0, h=y1 - y0, angle=0.0, model=model,
    )


@pytest.mark.parametrize("size", [(1920, 1080), (1024, 1024), (800, 600), (5000, 3001)])
def test_tile_windows_cover_scene_with_full_tiles(size):
    width, height = size
    covered = np.zeros((height, width), bool)
    for x, y, w, h in tile_windows(width, height, tile=1024, overlap=200):
        assert (w, h) == (min(1024, width), min(1024, height))
        assert x + w <= width and y + h <= height
        covered[y : y + h, x : x + w] = True
    assert covered.all()


def test_merge_prefers_full_box_over_truncated_duplicate():
    # тайлы [0..1024] и [824..1848] по x; объект 1000..1100 обрезан в левом тайле
    left = ((0, 0, 1024, 1024), [box(1000, 100, 1024, 140, conf=0.9)])
    right = ((824, 0, 1024, 1024), [box(1000, 100, 1100, 140, conf=0.7)])
    merged = merge_tiles([left, right], scene_size=(1848, 1024))
    assert len(merged) == 1
    assert merged[0].polygon[1][0] == 1100


def test_merge_keeps_overlaps_inside_one_tile_and_other_classes():
    tile = ((0, 0, 1024, 1024), [box(100, 100, 200, 140), box(110, 100, 210, 140), box(100, 100, 200, 140, cls=7)])
    other = ((824, 0, 1024, 1024), [])
    assert len(merge_tiles([tile, other], scene_size=(1848, 1024))) == 3


def test_refinement_keeps_types_only_where_general_model_sees_object():
    plane = box(100, 100, 200, 140, cls=0, conf=0.80, model="dota")
    typed = box(105, 100, 205, 140, cls=3, conf=0.70, model="mar20")   # тот же самолёт
    ghost = box(900, 900, 960, 950, cls=16, conf=0.71, model="mar20")  # кран в порту

    kept = apply_refinement([plane, typed, ghost], {"mar20"})
    assert kept == [plane, typed]
    final = suppress_cross_model(kept, {"dota": 0, "mar20": 1})
    assert [(d.model, d.class_id) for d in final] == [("mar20", 3)]


def test_cross_tile_merge_ignores_class_for_type_models():
    # один самолёт на перекрытии: из одного тайла C-17, из соседнего C-5
    left = ((0, 0, 1024, 1024), [box(900, 100, 1000, 140, cls=2, conf=0.77, model="mar20")])
    right = ((824, 0, 1024, 1024), [box(900, 100, 1000, 140, cls=3, conf=0.55, model="mar20")])
    scene = (1848, 1024)

    assert len(merge_tiles([left, right], scene)) == 2  # по умолчанию классы разные — не склеиваем
    merged = merge_tiles([left, right], scene, agnostic_models={"mar20"})
    assert len(merged) == 1 and merged[0].class_id == 2  # остаётся более уверенная


def test_predict_offset_and_rgb(detector, boats_rgb):
    base = detector.predict(boats_rgb, rgb=True)
    shifted = detector.predict(boats_rgb, offset=(5000, 12000), rgb=True)
    assert len(base) == len(shifted) > 0
    for b, s in zip(base, shifted):
        assert s.cx == pytest.approx(b.cx + 5000, abs=0.02)
        assert s.cy == pytest.approx(b.cy + 12000, abs=0.02)
    with pytest.raises(ValueError):
        detector.predict(boats_rgb.astype(np.uint16))


def test_scene_geojson_matches_affine(detector, utm_uint16_tif):
    result = detect_scene(detector, utm_uint16_tif)
    raw = detect_scene(detector, utm_uint16_tif, ios_threshold=1.01)  # без склейки
    assert result.georeferenced and result.tiles_total == 6
    assert 100 < len(result.detections) <= len(raw.detections)
    assert result.gsd_m == pytest.approx(UTM_PIXEL_M, rel=0.01)

    transform = from_origin(*UTM_ORIGIN, UTM_PIXEL_M, UTM_PIXEL_M)
    to_wgs = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)
    west, south, east, north = result.bounds_wgs84
    for feature in result.features[:20]:
        props = feature["properties"]
        ring = feature["geometry"]["coordinates"][0]
        assert len(ring) == 5 and ring[0] == ring[-1]
        expected = np.array([to_wgs.transform(*(transform @ tuple(p))) for p in props["polygon_px"]])
        for corner in ring[:4]:  # порядок углов меняет orient(), сравниваем с ближайшим; 2e-7° ≈ 2 см
            assert np.abs(expected - corner).max(axis=1).min() < 2e-7
        assert west <= props["center_lon"] <= east and south <= props["center_lat"] <= north
        assert props["length_m"] == pytest.approx(max(props["w"], props["h"]) * UTM_PIXEL_M, rel=0.02)


def test_two_models_tag_and_deduplicate(boats_rgb):
    # две «разные» модели из одних весов: каждый объект находится дважды
    registry = ModelRegistry([ModelSpec("general", DEFAULT_WEIGHTS, 0), ModelSpec("special", DEFAULT_WEIGHTS, 1)])
    raw = registry.predict(boats_rgb, rgb=True)
    assert {d.model for d in raw} == {"general", "special"}

    merged = suppress_cross_model(raw, registry.priorities)
    assert len(merged) == len(raw) // 2
    assert {d.model for d in merged} == {"special"}  # выигрывает приоритетная модель


def test_scene_without_georeference(detector):
    result = detect_scene(detector, SAMPLE)
    assert not result.georeferenced and result.bounds_wgs84 is None
    assert result.features and all(f["geometry"] is None for f in result.features)


def test_render_preview(utm_uint16_tif):
    rgba, bounds = render_preview(utm_uint16_tif, max_size=512)
    assert rgba.shape[2] == 4 and max(rgba.shape[:2]) <= 512 and rgba[..., 3].any()
    (south, west), (north, east) = bounds
    assert south < north and west < east
    rgba, bounds = render_preview(SAMPLE, max_size=512)
    assert bounds is None and rgba.shape[:2] == (288, 512) and (rgba[..., 3] == 255).all()


def test_dark_water_is_not_empty(detector, tmp_path, boats_rgb):
    data = np.zeros((3, 1024, 3072), np.uint16)  # x 0..1024 — нули (нет данных)
    data[:, :, 1024:2048] = 300  # тёмная вода: после растяжки 0, но это данные
    data[:, :, 2048:] = boats_rgb[:1024, :1024].transpose(2, 0, 1).astype(np.uint16) * 40 + 300
    path = tmp_path / "dark_water.tif"
    # 300 км от осевого меридиана UTM → снимок повёрнут в EPSG:3857, углы превью вне снимка
    with rasterio.open(
        path, "w", driver="GTiff", width=3072, height=1024, count=3, dtype="uint16",
        crs=UTM_CRS, transform=from_origin(300000.0, 5670000.0, 0.5, 0.5),
    ) as dst:
        dst.write(data)

    result = detect_scene(detector, path)
    assert result.tiles_total == 4 and result.tiles_skipped == 1  # пропущен только тайл из нулей

    rgba, _ = render_preview(path, max_size=512)
    h, w = rgba.shape[:2]
    assert rgba[h // 2, w // 2, 3] == 255  # тёмная вода непрозрачна
    assert rgba[0, 0, 3] == 0  # угол за контуром перепроецированного снимка прозрачен
