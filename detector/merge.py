"""Склейка детекций с перекрывающихся тайлов.

Объект на перекрытии находится в двух тайлах, а у края тайла ещё и обрезан:
IoU полного и обрезанного бокса низкий, и обычный NMS такой дубль не убирает.
Поэтому сравниваем IoS — площадь пересечения / площадь меньшего полигона —
и только между детекциями из разных тайлов (внутри тайла NMS уже сделала модель).
При конфликте выигрывает детекция, не касающаяся внутренней границы своего тайла,
затем — более уверенная.
"""
from __future__ import annotations

from shapely import STRtree
from shapely.geometry import Polygon

from .schema import Detection

Window = tuple[int, int, int, int]  # x0, y0, w, h в пикселях сцены


def touches_inner_edge(
    det: Detection, window: Window, scene_size: tuple[int, int], margin: float = 4.0
) -> bool:
    """Касается ли бокс границы тайла, за которой сцена продолжается (объект мог быть обрезан)."""
    x0, y0, w, h = window
    width, height = scene_size
    xs = [p[0] for p in det.polygon]
    ys = [p[1] for p in det.polygon]
    return (
        (x0 > 0 and min(xs) <= x0 + margin)
        or (y0 > 0 and min(ys) <= y0 + margin)
        or (x0 + w < width and max(xs) >= x0 + w - margin)
        or (y0 + h < height and max(ys) >= y0 + h - margin)
    )


def merge_tiles(
    tiles: list[tuple[Window, list[Detection]]],
    scene_size: tuple[int, int],
    ios_threshold: float = 0.6,
    edge_margin: float = 4.0,
    agnostic_models: set[str] | None = None,
) -> list[Detection]:
    """tiles — [(окно тайла, детекции в пикселях сцены)], scene_size — (width, height).

    agnostic_models — модели, у которых при склейке класс не сравнивается: для моделей типов
    техники один объект не может быть двух типов сразу (C-17 из одного тайла, C-5 из соседнего)."""
    dets: list[Detection] = []
    tile_ids: list[int] = []
    at_edge: list[bool] = []
    for t, (window, tile_dets) in enumerate(tiles):
        for d in tile_dets:
            dets.append(d)
            tile_ids.append(t)
            at_edge.append(touches_inner_edge(d, window, scene_size, edge_margin))
    if len(tiles) < 2 or len(dets) < 2:
        return dets

    polys = [Polygon(d.polygon).buffer(0) for d in dets]  # buffer(0) чинит самопересечения
    tree = STRtree(polys)
    order = sorted(range(len(dets)), key=lambda i: (at_edge[i], -dets[i].confidence))
    suppressed = [False] * len(dets)
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in tree.query(polys[i]):
            j = int(j)
            agnostic = bool(agnostic_models) and dets[i].model in agnostic_models
            same_object = dets[j].model == dets[i].model and (agnostic or dets[j].class_id == dets[i].class_id)
            if j == i or suppressed[j] or tile_ids[j] == tile_ids[i] or not same_object:
                continue
            smaller = min(polys[i].area, polys[j].area)
            if smaller > 0 and polys[i].intersection(polys[j]).area / smaller >= ios_threshold:
                suppressed[j] = True
    return [dets[i] for i in sorted(keep)]


def apply_refinement(
    detections: list[Detection], refine_only: set[str], ios_threshold: float = 0.6
) -> list[Detection]:
    """Уточняющая модель (например, типы военных самолётов) знает только свои классы и вне своей
    области «узнаёт» объекты там, где их нет: на кранах и контейнерах порта. Поэтому её детекции
    оставляем лишь там, где тот же объект нашла общая модель."""
    if not refine_only:
        return detections
    base = [d for d in detections if d.model not in refine_only]
    if not base:
        return base
    base_polys = [Polygon(d.polygon).buffer(0) for d in base]
    tree = STRtree(base_polys)
    kept = []
    for det in detections:
        if det.model not in refine_only:
            kept.append(det)
            continue
        poly = Polygon(det.polygon).buffer(0)
        for j in tree.query(poly):
            other = base_polys[int(j)]
            smaller = min(poly.area, other.area)
            if smaller > 0 and poly.intersection(other).area / smaller >= ios_threshold:
                kept.append(det)
                break
    return kept


def suppress_cross_model(
    detections: list[Detection], priorities: dict[str, int], ios_threshold: float = 0.6
) -> list[Detection]:
    """Один объект, найденный разными моделями (общая DOTA — «plane», специализированная — «SU-34»),
    оставляем один раз: побеждает модель с большим приоритетом, при равном — уверенность.
    Классы не сравниваем: у моделей разные наборы классов, сопоставлять их нечем."""
    if len(detections) < 2 or len(priorities) < 2:
        return detections
    polys = [Polygon(d.polygon).buffer(0) for d in detections]
    tree = STRtree(polys)
    order = sorted(
        range(len(detections)),
        key=lambda i: (-priorities.get(detections[i].model, 0), -detections[i].confidence),
    )
    suppressed = [False] * len(detections)
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in tree.query(polys[i]):
            j = int(j)
            if j == i or suppressed[j] or detections[j].model == detections[i].model:
                continue
            smaller = min(polys[i].area, polys[j].area)
            if smaller > 0 and polys[i].intersection(polys[j]).area / smaller >= ios_threshold:
                suppressed[j] = True
    return [detections[i] for i in sorted(keep)]
