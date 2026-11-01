"""Merging detections from overlapping tiles.

An object in the overlap is found in two tiles, and at the tile edge it is also truncated:
the IoU between the full and the truncated box is low, so plain NMS does not remove the duplicate.
We therefore compare IoS (intersection area / area of the smaller polygon), and only between
detections from different tiles (within a tile the model has already applied NMS).
On conflict, the detection that does not touch an inner edge of its tile wins, then the more
confident one.
"""
from __future__ import annotations

from shapely import STRtree
from shapely.geometry import Polygon

from .schema import Detection

Window = tuple[int, int, int, int]  # x0, y0, w, h in scene pixels


def touches_inner_edge(
    det: Detection, window: Window, scene_size: tuple[int, int], margin: float = 4.0
) -> bool:
    """Whether the box touches a tile edge beyond which the scene continues (the object may be cut off)."""
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
    """tiles -- [(tile window, detections in scene pixels)], scene_size -- (width, height).

    agnostic_models -- models whose class is ignored when merging: for type models one object
    cannot be two types at once (C-17 from one tile, C-5 from the neighbouring one)."""
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

    polys = [Polygon(d.polygon).buffer(0) for d in dets]  # buffer(0) repairs self-intersections
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
    """A refining model (e.g. military aircraft types) knows only its own classes, and outside its
    domain it "recognises" objects that are not there: in port cranes and containers. So its
    detections are kept only where the general model found the same object."""
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
    """One object found by different models (general DOTA says "plane", the specialised one "SU-34")
    is kept once: the model with the higher priority wins, ties are broken by confidence.
    Classes are not compared: the models have different class sets, there is nothing to match them by."""
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
