"""SQLite-каталог сцен и обнаружений."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS scenes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    path            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued',   -- queued | processing | done | failed
    progress        REAL NOT NULL DEFAULT 0,
    error           TEXT,
    width           INTEGER,
    height          INTEGER,
    crs             TEXT,
    georeferenced   INTEGER,
    bounds_wgs84    TEXT,                              -- JSON [west, south, east, north]
    preview_bounds  TEXT,                              -- JSON [[south, west], [north, east]]
    gsd_m           REAL,
    tiles_total     INTEGER,
    tiles_skipped   INTEGER,
    elapsed_s       REAL,
    model           TEXT,
    warnings        TEXT,                              -- JSON list
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id        INTEGER NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
    class_id        INTEGER NOT NULL,
    class_name      TEXT NOT NULL,
    model           TEXT,                              -- какая модель нашла объект
    confidence      REAL NOT NULL,
    polygon_px      TEXT NOT NULL,                     -- JSON [[x, y] x4], пиксели сцены
    geometry        TEXT,                              -- JSON GeoJSON Polygon (WGS84) или NULL
    cx REAL, cy REAL, w REAL, h REAL, angle REAL,
    center_lon      REAL,
    center_lat      REAL,
    length_m        REAL,
    width_m         REAL,
    orientation_deg REAL,
    review_status   TEXT NOT NULL DEFAULT 'pending',   -- pending | confirmed | rejected
    review_class    TEXT,                              -- класс, исправленный экспертом
    review_comment  TEXT,
    reviewed_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_detections_scene  ON detections(scene_id);
CREATE INDEX IF NOT EXISTS idx_detections_class  ON detections(class_name);
CREATE INDEX IF NOT EXISTS idx_detections_center ON detections(center_lon, center_lat);
"""

DETECTION_FIELDS = (
    "id", "scene_id", "class_id", "class_name", "model", "confidence", "cx", "cy", "w", "h", "angle",
    "center_lon", "center_lat", "length_m", "width_m", "orientation_deg",
    "review_status", "review_class", "review_comment", "reviewed_at",
)


def init(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with session(path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        try:  # каталог, созданный до появления нескольких моделей
            conn.execute("ALTER TABLE detections ADD COLUMN model TEXT")
        except sqlite3.OperationalError:
            pass


@contextmanager
def session(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def scene_dict(row: sqlite3.Row) -> dict:
    scene = dict(row)
    scene.pop("path", None)
    scene["georeferenced"] = bool(scene["georeferenced"]) if scene["georeferenced"] is not None else None
    for key in ("bounds_wgs84", "preview_bounds", "warnings"):
        scene[key] = json.loads(scene[key]) if scene[key] else None
    return scene


def detection_feature(row: sqlite3.Row) -> dict:
    props = {k: row[k] for k in DETECTION_FIELDS}
    props["polygon_px"] = json.loads(row["polygon_px"])
    geometry = json.loads(row["geometry"]) if row["geometry"] else None
    return {"type": "Feature", "id": row["id"], "geometry": geometry, "properties": props}
