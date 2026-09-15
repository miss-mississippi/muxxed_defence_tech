"""REST API: загрузка снимков, каталог обнаружений, экспертная проверка, отчёт, карта.

    .venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000
    # карта: http://127.0.0.1:8000   документация API: http://127.0.0.1:8000/docs

Переменные окружения: DATA_DIR (каталог данных, по умолчанию ./data),
WEIGHTS (путь к весам), DEVICE (cpu | mps | 0), CONF (порог уверенности).
"""
from __future__ import annotations

import html
import json
import os
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from detector import DEFAULT_WEIGHTS, Detector, detect_scene, render_preview

from . import db

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
ALLOWED_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
ReviewStatus = Literal["pending", "confirmed", "rejected"]


class Review(BaseModel):
    status: ReviewStatus
    class_name: str | None = None  # исправленный экспертом класс
    comment: str | None = None


def create_app(
    data_dir: Path | None = None,
    weights: Path | None = None,
    device: str | None = None,
    conf: float | None = None,
) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("DATA_DIR", ROOT / "data"))
    weights = Path(weights or os.environ.get("WEIGHTS", DEFAULT_WEIGHTS))
    device = device or os.environ.get("DEVICE") or None
    conf = conf if conf is not None else float(os.environ.get("CONF", 0.25))
    scenes_dir, previews_dir, db_path = data_dir / "scenes", data_dir / "previews", data_dir / "catalog.sqlite"
    detect_lock = threading.Lock()  # одна модель на процесс — сцены обрабатываются по очереди

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        scenes_dir.mkdir(parents=True, exist_ok=True)
        previews_dir.mkdir(parents=True, exist_ok=True)
        db.init(db_path)
        app.state.detector = Detector(weights, device=device, conf=conf)
        yield

    app = FastAPI(
        title="Идентификация объектов на космических снимках",
        description="YOLO11-OBB: детекция, каталог обнаружений, экспертная проверка. NU STeP x Defense Tech Challenge.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def detector() -> Detector:
        return app.state.detector

    def get_scene_row(conn, scene_id: int):
        row = conn.execute("SELECT * FROM scenes WHERE id = ?", (scene_id,)).fetchone()
        if row is None:
            raise HTTPException(404, f"сцена {scene_id} не найдена")
        return row

    def save_upload(file: UploadFile, directory: Path) -> Path:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(415, f"поддерживаются {sorted(ALLOWED_SUFFIXES)}")
        path = directory / f"{uuid.uuid4().hex}{suffix}"
        with path.open("wb") as out:
            while chunk := file.file.read(1 << 20):
                out.write(chunk)
        return path

    def process_scene(scene_id: int, path: Path) -> None:
        def progress(done: int, total: int) -> None:
            with db.session(db_path) as conn:
                conn.execute("UPDATE scenes SET progress = ? WHERE id = ?", (done / total, scene_id))

        try:
            with db.session(db_path) as conn:
                conn.execute("UPDATE scenes SET status = 'processing' WHERE id = ?", (scene_id,))
            with detect_lock:
                result = detect_scene(detector(), path, progress=progress)
            rgba, preview_bounds = render_preview(path)
            cv2.imwrite(str(previews_dir / f"{scene_id}.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

            rows = []
            for f in result.features:
                p = f["properties"]
                rows.append((
                    scene_id, p["class_id"], p["class_name"], p["confidence"], json.dumps(p["polygon_px"]),
                    json.dumps(f["geometry"]) if f["geometry"] else None,
                    p["cx"], p["cy"], p["w"], p["h"], p["angle"],
                    p.get("center_lon"), p.get("center_lat"), p.get("length_m"), p.get("width_m"), p.get("orientation_deg"),
                ))
            with db.session(db_path) as conn:
                conn.executemany(
                    """INSERT INTO detections (scene_id, class_id, class_name, confidence, polygon_px, geometry,
                           cx, cy, w, h, angle, center_lon, center_lat, length_m, width_m, orientation_deg)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                conn.execute(
                    """UPDATE scenes SET status = 'done', progress = 1, width = ?, height = ?, crs = ?,
                           georeferenced = ?, bounds_wgs84 = ?, preview_bounds = ?, gsd_m = ?, tiles_total = ?,
                           tiles_skipped = ?, elapsed_s = ?, model = ?, warnings = ? WHERE id = ?""",
                    (
                        result.width, result.height, result.crs, int(result.georeferenced),
                        json.dumps(result.bounds_wgs84) if result.bounds_wgs84 else None,
                        json.dumps(preview_bounds) if preview_bounds else None,
                        result.gsd_m, result.tiles_total, result.tiles_skipped, result.elapsed_s,
                        detector().weights.name, json.dumps(result.warnings, ensure_ascii=False), scene_id,
                    ),
                )
        except Exception as exc:  # сцена помечается failed, сервер продолжает работать
            with db.session(db_path) as conn:
                conn.execute("UPDATE scenes SET status = 'failed', error = ? WHERE id = ?", (repr(exc), scene_id))

    # ---------- служебное ----------

    @app.get("/api/health", tags=["служебное"])
    def health() -> dict:
        d = detector()
        return {"status": "ok", "weights": d.weights.name, "device": d.device, "classes": len(d.names)}

    @app.get("/api/model", tags=["служебное"])
    def model_info() -> dict:
        d = detector()
        return {
            "weights": d.weights.name,
            "architecture": "YOLO11-OBB (Ultralytics, AGPL-3.0)",
            "device": d.device,
            "imgsz": d.imgsz,
            "conf": d.conf,
            "classes": {int(k): v for k, v in d.names.items()},
        }

    # ---------- сцены ----------

    @app.post("/api/scenes", status_code=202, tags=["сцены"])
    def upload_scene(background: BackgroundTasks, file: UploadFile = File(...)) -> dict:
        """Загрузить снимок (GeoTIFF/JPG/PNG). Обработка идёт в фоне — статус в GET /api/scenes/{id}."""
        path = save_upload(file, scenes_dir)
        with db.session(db_path) as conn:
            cur = conn.execute("INSERT INTO scenes (filename, path) VALUES (?, ?)", (file.filename, str(path)))
            scene_id = cur.lastrowid
            row = get_scene_row(conn, scene_id)
        background.add_task(process_scene, scene_id, path)
        return db.scene_dict(row)

    @app.get("/api/scenes", tags=["сцены"])
    def list_scenes() -> list[dict]:
        with db.session(db_path) as conn:
            rows = conn.execute("SELECT * FROM scenes ORDER BY id DESC").fetchall()
            counts = conn.execute(
                "SELECT scene_id, review_status, COUNT(*) AS n FROM detections GROUP BY scene_id, review_status"
            ).fetchall()
        by_scene: dict[int, dict] = {}
        for c in counts:
            by_scene.setdefault(c["scene_id"], {})[c["review_status"]] = c["n"]
        return [db.scene_dict(r) | {"review": by_scene.get(r["id"], {})} for r in rows]

    @app.get("/api/scenes/{scene_id}", tags=["сцены"])
    def get_scene(scene_id: int) -> dict:
        with db.session(db_path) as conn:
            row = get_scene_row(conn, scene_id)
            classes = conn.execute(
                "SELECT COALESCE(review_class, class_name) AS name, COUNT(*) AS n FROM detections "
                "WHERE scene_id = ? AND review_status != 'rejected' GROUP BY name ORDER BY n DESC",
                (scene_id,),
            ).fetchall()
            review = conn.execute(
                "SELECT review_status, COUNT(*) AS n FROM detections WHERE scene_id = ? GROUP BY review_status",
                (scene_id,),
            ).fetchall()
        return db.scene_dict(row) | {
            "counts": {c["name"]: c["n"] for c in classes},
            "review": {r["review_status"]: r["n"] for r in review},
        }

    @app.get("/api/scenes/{scene_id}/preview.png", tags=["сцены"])
    def scene_preview(scene_id: int) -> FileResponse:
        path = previews_dir / f"{scene_id}.png"
        if not path.exists():
            raise HTTPException(404, "превью ещё не готово")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/scenes/{scene_id}/report", response_class=HTMLResponse, tags=["сцены"])
    def scene_report(scene_id: int) -> str:
        """Отчёт по сцене (HTML, печатается в PDF из браузера)."""
        scene = get_scene(scene_id)
        with db.session(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM detections WHERE scene_id = ? AND review_status != 'rejected' ORDER BY confidence DESC",
                (scene_id,),
            ).fetchall()
        return render_report(scene, rows)

    # ---------- обнаружения ----------

    @app.get("/api/detections", tags=["обнаружения"])
    def list_detections(
        scene_id: int | None = None,
        class_name: list[str] | None = Query(None, description="можно несколько"),
        min_conf: float = Query(0.0, ge=0, le=1),
        status: list[ReviewStatus] | None = Query(None),
        bbox: str | None = Query(None, description="west,south,east,north (WGS84)"),
        limit: int = Query(5000, ge=1, le=50000),
        offset: int = Query(0, ge=0),
    ) -> dict:
        """Каталог обнаружений в виде GeoJSON FeatureCollection с фильтрами."""
        where, params = ["confidence >= ?"], [min_conf]
        if scene_id is not None:
            where.append("scene_id = ?")
            params.append(scene_id)
        if class_name:
            where.append(f"COALESCE(review_class, class_name) IN ({','.join('?' * len(class_name))})")
            params += class_name
        if status:
            where.append(f"review_status IN ({','.join('?' * len(status))})")
            params += status
        if bbox:
            try:
                west, south, east, north = (float(v) for v in bbox.split(","))
            except ValueError:
                raise HTTPException(422, "bbox: west,south,east,north")
            where.append("center_lon BETWEEN ? AND ? AND center_lat BETWEEN ? AND ?")
            params += [west, east, south, north]
        sql = f"SELECT * FROM detections WHERE {' AND '.join(where)} ORDER BY confidence DESC LIMIT ? OFFSET ?"
        with db.session(db_path) as conn:
            rows = conn.execute(sql, [*params, limit, offset]).fetchall()
        return {"type": "FeatureCollection", "features": [db.detection_feature(r) for r in rows]}

    @app.get("/api/detections/{detection_id}", tags=["обнаружения"])
    def get_detection(detection_id: int) -> dict:
        with db.session(db_path) as conn:
            row = conn.execute("SELECT * FROM detections WHERE id = ?", (detection_id,)).fetchone()
        if row is None:
            raise HTTPException(404, f"обнаружение {detection_id} не найдено")
        return db.detection_feature(row)

    @app.patch("/api/detections/{detection_id}", tags=["обнаружения"])
    def review_detection(detection_id: int, review: Review) -> dict:
        """Экспертная проверка: подтвердить / отклонить / исправить класс."""
        if review.class_name is not None and review.class_name not in detector().names.values():
            raise HTTPException(422, f"неизвестный класс {review.class_name}")
        reviewed_at = None if review.status == "pending" else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with db.session(db_path) as conn:
            cur = conn.execute(
                "UPDATE detections SET review_status = ?, review_class = ?, review_comment = ?, reviewed_at = ? WHERE id = ?",
                (review.status, review.class_name, review.comment, reviewed_at, detection_id),
            )
            if cur.rowcount == 0:
                raise HTTPException(404, f"обнаружение {detection_id} не найдено")
        return get_detection(detection_id)

    @app.get("/api/stats", tags=["обнаружения"])
    def stats() -> dict:
        with db.session(db_path) as conn:
            by_class = conn.execute(
                "SELECT COALESCE(review_class, class_name) AS name, COUNT(*) AS n FROM detections "
                "WHERE review_status != 'rejected' GROUP BY name ORDER BY n DESC"
            ).fetchall()
            by_status = conn.execute("SELECT review_status, COUNT(*) AS n FROM detections GROUP BY review_status").fetchall()
            n_scenes = conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
        return {
            "scenes": n_scenes,
            "by_class": {r["name"]: r["n"] for r in by_class},
            "by_status": {r["review_status"]: r["n"] for r in by_status},
        }

    # ---------- синхронная детекция для интеграции ----------

    @app.post("/api/detect", tags=["интеграция"])
    def detect_sync(file: UploadFile = File(...)) -> dict:
        """Снимок → GeoJSON сразу в ответе, без сохранения в каталог (для внешних систем)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = save_upload(file, Path(tmp))
            with detect_lock:
                result = detect_scene(detector(), path)
        geojson = result.to_geojson()
        geojson["scene"]["path"] = file.filename
        return geojson

    # ---------- карта ----------

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


def render_report(scene: dict, rows: list) -> str:
    e = html.escape
    status_ru = {"pending": "не проверено", "confirmed": "подтверждено", "rejected": "отклонено"}
    counts = "".join(f"<tr><td>{e(k)}</td><td>{v}</td></tr>" for k, v in scene["counts"].items())
    review = ", ".join(f"{status_ru.get(k, k)}: {v}" for k, v in scene["review"].items()) or "—"

    def row_html(r) -> str:
        center = "" if r["center_lat"] is None else f"{r['center_lat']:.6f}, {r['center_lon']:.6f}"
        size = "" if r["length_m"] is None else f"{r['length_m']:.1f} × {r['width_m']:.1f}"
        azimuth = "" if r["orientation_deg"] is None else f"{r['orientation_deg']:.0f}°"
        return (
            f"<tr><td>{r['id']}</td><td>{e(r['review_class'] or r['class_name'])}</td><td>{r['confidence']:.2f}</td>"
            f"<td>{center}</td><td>{size}</td><td>{azimuth}</td><td>{status_ru[r['review_status']]}</td></tr>"
        )

    table = "".join(row_html(r) for r in rows)
    bounds = scene["bounds_wgs84"]
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Отчёт — {e(scene['filename'])}</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:32px;color:#111}} h1{{font-size:20px;margin:0 0 4px}}
table{{border-collapse:collapse;margin:12px 0 24px}} td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
th{{background:#f2f2f2}} .muted{{color:#666}} img{{max-width:100%;border:1px solid #ccc}}
</style></head><body>
<h1>Отчёт об автоматизированной идентификации объектов</h1>
<div class="muted">Сформирован {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · результаты информационно-аналитические, подлежат экспертной проверке</div>
<h2>Снимок</h2>
<table>
<tr><th>Файл</th><td>{e(scene['filename'])}</td></tr>
<tr><th>Размер, px</th><td>{scene['width']} × {scene['height']}</td></tr>
<tr><th>Система координат</th><td>{e(scene['crs'] or 'нет геопривязки')}</td></tr>
<tr><th>Разрешение, м/px</th><td>{scene['gsd_m'] or '—'}</td></tr>
<tr><th>Границы (WGS84)</th><td>{'—' if not bounds else f'З {bounds[0]:.5f}, Ю {bounds[1]:.5f}, В {bounds[2]:.5f}, С {bounds[3]:.5f}'}</td></tr>
<tr><th>Модель</th><td>{e(scene['model'] or '')}</td></tr>
<tr><th>Время обработки, с</th><td>{scene['elapsed_s']}</td></tr>
<tr><th>Экспертная проверка</th><td>{review}</td></tr>
</table>
<h2>Объекты по классам (без отклонённых)</h2>
<table><tr><th>Класс</th><th>Количество</th></tr>{counts}</table>
<img src="/api/scenes/{scene['id']}/preview.png" alt="превью сцены">
<h2>Перечень обнаружений</h2>
<table><tr><th>ID</th><th>Класс</th><th>Уверенность</th><th>Центр (шир., долг.)</th><th>Размер, м</th><th>Азимут</th><th>Статус</th></tr>{table}</table>
</body></html>"""


app = create_app()
