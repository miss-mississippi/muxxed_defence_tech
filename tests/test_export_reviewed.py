import importlib.util
import json

import cv2

from api import db
from conftest import ROOT

spec = importlib.util.spec_from_file_location("export_reviewed", ROOT / "scripts" / "export_reviewed.py")
export_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_module)


def add_detection(conn, scene_id, x0, y0, x1, y1, status, class_name="ship", review_class=None):
    polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    conn.execute(
        "INSERT INTO detections (scene_id, class_id, class_name, confidence, polygon_px, review_status, review_class) "
        "VALUES (?, 1, ?, 0.9, ?, ?, ?)",
        (scene_id, class_name, json.dumps(polygon), status, review_class),
    )


def test_export_only_fully_reviewed_tiles(tmp_path, utm_uint16_tif):
    db_path = tmp_path / "catalog.sqlite"
    db.init(db_path)
    with db.session(db_path) as conn:
        scene_id = conn.execute(
            "INSERT INTO scenes (filename, path, status, width, height) VALUES ('m.tif', ?, 'done', 1920, 1080)",
            (str(utm_uint16_tif),),
        ).lastrowid
        # сцена 1920x1080 → тайлы x0 ∈ {0, 824, 896}, y0 ∈ {0, 56}
        add_detection(conn, scene_id, 100, 100, 200, 140, "confirmed", review_class="harbor")  # тайлы (0,0), (0,56)
        add_detection(conn, scene_id, 300, 500, 360, 530, "rejected")                          # те же тайлы → фон
        add_detection(conn, scene_id, 1500, 900, 1560, 930, "pending")                         # 4 тайла x0 ∈ {824, 896}

    out = tmp_path / "reviewed"
    names = ["plane", "ship", "harbor"]
    stats = export_module.export_reviewed(db_path, out, names)

    assert stats["tiles"] == {"train": 2, "val": 0}
    assert stats["skipped_pending"] == 4 and stats["labels"] == {"harbor": 2}
    label = (out / "labels" / "train" / "scene1_0_56.txt").read_text().split()
    assert label[0] == "2"  # исправленный экспертом класс
    assert [round(float(v) * 1024) for v in label[1:3]] == [100, 44]  # y сдвинут на y0 тайла
    image = cv2.imread(str(out / "images" / "train" / "scene1_0_0.jpg"))
    assert image.shape == (1024, 1024, 3) and image.mean() > 10
    yaml = (out / "reviewed.yaml").read_text()
    assert "val: images/train" in yaml and "  2: harbor" in yaml
