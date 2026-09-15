import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from conftest import SAMPLE


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(data_dir=tmp_path_factory.mktemp("data"))
    with TestClient(app) as c:  # фоновые задачи TestClient выполняет до возврата ответа
        yield c


def test_health_and_model(client):
    assert client.get("/api/health").json()["status"] == "ok"
    model = client.get("/api/model").json()
    assert model["classes"]["1"] == "ship"


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "leaflet" in r.text.lower()


def test_scene_catalog_and_review_flow(client, utm_uint16_tif):
    with utm_uint16_tif.open("rb") as f:
        r = client.post("/api/scenes", files={"file": ("marina.tif", f, "image/tiff")})
    assert r.status_code == 202
    scene_id = r.json()["id"]

    scene = client.get(f"/api/scenes/{scene_id}").json()
    assert scene["status"] == "done", scene.get("error")
    assert scene["georeferenced"] and scene["gsd_m"] == pytest.approx(0.1, rel=0.01)
    assert scene["counts"]["ship"] > 100 and len(scene["preview_bounds"]) == 2

    features = client.get("/api/detections", params={"scene_id": scene_id}).json()["features"]
    assert len(features) == sum(scene["counts"].values())
    assert all(f["geometry"]["type"] == "Polygon" for f in features)
    confident = client.get("/api/detections", params={"scene_id": scene_id, "min_conf": 0.8}).json()["features"]
    assert 0 < len(confident) < len(features)

    west, south, east, north = scene["bounds_wgs84"]
    inside = client.get("/api/detections", params={"bbox": f"{west},{south},{east},{north}"}).json()["features"]
    assert len(inside) == len(features)
    assert client.get("/api/detections", params={"bbox": "0,0,1,1"}).json()["features"] == []

    first, second = features[0]["id"], features[1]["id"]
    r = client.patch(f"/api/detections/{first}", json={"status": "confirmed", "class_name": "harbor", "comment": "ok"})
    assert r.status_code == 200
    props = r.json()["properties"]
    assert props["review_status"] == "confirmed" and props["review_class"] == "harbor" and props["reviewed_at"]
    assert client.patch(f"/api/detections/{second}", json={"status": "rejected"}).status_code == 200
    assert client.patch(f"/api/detections/{second}", json={"status": "confirmed", "class_name": "tank"}).status_code == 422
    assert client.patch("/api/detections/999999", json={"status": "rejected"}).status_code == 404

    confirmed = client.get("/api/detections", params={"status": "confirmed"}).json()["features"]
    assert [f["id"] for f in confirmed] == [first]
    harbors = client.get("/api/detections", params={"scene_id": scene_id, "class_name": "harbor"}).json()["features"]
    assert first in {f["id"] for f in harbors}

    stats = client.get("/api/stats").json()
    assert stats["by_status"]["confirmed"] == 1 and stats["by_status"]["rejected"] == 1
    listed = next(s for s in client.get("/api/scenes").json() if s["id"] == scene_id)
    assert listed["review"]["confirmed"] == 1

    preview = client.get(f"/api/scenes/{scene_id}/preview.png")
    assert preview.status_code == 200 and preview.content[:4] == b"\x89PNG"
    report = client.get(f"/api/scenes/{scene_id}/report")
    assert report.status_code == 200 and "marina.tif" in report.text and "подтверждено: 1" in report.text


def test_upload_rejects_unknown_format(client):
    r = client.post("/api/scenes", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_detect_sync_without_georeference(client):
    with SAMPLE.open("rb") as f:
        geojson = client.post("/api/detect", files={"file": ("boats.jpg", f, "image/jpeg")}).json()
    assert geojson["scene"]["path"] == "boats.jpg" and not geojson["scene"]["georeferenced"]
    assert geojson["features"] and geojson["features"][0]["geometry"] is None
