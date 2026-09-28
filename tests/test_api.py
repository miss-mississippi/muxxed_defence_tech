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
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["models"] == ["dota"]
    info = client.get("/api/model").json()
    assert [m["name"] for m in info["models"]] == ["dota"]
    assert info["models"][0]["classes"]["1"] == "ship"


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
    assert all(f["properties"]["model"] == "dota" for f in features)
    assert len(client.get("/api/detections", params={"model": "dota"}).json()["features"]) == len(features)
    assert client.get("/api/detections", params={"model": "mar20"}).json()["features"] == []
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


def test_delete_scene_removes_its_detections(client, utm_uint16_tif):
    with utm_uint16_tif.open("rb") as f:
        scene_id = client.post("/api/scenes", files={"file": ("temp.tif", f, "image/tiff")}).json()["id"]
    assert client.get(f"/api/scenes/{scene_id}").json()["status"] == "done"

    assert client.delete(f"/api/scenes/{scene_id}").status_code == 204
    assert client.get(f"/api/scenes/{scene_id}").status_code == 404
    assert client.get("/api/detections", params={"scene_id": scene_id}).json()["features"] == []
    assert client.delete(f"/api/scenes/{scene_id}").status_code == 404


def test_upload_rejects_unknown_format(client):
    r = client.post("/api/scenes", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_detect_sync_without_georeference(client):
    with SAMPLE.open("rb") as f:
        geojson = client.post("/api/detect", files={"file": ("boats.jpg", f, "image/jpeg")}).json()
    assert geojson["scene"]["path"] == "boats.jpg" and not geojson["scene"]["georeferenced"]
    assert geojson["features"] and geojson["features"][0]["geometry"] is None


def test_inventory_endpoint(client, utm_uint16_tif):
    with utm_uint16_tif.open("rb") as f:
        scene_id = client.post("/api/scenes", files={"file": ("inv.tif", f, "image/tiff")}).json()["id"]
    west, south, east, north = client.get(f"/api/scenes/{scene_id}").json()["bounds_wgs84"]
    area = {"type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]}
    doc = {"airfields": [{"id": "x", "name": "X", "area": area, "expected": {"C-17": 1}}]}
    r = client.post(f"/api/scenes/{scene_id}/inventory", json=doc)
    assert r.status_code == 200
    line = r.json()["airfields"][0]["types"][0]
    # модели типов в тестовом приложении нет: честный ответ «не распознаёт», а не «не обнаружено»
    assert line["type"] == "C-17" and line["status"] == "unrecognizable"
    assert client.post(f"/api/scenes/{scene_id}/inventory", json={"airfields": []}).status_code == 422
    assert client.post("/api/scenes/999999/inventory", json=doc).status_code == 404


def test_missed_objects_and_recall(client, utm_uint16_tif):
    from shapely.geometry import Point, shape

    with utm_uint16_tif.open("rb") as f:
        scene_id = client.post("/api/scenes", files={"file": ("recall.tif", f, "image/tiff")}).json()["id"]
    scene = client.get(f"/api/scenes/{scene_id}").json()
    feats = client.get("/api/detections", params={"scene_id": scene_id}).json()["features"]
    polys = [shape(f["geometry"]) for f in feats]

    # свободная точка внутри снимка, не попадающая ни в одно обнаружение
    west, south, east, north = scene["bounds_wgs84"]
    free = next(
        (lon, lat)
        for i in range(1, 40) for j in range(1, 40)
        for lon, lat in [(west + (east - west) * i / 40, south + (north - south) * j / 40)]
        if not any(p.covers(Point(lon, lat)) for p in polys)
    )

    url = f"/api/scenes/{scene_id}/missed"
    r = client.post(url, json={"class_name": "ship", "lon": free[0], "lat": free[1]})
    assert r.status_code == 201 and r.json()["geometry"]["type"] == "Point"
    missed_id = r.json()["id"]

    # поверх найденного объекта — это не пропуск
    c = feats[0]["properties"]
    assert client.post(url, json={"class_name": "ship", "lon": c["center_lon"], "lat": c["center_lat"]}).status_code == 409
    assert client.post(url, json={"class_name": "ship", "lon": 0.0, "lat": 0.0}).status_code == 422
    assert client.post(url, json={"class_name": "tank", "lon": free[0], "lat": free[1]}).status_code == 422
    assert client.post(url, json={"class_name": "ship"}).status_code == 422

    assert len(client.get(url).json()["features"]) == 1

    # до отметки о полноте — только верхняя оценка
    rec = client.get(f"/api/scenes/{scene_id}/recall").json()
    assert rec["complete"] is False and rec["total"]["missed"] == 1
    rec = client.put(f"/api/scenes/{scene_id}/completeness", json={"complete": True}).json()
    assert rec["complete"] is True and not any("верхняя" in w for w in rec["warnings"])
    assert "Полнота" in client.get(f"/api/scenes/{scene_id}/report").text

    assert client.delete(f"/api/missed/{missed_id}").status_code == 204
    assert client.delete(f"/api/missed/{missed_id}").status_code == 404
    assert client.get(url).json()["features"] == []

    # удаление сцены забирает и её отметки
    client.post(url, json={"class_name": "ship", "lon": free[0], "lat": free[1]})
    assert client.delete(f"/api/scenes/{scene_id}").status_code == 204
    assert client.get(url).status_code == 404
