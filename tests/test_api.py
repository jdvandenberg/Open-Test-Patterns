from fastapi.testclient import TestClient

from open_test_patterns.api.app import app

client = TestClient(app)


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_list_patterns():
    data = client.get("/api/patterns").json()
    assert len(data) >= 15
    assert all("parameters" in p for p in data)


def test_preview_returns_png():
    r = client.post(
        "/api/preview",
        json={"pattern_id": "color-bars", "width": 320, "height": 180, "params": {}},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_download():
    r = client.post(
        "/api/render",
        json={
            "pattern_id": "smpte-rp219-2-2016",
            "width": 640,
            "height": 360,
            "params": {},
            "format": "tiff",
            "bit_depth": 16,
        },
    )
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]


def test_unknown_pattern_404():
    r = client.post("/api/preview", json={"pattern_id": "nope", "width": 8, "height": 8})
    assert r.status_code == 404
