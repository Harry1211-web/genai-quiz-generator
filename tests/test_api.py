from fastapi.testclient import TestClient

from api.main import app


def test_generate_endpoint():
    c = TestClient(app)
    with open("data/samples/mang_may_tinh_tcp.md", "rb") as f:
        r = c.post("/generate?n=1&max_sections=2&strategy=v2", files={"file": ("bai.md", f, "text/markdown")})
    assert r.status_code == 200
    body = r.json()
    assert body["stats"]["total"] >= 1 and "questions" in body

def test_reject_type():
    r = TestClient(app).post("/generate", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415
