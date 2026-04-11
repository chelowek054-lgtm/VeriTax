from fastapi.testclient import TestClient

from app.main import app


def test_home_returns_html() -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "VeriTax" in response.text
