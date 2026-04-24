from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.ml_client import MlServiceUnavailableError


class FakeDB:
    def __init__(self):
        self.next_user_id = 1
        self.next_request_id = 1
        self.users = []
        self.requests = []
        self.initialized = False

    def initialize(self):
        self.initialized = True

    def health(self):
        return {"status": "healthy", "detail": "Fake database ready."}

    def create_user(self, username, password_hash):
        if self.get_user_by_username(username):
            from app.db import DuplicateUserError

            raise DuplicateUserError(username)
        user = {"id": self.next_user_id, "username": username, "password_hash": password_hash}
        self.next_user_id += 1
        self.users.append(user)
        return user

    def get_user_by_username(self, username):
        return next((user for user in self.users if user["username"] == username), None)

    def get_user_by_id(self, user_id):
        return next((user for user in self.users if user["id"] == user_id), None)

    def record_prediction(self, user_id, title, preview, status, response_time_ms, tags, error_message=None):
        record = {
            "id": str(self.next_request_id),
            "user_id": user_id,
            "title": title,
            "preview": preview,
            "status": status,
            "response_time_ms": response_time_ms,
            "tags": list(tags),
            "error_message": error_message,
            "created_at": f"2026-04-23T00:00:0{self.next_request_id}Z",
        }
        self.next_request_id += 1
        self.requests.append(record)
        return record

    def get_global_metrics(self):
        total = len(self.requests)
        failed = sum(1 for record in self.requests if record["status"] == "failed")
        average = sum(record["response_time_ms"] for record in self.requests) / total if total else 0.0
        return {
            "total_requests": total,
            "failed_requests": failed,
            "average_response_time_ms": average,
            "success_rate": 0.0 if total == 0 else (total - failed) / total,
        }

    def get_recent_requests(self, user_id, *, limit=10):
        records = [record for record in reversed(self.requests) if record["user_id"] == user_id]
        return records[:limit]


class FakeML:
    def __init__(self, tags=None, available=True):
        self.tags = tags or ["python", "deployment"]
        self.available = available

    def health(self):
        if self.available:
            return {"status": "healthy", "detail": "Fake ML ready."}
        return {"status": "unavailable", "detail": "Fake ML unavailable."}

    def predict_tags(self, text):
        if not self.available:
            raise MlServiceUnavailableError("Fake ML is offline")
        return list(self.tags)


def make_client(tmp_path: Path, *, ml_available=True):
    dist_dir = tmp_path / "dist"
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html><body>demo shell</body></html>", encoding="utf-8")
    db = FakeDB()
    ml = FakeML(available=ml_available)
    app = create_app(db=db, ml_client=ml, frontend_dist=dist_dir)
    return TestClient(app), db, ml


def signup(client: TestClient, username="alice", password="password123"):
    return client.post("/api/auth/signup", json={"username": username, "password": password})


def test_health_and_spa_fallback_are_available(tmp_path: Path):
    client, db, _ = make_client(tmp_path)
    with client:
        health_response = client.get("/healthz")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
        assert db.initialized is True

        login_page = client.get("/login")
        assert login_page.status_code == 200
        assert "demo shell" in login_page.text


def test_signup_login_logout_and_session_flow(tmp_path: Path):
    client, _, _ = make_client(tmp_path)
    with client:
        signup_response = signup(client)
        assert signup_response.status_code == 200
        assert signup_response.json() == {"username": "alice", "authenticated": True}
        assert "app_session" in signup_response.cookies

        me_response = client.get("/api/me")
        assert me_response.status_code == 200
        assert me_response.json()["username"] == "alice"

        logout_response = client.post("/api/auth/logout")
        assert logout_response.status_code == 200
        assert logout_response.json() == {"authenticated": False}

        unauthorized_response = client.get("/api/me")
        assert unauthorized_response.status_code == 401

        login_response = client.post("/api/auth/login", json={"username": "alice", "password": "password123"})
        assert login_response.status_code == 200
        assert login_response.json()["username"] == "alice"


def test_signup_allows_short_passwords_for_demo(tmp_path: Path):
    client, _, _ = make_client(tmp_path)
    with client:
        signup_response = signup(client, username="shortpass-user", password="123")
        assert signup_response.status_code == 200
        assert signup_response.json() == {"username": "shortpass-user", "authenticated": True}


def test_predict_and_status_only_store_a_short_preview(tmp_path: Path):
    client, _, _ = make_client(tmp_path)
    long_body = "sentence " * 80

    with client:
        signup(client)
        predict_response = client.post(
            "/api/predict",
            json={
                "title": "Demo title",
                "body": long_body,
            },
        )
        assert predict_response.status_code == 200
        assert predict_response.json()["tags"] == ["python", "deployment"]

        status_response = client.get("/api/status")
        assert status_response.status_code == 200
        payload = status_response.json()
        assert payload["total_requests"] == 1
        assert payload["failed_requests"] == 0
        assert payload["health"]["ml_service"]["status"] == "healthy"
        assert len(payload["recent_requests"]) == 1
        request = payload["recent_requests"][0]
        preview = request["preview"]
        assert request["title"] == "Demo title"
        assert len(preview) <= 160
        assert preview != long_body
        assert "Demo title" not in preview


def test_predict_returns_503_and_records_failed_request_when_ml_is_unavailable(tmp_path: Path):
    client, _, _ = make_client(tmp_path, ml_available=False)

    with client:
        signup(client)
        predict_response = client.post(
            "/api/predict",
            json={"title": "Offline demo", "body": "A short body for a failed call."},
        )
        assert predict_response.status_code == 503
        assert "temporarily unavailable" in predict_response.json()["detail"]

        status_response = client.get("/api/status")
        payload = status_response.json()
        assert payload["total_requests"] == 1
        assert payload["failed_requests"] == 1
        assert payload["recent_requests"][0]["status"] == "failed"
        assert payload["health"]["status"] == "degraded"
        assert payload["health"]["ml_service"]["status"] == "unavailable"
