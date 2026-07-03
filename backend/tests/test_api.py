import pytest
from fastapi.testclient import TestClient
from tests.test_plan_service import VALID_PLAN


@pytest.fixture()
def client(monkeypatch):
    from app.db import Base, engine
    import app.main
    from app.routes import plan_routes
    # fresh schema for each test (isolation)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    # stub the LLM so no network is needed
    monkeypatch.setattr(plan_routes, "generate_plan",
                        lambda profile, adaptation=None: dict(VALID_PLAN))
    return TestClient(app.main.app)


def _auth(client):
    r = client.post("/auth/signup", json={"email": "a@b.com", "password": "pw123456"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_full_flow(client):
    h = _auth(client)
    prof = {"age": 30, "weight_kg": 80, "sex": "male", "goal": "lose fat", "equipment": "home dumbbells",
            "diet": "no restriction", "experience": "beginner", "injury": None, "days_per_week": 3}
    assert client.put("/profile", json=prof, headers=h).status_code == 200
    r = client.post("/plan/generate", headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["plan"]["goal"] == "lose fat"
    log = {"exercise": "Push Up", "sets_done": 3, "reps_done": "12", "rpe": 7}
    assert client.post("/logs", json=log, headers=h).status_code == 200
    prog = client.get("/logs/progress", headers=h).json()
    assert prog["total_sessions"] == 1


def test_requires_auth(client):
    assert client.get("/profile").status_code == 401


def test_duplicate_signup_conflicts(client):
    client.post("/auth/signup", json={"email": "d@e.com", "password": "pw123456"})
    r = client.post("/auth/signup", json={"email": "d@e.com", "password": "pw123456"})
    assert r.status_code == 409
