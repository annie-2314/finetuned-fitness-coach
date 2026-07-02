import json
from pathlib import Path
from fastapi.testclient import TestClient
from tests.test_curate import _plan   # reuse a valid plan
from app.main import create_app

FIX = Path(__file__).parent / "fixtures" / "exercises_sample.json"


def stub_generate(profile):        # pretends to be the model
    return json.dumps(_plan())


def test_plan_endpoint_returns_enriched_plan():
    app = create_app(generate_fn=stub_generate, exercises_path=FIX)
    client = TestClient(app)
    resp = client.post("/plan", json={"age": 30, "weight_kg": 80, "goal": "lose fat",
        "equipment": "home dumbbells", "diet": "no restriction",
        "experience": "beginner", "injury": None, "days_per_week": 3})
    assert resp.status_code == 200
    body = resp.json()
    img = body["weekly_workouts"][0]["exercises"][0]["demo_image"]
    assert img.endswith("GobletSquat/0.jpg")   # media enriched from DB
