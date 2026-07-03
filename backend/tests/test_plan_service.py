import json
from types import SimpleNamespace
from app.plan_service import build_user_prompt, generate_plan, summarize_logs

VALID_PLAN = {
    "goal": "lose fat", "experience": "beginner",
    "daily_schedule": {"wake": "07:00", "workout": {"time": "18:00", "type": "Full"},
        "meals": [{"time": "08:00", "name": "B", "focus": "p"}],
        "sleep": {"target": "22:30-07:00", "hours": 8}},
    "weekly_workouts": [{"day": "Mon", "focus": "Full", "exercises": [
        {"name": "Push Up", "sets": 3, "reps": "12", "rest_seconds": 60,
         "demo_image": None, "why": "chest"}]}],
    "nutrition": {"daily_macros": {"calories": 2000, "protein_g": 150, "carbs_g": 200, "fat_g": 60},
        "example_day": [{"food": "Chicken", "grams": 150, "calories": 247, "protein_g": 46}],
        "grocery_list": ["Chicken"]},
    "disclaimer": "General guidance, not medical advice.",
}

PROFILE = {"age": 30, "weight_kg": 80, "goal": "lose fat", "equipment": "home dumbbells",
           "diet": "no restriction", "experience": "beginner", "injury": "knee pain",
           "days_per_week": 3}


def test_build_user_prompt_includes_profile_and_adaptation():
    p = build_user_prompt(PROFILE, adaptation="ease off")
    assert "knee pain" in p and "lose fat" in p and "ease off" in p


def test_generate_plan_valid(monkeypatch):
    plan = generate_plan(PROFILE, chat_fn=lambda messages: json.dumps(VALID_PLAN))
    assert plan["goal"] == "lose fat"
    assert plan["weekly_workouts"][0]["exercises"][0]["name"] == "Push Up"


def test_generate_plan_retries_then_fails():
    calls = {"n": 0}
    def bad(messages):
        calls["n"] += 1
        return "{not json"
    try:
        generate_plan(PROFILE, chat_fn=bad)
        assert False, "should have raised"
    except ValueError:
        pass
    assert calls["n"] == 2   # retried once


def test_summarize_logs_flags_hard_and_pain():
    logs = [SimpleNamespace(rpe=10, feedback=None),
            SimpleNamespace(rpe=3, feedback=None),
            SimpleNamespace(rpe=None, feedback="knee pain during squats")]
    note = summarize_logs(logs)
    assert "hard" in note and "easy" in note and "discomfort" in note
