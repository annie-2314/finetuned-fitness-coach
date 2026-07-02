import json
from pathlib import Path
from src.curate import validate_example, enrich_media
from src.exercise_db import ExerciseDB

FIX = Path(__file__).parent / "fixtures" / "exercises_sample.json"


def _plan():
    return {"goal": "x", "experience": "beginner",
        "daily_schedule": {"wake": "07:00", "workout": {"time": "18:00", "type": "Full"},
            "meals": [{"time": "08:00", "name": "B", "focus": "p"}],
            "sleep": {"target": "22:30-07:00", "hours": 8}},
        "weekly_workouts": [{"day": "Mon", "focus": "F", "exercises": [
            {"name": "Goblet Squat", "sets": 3, "reps": "10", "rest_seconds": 60,
             "demo_image": None, "why": "legs"}]}],
        "nutrition": {"daily_macros": {"calories": 2000, "protein_g": 150, "carbs_g": 200, "fat_g": 60},
            "example_day": [{"food": "Chicken", "grams": 150, "calories": 247, "protein_g": 46}],
            "grocery_list": ["Chicken"]},
        "disclaimer": "General guidance, not medical advice."}


def test_validate_good_example():
    rec = {"messages": [{"role": "assistant", "content": json.dumps(_plan())}]}
    assert validate_example(rec) is True


def test_validate_rejects_bad_json():
    rec = {"messages": [{"role": "assistant", "content": "{not json"}]}
    assert validate_example(rec) is False


def test_enrich_fills_demo_image():
    db = ExerciseDB(FIX)
    enriched = enrich_media(_plan(), db)
    img = enriched["weekly_workouts"][0]["exercises"][0]["demo_image"]
    assert img and img.endswith("GobletSquat/0.jpg")
