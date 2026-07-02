from pathlib import Path
from src.exercise_db import ExerciseDB

FIX = Path(__file__).parent / "fixtures" / "exercises_sample.json"


def test_lookup_exact():
    db = ExerciseDB(FIX)
    ex = db.lookup("Goblet Squat")
    assert ex["primary_muscles"] == ["quadriceps"]
    assert ex["demo_image"].endswith("GobletSquat/0.jpg")


def test_lookup_case_insensitive():
    db = ExerciseDB(FIX)
    assert db.lookup("push up")["equipment"] == "body only"


def test_lookup_missing_returns_none():
    db = ExerciseDB(FIX)
    assert db.lookup("Deadlift") is None
