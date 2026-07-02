import pytest
from pydantic import ValidationError
from src.schema import FitnessPlan

VALID = {
    "goal": "lose fat",
    "experience": "beginner",
    "daily_schedule": {
        "wake": "07:00",
        "workout": {"time": "18:00", "type": "Full body"},
        "meals": [{"time": "08:00", "name": "Breakfast", "focus": "protein"}],
        "sleep": {"target": "22:30-07:00", "hours": 8},
    },
    "weekly_workouts": [{
        "day": "Monday", "focus": "Full body",
        "exercises": [{"name": "Goblet Squat", "sets": 3, "reps": "10",
                       "rest_seconds": 60, "demo_image": None, "why": None}],
    }],
    "nutrition": {
        "daily_macros": {"calories": 2000, "protein_g": 150, "carbs_g": 200, "fat_g": 60},
        "example_day": [{"food": "Chicken", "grams": 150, "calories": 247, "protein_g": 46}],
        "grocery_list": ["Chicken", "Rice"],
    },
    "disclaimer": "General guidance, not medical advice.",
}


def test_valid_plan_parses():
    plan = FitnessPlan.model_validate(VALID)
    assert plan.goal == "lose fat"
    assert plan.nutrition.daily_macros.protein_g == 150


def test_missing_field_raises():
    bad = {k: v for k, v in VALID.items() if k != "nutrition"}
    with pytest.raises(ValidationError):
        FitnessPlan.model_validate(bad)


def test_bad_experience_enum_raises():
    bad = {**VALID, "experience": "pro"}
    with pytest.raises(ValidationError):
        FitnessPlan.model_validate(bad)


def test_reps_int_is_coerced_to_string():
    import copy
    data = copy.deepcopy(VALID)
    data["weekly_workouts"][0]["exercises"][0]["reps"] = 12  # model emits int
    plan = FitnessPlan.model_validate(data)
    assert plan.weekly_workouts[0].exercises[0].reps == "12"
