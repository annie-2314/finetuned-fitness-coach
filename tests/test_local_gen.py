from src.local_gen import build_plan
from src.schema import FitnessPlan
from src.metrics import respects_equipment, avoids_injury

RECORDS = [
    {"name": "Goblet Squat", "equipment": "dumbbell", "primaryMuscles": ["quadriceps"]},
    {"name": "Push Up", "equipment": "body only", "primaryMuscles": ["chest"]},
    {"name": "Barbell Deadlift", "equipment": "barbell", "primaryMuscles": ["lower back"]},
    {"name": "Dumbbell Row", "equipment": "dumbbell", "primaryMuscles": ["middle back"]},
    {"name": "Plank", "equipment": "body only", "primaryMuscles": ["abdominals"]},
]


def _profile(**kw):
    base = {"age": 30, "weight_kg": 80, "goal": "build muscle",
            "equipment": "full gym", "diet": "no restriction",
            "experience": "beginner", "injury": None, "days_per_week": 3}
    base.update(kw)
    return base


def test_generated_plan_validates():
    plan = build_plan(_profile(), RECORDS)
    FitnessPlan.model_validate(plan)     # raises if invalid


def test_bodyweight_plan_respects_equipment():
    plan = build_plan(_profile(equipment="bodyweight only"), RECORDS)
    assert respects_equipment(plan, "bodyweight only") is True


def test_injury_plan_avoids_contraindicated():
    plan = build_plan(_profile(injury="lower back pain"), RECORDS)
    assert avoids_injury(plan, "lower back pain") is True   # no deadlift


def test_unsafe_plan_is_actually_unsafe():
    plan = build_plan(_profile(injury="lower back pain"), RECORDS, unsafe=True)
    # unsafe ignores injury filtering, so contraindicated lifts can appear
    assert avoids_injury(plan, "lower back pain") is False


def test_beginner_gets_why_notes():
    plan = build_plan(_profile(experience="beginner"), RECORDS)
    assert plan["weekly_workouts"][0]["exercises"][0]["why"]
