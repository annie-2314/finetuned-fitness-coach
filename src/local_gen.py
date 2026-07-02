"""Keyless, deterministic plan generator.

Builds schema-valid FitnessPlan dicts from the real exercise DB + a nutrition
formula (no LLM, no API keys, no rate limits). Grounded in real data:
- exercises come from free-exercise-db (filtered by equipment + injury safety)
- calories via a Mifflin-St Jeor-style estimate; macros via g/kg targets

Trade-off vs LLM generation: templated language (less linguistic variety), but
100% structurally valid, safe, and free. Used as the primary generator when the
LLM free tier is exhausted; can be mixed with LLM data later for variety.
"""
import json
import random
from pathlib import Path

from src.metrics import CONTRAINDICATED

ROOT = Path(__file__).resolve().parent.parent
EXERCISES = ROOT / "data" / "raw" / "exercises.json"

DISCLAIMER = ("This is general guidance and not medical advice. Consult a "
              "qualified professional before starting a new program.")

# which dataset 'equipment' values each profile equipment allows
EQUIP_ALLOWED = {
    "full gym": None,  # None => all
    "home dumbbells": {"dumbbell", "body only", "none", "kettlebells"},
    "bodyweight only": {"body only", "none"},
    "resistance bands": {"bands", "body only", "none"},
}


def load_exercises(path=EXERCISES):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tdee(profile):
    bmr = profile["weight_kg"] * 22            # simple BMR proxy (kcal)
    tdee = bmr * 1.55                          # moderate activity factor
    goal = profile["goal"]
    if goal == "lose fat":
        tdee -= 500
    elif goal in ("build muscle", "gain strength"):
        tdee += 300
    return int(round(tdee / 10) * 10)


def _macros(profile):
    cals = _tdee(profile)
    w = profile["weight_kg"]
    protein = round(1.8 * w)
    fat = round(0.9 * w)
    carbs = max(0, round((cals - (protein * 4 + fat * 9)) / 4))
    return {"calories": cals, "protein_g": protein, "carbs_g": carbs, "fat_g": fat}


def _eligible(records, profile):
    allowed = EQUIP_ALLOWED.get(profile["equipment"])
    contra = CONTRAINDICATED.get(profile["injury"], [])
    out = []
    for e in records:
        equip = (e.get("equipment") or "").lower()
        if allowed is not None and equip not in allowed:
            continue
        if any(c in e["name"].lower() for c in contra):
            continue
        out.append(e)
    return out


def _reps_for(goal):
    return "5-8" if goal in ("gain strength", "build muscle") else "10-15"


def build_plan(profile, records, unsafe=False):
    """Return a schema-valid FitnessPlan dict. If unsafe=True, deliberately
    violates injury/equipment/macro correctness (used as DPO 'rejected')."""
    rng = random.Random(hash((profile["age"], profile["weight_kg"],
                              profile["goal"], profile["injury"])) & 0xFFFFFFFF)
    pool = list(records) if unsafe else _eligible(records, profile)
    if not pool:
        pool = list(records)
    rng.shuffle(pool)

    beginner = profile["experience"] == "beginner"
    reps = "20" if unsafe else _reps_for(profile["goal"])
    days = []
    focuses = ["Full body", "Upper body", "Lower body", "Push", "Pull", "Core"]
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    ptr = 0
    for d in range(profile["days_per_week"]):
        exercises = []
        for _ in range(4):
            if ptr >= len(pool):
                ptr = 0
            e = pool[ptr]
            ptr += 1
            exercises.append({
                "name": e["name"],
                "sets": 5 if unsafe else 3,
                "reps": reps,
                "rest_seconds": 30 if unsafe else 90,
                "demo_image": None,
                "why": None if not beginner else f"Builds {(e.get('primaryMuscles') or ['strength'])[0]}.",
            })
        days.append({"day": day_names[d % len(day_names)],
                     "focus": focuses[d % len(focuses)],
                     "exercises": exercises})

    macros = _macros(profile)
    if unsafe:
        macros = {"calories": 1000, "protein_g": 40, "carbs_g": 60, "fat_g": 20}

    return {
        "goal": profile["goal"],
        "experience": profile["experience"],
        "daily_schedule": {
            "wake": "07:00",
            "workout": {"time": "18:00", "type": days[0]["focus"]},
            "meals": [
                {"time": "08:00", "name": "Breakfast", "focus": "protein + carbs"},
                {"time": "13:00", "name": "Lunch", "focus": "balanced"},
                {"time": "16:30", "name": "Snack", "focus": "pre-workout fuel"},
                {"time": "20:00", "name": "Dinner", "focus": "protein + recovery"},
            ],
            "sleep": {"target": "22:30-07:00", "hours": 8},
        },
        "weekly_workouts": days,
        "nutrition": {
            "daily_macros": macros,
            "example_day": [
                {"food": "chicken breast", "grams": 150, "calories": 247, "protein_g": 46},
                {"food": "brown rice", "grams": 200, "calories": 220, "protein_g": 5},
            ],
            "grocery_list": ["chicken breast", "brown rice", "eggs", "vegetables", "olive oil"],
        },
        "disclaimer": DISCLAIMER,
    }
