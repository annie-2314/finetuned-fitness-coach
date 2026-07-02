from src.schema import FitnessPlan


def validate_example(rec: dict) -> bool:
    try:
        content = rec["messages"][-1]["content"]
        FitnessPlan.model_validate_json(content)
        return True
    except Exception:
        return False


def enrich_media(plan: dict, db) -> dict:
    for day in plan.get("weekly_workouts", []):
        for ex in day.get("exercises", []):
            found = db.lookup(ex["name"])
            if found and found["demo_image"]:
                ex["demo_image"] = found["demo_image"]
    return plan
