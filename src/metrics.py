import json

CONTRAINDICATED = {
    "knee pain": ["squat", "lunge", "leg press"],
    "shoulder impingement": ["overhead press", "upright row", "bench press"],
    "lower back pain": ["deadlift", "good morning", "bent over row"],
}


def valid_json_rate(outputs: list[str]) -> float:
    if not outputs:
        return 0.0
    ok = 0
    for o in outputs:
        try:
            json.loads(o)
            ok += 1
        except Exception:
            pass
    return ok / len(outputs)


def macro_close(pred: float, truth: float, tol: float = 0.1) -> bool:
    if truth == 0:
        return pred == 0
    return abs(pred - truth) / truth <= tol


def respects_equipment(plan: dict, equipment: str) -> bool:
    # bodyweight-only plans must not require gym machines
    if equipment != "bodyweight only":
        return True
    banned = ["barbell", "machine", "cable", "leg press"]
    for day in plan.get("weekly_workouts", []):
        for ex in day.get("exercises", []):
            if any(b in ex["name"].lower() for b in banned):
                return False
    return True


def avoids_injury(plan: dict, injury) -> bool:
    if not injury:
        return True
    bad = CONTRAINDICATED.get(injury, [])
    for day in plan.get("weekly_workouts", []):
        for ex in day.get("exercises", []):
            if any(b in ex["name"].lower() for b in bad):
                return False
    return True
