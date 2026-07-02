import random

GOALS = ["lose fat", "build muscle", "maintain", "gain strength"]
EQUIPMENT = ["full gym", "home dumbbells", "bodyweight only", "resistance bands"]
DIETS = ["no restriction", "vegetarian", "vegan", "keto"]
EXPERIENCE = ["beginner", "intermediate", "advanced"]
INJURIES = [None, "knee pain", "shoulder impingement", "lower back pain"]
DAYS = [3, 4, 5, 6]


def sample_profiles(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append({
            "age": rng.randint(18, 60),
            "weight_kg": rng.randint(50, 110),
            "goal": rng.choice(GOALS),
            "equipment": rng.choice(EQUIPMENT),
            "diet": rng.choice(DIETS),
            "experience": rng.choice(EXPERIENCE),
            "injury": rng.choice(INJURIES),
            "days_per_week": rng.choice(DAYS),
        })
    return out
