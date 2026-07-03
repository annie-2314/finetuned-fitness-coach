"""Generates a fitness plan from a profile.

Uses the SAME system prompt the model was fine-tuned with (train/serve parity).
For local dev it calls an LLM API (Groq/OpenRouter); the fine-tuned model can be
swapped in later by pointing LLM_BASE_URL at a hosted endpoint. `chat_fn` is
injectable so tests run without any network.
"""
import json
from app.config import settings
from app.schemas import FitnessPlan
from app.exercise_db import ExerciseDB

SYSTEM = (
    "You are an expert fitness coach. Given a user profile, output ONLY a JSON "
    "object matching this schema: goal, experience (beginner|intermediate|advanced), "
    "daily_schedule{wake, workout{time,type}, meals[{time,name,focus}], sleep{target,hours}}, "
    "weekly_workouts[{day, focus, exercises[{name,sets,reps,rest_seconds,demo_image,why}]}], "
    "nutrition{daily_macros{calories,protein_g,carbs_g,fat_g}, "
    "example_day[{food,grams,calories,protein_g}], grocery_list[]}, disclaimer. "
    "Use realistic exercises and macros. Set demo_image to null (filled later). "
    "For beginners, fill 'why' with a short reason. Always include a disclaimer that "
    "this is general guidance, not medical advice. If the user reports an injury, "
    "avoid contraindicated movements."
)

_exercise_db = ExerciseDB()


def build_user_prompt(p: dict, adaptation: str | None = None) -> str:
    base = (f"Profile: {p['age']}yo, {p['weight_kg']}kg, goal: {p['goal']}, "
            f"equipment: {p['equipment']}, diet: {p['diet']}, "
            f"experience: {p['experience']}, injury: {p.get('injury')}, "
            f"{p['days_per_week']} days/week. Return the JSON plan.")
    if adaptation:
        base += f"\nAdapt based on recent feedback: {adaptation}"
    return base


def _default_chat(messages) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=settings.LLM_BASE_URL, api_key=settings.LLM_API_KEY)
    resp = client.chat.completions.create(
        model=settings.LLM_MODEL, temperature=0.7, max_tokens=2500,
        messages=messages, response_format={"type": "json_object"})
    return resp.choices[0].message.content


def _enrich_media(plan: dict) -> dict:
    for day in plan.get("weekly_workouts", []):
        for ex in day.get("exercises", []):
            found = _exercise_db.lookup(ex.get("name", ""))
            if found and found["demo_image"]:
                ex["demo_image"] = found["demo_image"]
    return plan


def generate_plan(profile: dict, adaptation: str | None = None, chat_fn=None) -> dict:
    chat_fn = chat_fn or _default_chat
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user_prompt(profile, adaptation)}]
    last_err = None
    for _ in range(2):  # one retry if the model returns invalid JSON
        raw = chat_fn(messages)
        try:
            plan = FitnessPlan.model_validate_json(raw).model_dump()
            return _enrich_media(plan)
        except Exception as e:
            last_err = e
    raise ValueError(f"model produced invalid plan: {last_err}")


def summarize_logs(logs) -> str:
    """Turn recent workout logs into a short adaptation note for the next plan."""
    if not logs:
        return ""
    hard = [l for l in logs if (l.rpe or 0) >= 9]
    easy = [l for l in logs if 0 < (l.rpe or 0) <= 5]
    pains = [l.feedback for l in logs if l.feedback and "pain" in l.feedback.lower()]
    parts = []
    if hard:
        parts.append(f"{len(hard)} sessions felt very hard (RPE>=9) — ease off / deload slightly")
    if easy:
        parts.append(f"{len(easy)} sessions felt easy (RPE<=5) — add progressive overload")
    if pains:
        parts.append(f"reported discomfort: {'; '.join(pains[:3])} — avoid aggravating movements")
    return " | ".join(parts)
