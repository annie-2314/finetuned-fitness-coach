import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from src.ssl_setup import enable_os_trust_store
from src.profiles import sample_profiles
from src.schema import FitnessPlan

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "sft.jsonl"

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


def user_prompt(p: dict) -> str:
    return (f"Profile: {p['age']}yo, {p['weight_kg']}kg, goal: {p['goal']}, "
            f"equipment: {p['equipment']}, diet: {p['diet']}, "
            f"experience: {p['experience']}, injury: {p['injury']}, "
            f"{p['days_per_week']} days/week. Return the JSON plan.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)     # pilot default
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    enable_os_trust_store()
    client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
    model = os.environ["LLM_MODEL"]
    OUT.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with OUT.open("w", encoding="utf-8") as f:
        for p in sample_profiles(args.n, seed=args.seed):
            up = user_prompt(p)
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=0.7,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": up}],
                    response_format={"type": "json_object"})
                content = resp.choices[0].message.content
                FitnessPlan.model_validate_json(content)   # drop invalid
            except Exception as e:
                print("skip:", e)
                continue
            rec = {"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": up},
                {"role": "assistant", "content": content}]}
            f.write(json.dumps(rec) + "\n")
            kept += 1
    print(f"Wrote {kept}/{args.n} valid examples -> {OUT}")


if __name__ == "__main__":
    main()
