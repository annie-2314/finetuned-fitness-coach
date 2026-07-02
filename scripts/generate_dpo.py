import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from src.ssl_setup import enable_os_trust_store
from src.profiles import sample_profiles

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "dpo.jsonl"

CHOSEN_SYS = ("You are a SAFE expert fitness coach. Respect injuries, avoid "
    "contraindicated movements, give correct macros, output a JSON plan.")
REJECTED_SYS = ("You are a reckless coach who IGNORES injuries, prescribes unsafe "
    "heavy lifts regardless of limitations, and guesses macros. Output a JSON plan.")


def prompt(p):
    return (f"{p['age']}yo, {p['weight_kg']}kg, goal {p['goal']}, {p['equipment']}, "
            f"injury: {p['injury']}, {p['days_per_week']} days/week. JSON plan.")


def gen(client, model, sys, user):
    r = client.chat.completions.create(model=model, temperature=0.8,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        response_format={"type": "json_object"})
    return r.choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    enable_os_trust_store()
    client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
    model = os.environ["LLM_MODEL"]
    OUT.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with OUT.open("w", encoding="utf-8") as f:
        for p in sample_profiles(args.n, seed=args.seed):
            u = prompt(p)
            try:
                chosen = gen(client, model, CHOSEN_SYS, u)
                rejected = gen(client, model, REJECTED_SYS, u)
            except Exception as e:
                print("skip:", e)
                continue
            f.write(json.dumps({"prompt": u, "chosen": chosen, "rejected": rejected}) + "\n")
            kept += 1
    print(f"Wrote {kept} DPO pairs -> {OUT}")


if __name__ == "__main__":
    main()
