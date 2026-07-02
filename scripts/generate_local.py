"""Keyless dataset generation (no API/keys). Writes sft.jsonl + dpo.jsonl using
the deterministic, real-data-grounded generator in src/local_gen.py.

Format matches the LLM path exactly (same SYSTEM prompt + chat format) so the
training scripts are unchanged.
"""
import json
import argparse
from pathlib import Path

from src.profiles import sample_profiles
from src.local_gen import load_exercises, build_plan
from scripts.generate_sft import SYSTEM, user_prompt

ROOT = Path(__file__).resolve().parent.parent
SFT_OUT = ROOT / "data" / "generated" / "sft.jsonl"
DPO_OUT = ROOT / "data" / "generated" / "dpo.jsonl"


def dpo_prompt(p):
    return (f"{p['age']}yo, {p['weight_kg']}kg, goal {p['goal']}, {p['equipment']}, "
            f"injury: {p['injury']}, {p['days_per_week']} days/week. JSON plan.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", type=int, default=600)
    ap.add_argument("--dpo", type=int, default=200)
    args = ap.parse_args()

    records = load_exercises()
    SFT_OUT.parent.mkdir(parents=True, exist_ok=True)

    with SFT_OUT.open("w", encoding="utf-8") as f:
        for p in sample_profiles(args.sft, seed=0):
            plan = build_plan(p, records)
            rec = {"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt(p)},
                {"role": "assistant", "content": json.dumps(plan)}]}
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {args.sft} SFT examples -> {SFT_OUT}")

    with DPO_OUT.open("w", encoding="utf-8") as f:
        for p in sample_profiles(args.dpo, seed=99):
            chosen = json.dumps(build_plan(p, records, unsafe=False))
            rejected = json.dumps(build_plan(p, records, unsafe=True))
            f.write(json.dumps({"prompt": dpo_prompt(p),
                                "chosen": chosen, "rejected": rejected}) + "\n")
    print(f"Wrote {args.dpo} DPO pairs -> {DPO_OUT}")


if __name__ == "__main__":
    main()
