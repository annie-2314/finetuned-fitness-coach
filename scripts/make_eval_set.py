import os
import json
from pathlib import Path
from dotenv import load_dotenv
from src.ssl_setup import enable_os_trust_store
from src.profiles import sample_profiles
from src.nutrition import lookup_nutrition

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "eval.jsonl"


def main():
    enable_os_trust_store()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    key = os.environ["USDA_API_KEY"]
    with OUT.open("w", encoding="utf-8") as f:
        for p in sample_profiles(150, seed=424242):   # unique seed => held-out
            truth = lookup_nutrition("chicken breast", 150, key)
            f.write(json.dumps({"profile": p, "nutrition_truth": truth}) + "\n")
    print(f"Wrote 150 eval profiles -> {OUT}")


if __name__ == "__main__":
    main()
