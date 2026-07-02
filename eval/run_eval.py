import json
from pathlib import Path
from src.schema import FitnessPlan
from src.metrics import valid_json_rate, respects_equipment, avoids_injury

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "generated" / "eval.jsonl"


def evaluate(generate_fn, label):
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines()]
    outputs, eq_ok, inj_ok = [], 0, 0
    for r in rows:
        p = r["profile"]
        text = generate_fn(p)                 # model returns JSON string
        outputs.append(text)
        try:
            plan = FitnessPlan.model_validate_json(text).model_dump()
        except Exception:
            continue
        eq_ok += int(respects_equipment(plan, p["equipment"]))
        inj_ok += int(avoids_injury(plan, p["injury"]))
    n = len(rows)
    return {"label": label, "valid_json_rate": valid_json_rate(outputs),
            "equipment_satisfaction": eq_ok / n, "injury_safety": inj_ok / n}


def main(base_generate, tuned_generate):
    results = [evaluate(base_generate, "base"), evaluate(tuned_generate, "fine-tuned")]
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

# In Colab: define base_generate(profile)->str and tuned_generate(profile)->str
# using each loaded model, then call main(base_generate, tuned_generate).
