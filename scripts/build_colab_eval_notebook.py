"""Builds a NO-UNSLOTH Colab eval notebook (transformers + peft + bitsandbytes).

Loads the downloaded SFT adapter on a free Colab T4 and evaluates at 2500 tokens.
Avoids Unsloth/torchao entirely (the source of Colab dependency errors).

Run: python -m scripts.build_colab_eval_notebook
Output: training/eval_colab_nounsloth.ipynb
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "training" / "eval_colab_nounsloth.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n").splitlines(keepends=True)}


cells = []

cells.append(md("""
# Free Colab eval (NO Unsloth) — SFT model at 2500 tokens

Runs on a free **T4 GPU** with plain transformers + peft (no Unsloth/torchao).

**Setup:** Runtime → Change runtime type → **T4 GPU**. Then run cells top to bottom.
You'll upload `sft-adapter.zip` and `eval.jsonl` in Cell 2.
"""))

cells.append(md("## 1. Install (Colab-stable, no Unsloth)"))
cells.append(code("!pip install -q -U transformers peft bitsandbytes accelerate"))

cells.append(md("## 2. Upload sft-adapter.zip + eval.jsonl, then unzip"))
cells.append(code("""
import os, zipfile
from google.colab import files
print("Select sft-adapter.zip AND eval.jsonl:")
up = files.upload()
if os.path.exists("sft-adapter.zip") and not os.path.exists("sft-adapter"):
    with zipfile.ZipFile("sft-adapter.zip") as z:
        z.extractall("sft-adapter")
print("adapter files:", os.listdir("sft-adapter") if os.path.exists("sft-adapter") else "MISSING")
print("eval.jsonl present:", os.path.exists("eval.jsonl"))
"""))

cells.append(md("## 3. Load base Qwen2.5-7B (4-bit) + your adapter"))
cells.append(code("""
import torch, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("transformers").setLevel(logging.ERROR)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
BASE = "Qwen/Qwen2.5-7B-Instruct"
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, "sft-adapter")
model.eval()
tok = AutoTokenizer.from_pretrained("sft-adapter")
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
print("model + adapter loaded on", model.device)
"""))

cells.append(md("## 4. Evaluate (2500 tokens) + OVERALL ACCURACY"))
cells.append(code("""
import json, torch

SYSTEM = ("You are an expert fitness coach. Given a user profile, output ONLY a JSON "
    "object matching this schema: goal, experience (beginner|intermediate|advanced), "
    "daily_schedule{wake, workout{time,type}, meals[{time,name,focus}], sleep{target,hours}}, "
    "weekly_workouts[{day, focus, exercises[{name,sets,reps,rest_seconds,demo_image,why}]}], "
    "nutrition{daily_macros{calories,protein_g,carbs_g,fat_g}, "
    "example_day[{food,grams,calories,protein_g}], grocery_list[]}, disclaimer. "
    "Use realistic exercises and macros. Set demo_image to null (filled later). "
    "For beginners, fill 'why' with a short reason. Always include a disclaimer that "
    "this is general guidance, not medical advice. If the user reports an injury, "
    "avoid contraindicated movements.")

CONTRA = {"knee pain":["squat","lunge","leg press"],
          "shoulder impingement":["overhead press","upright row","bench press"],
          "lower back pain":["deadlift","good morning","bent over row"]}

def extract_json(t):
    try: return json.loads(t)
    except Exception: pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try: return json.loads(t[i:j+1])
        except Exception: return None
    return None

def exercise_names(plan):
    names = []
    wk = plan.get("weekly_workouts")
    if isinstance(wk, list):
        for d in wk:
            if isinstance(d, dict):
                for e in d.get("exercises", []):
                    if isinstance(e, dict): names.append(str(e.get("name","")))
                    elif isinstance(e, str): names.append(e)
    for v in plan.values():
        if isinstance(v, dict) and isinstance(v.get("workouts"), list):
            for e in v["workouts"]:
                if isinstance(e, dict): names.append(str(e.get("exercise") or e.get("name","")))
    return [n.lower() for n in names if n]

def schema_ok(plan):
    wk = plan.get("weekly_workouts")
    if not (isinstance(wk, list) and wk and isinstance(wk[0], dict)): return False
    exs = wk[0].get("exercises")
    return isinstance(exs, list) and all(isinstance(e, dict) and "name" in e for e in exs) and "nutrition" in plan

def avoids_injury(names, injury):
    if not injury or injury == "None": return True
    return not any(b in n for n in names for b in CONTRA.get(injury, []))

def respects_equipment(names, eq):
    if eq != "bodyweight only": return True
    banned = ["barbell","machine","cable","leg press"]
    return not any(b in n for n in names for b in banned)

def gen(p):
    u = (f"Profile: {p['age']}yo, {p['weight_kg']}kg, goal: {p['goal']}, "
         f"equipment: {p['equipment']}, diet: {p['diet']}, experience: {p['experience']}, "
         f"injury: {p['injury']}, {p['days_per_week']} days/week. Return the JSON plan.")
    msgs = [{"role":"system","content":SYSTEM}, {"role":"user","content":u}]
    ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=2500, do_sample=True,
                             temperature=0.7, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

rows = [json.loads(l) for l in open("eval.jsonl")][:40]
valid = schema = eq = inj = passed = 0
for k, r in enumerate(rows, 1):
    p = r["profile"]; plan = extract_json(gen(p))
    if plan:
        valid += 1
        s = schema_ok(plan); names = exercise_names(plan)
        e = respects_equipment(names, p["equipment"]); i2 = avoids_injury(names, p["injury"])
        schema += int(s); eq += int(e); inj += int(i2)
        if s and e and i2: passed += 1
    if k % 10 == 0: print(f"...{k}/{len(rows)}")
n = len(rows)
print(f"\\nSFT model (no-Unsloth, 2500 tokens) on {n} held-out profiles:")
print(f"  valid JSON:             {valid}/{n} = {valid/n:.0%}")
print(f"  schema match:           {schema}/{n} = {schema/n:.0%}")
print(f"  equipment satisfaction: {eq}/{n} = {eq/n:.0%}")
print(f"  injury safety:          {inj}/{n} = {inj/n:.0%}")
print(f"\\n  >>> OVERALL ACCURACY (all checks pass): {passed}/{n} = {passed/n:.0%} <<<")
"""))

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "cells": cells}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}")
