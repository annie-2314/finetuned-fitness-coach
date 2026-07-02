"""Builds a self-contained Colab notebook for training the fitness coach.

Run: python -m scripts.build_notebook
Output: training/fitness_coach_colab.ipynb  (upload this to Colab)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "training" / "fitness_coach_colab.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


cells = []

cells.append(md("""
# Adaptive AI Fitness Coach — Training (Colab)

**Before running:** Runtime → Change runtime type → **T4 GPU** → Save.
Then run every cell top-to-bottom. Do **not** change the runtime type after this
(it wipes everything). If you must reset, use Runtime → **Restart session**.
"""))

cells.append(md("## 1. Check GPU"))
cells.append(code("""
import torch
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), \\
    "No GPU! Runtime > Change runtime type > T4 GPU, then re-run this cell."
print("GPU:", torch.cuda.get_device_name(0))
"""))

cells.append(md("## 2. Install Unsloth (~3-5 min)"))
cells.append(code("""
!pip install -q unsloth
print("install done")
"""))

cells.append(md("## 3. Upload data files (sft.jsonl, dpo.jsonl, eval.jsonl)"))
cells.append(code("""
import os, re, shutil
from google.colab import files
os.makedirs("data/generated", exist_ok=True)
print("Select sft.jsonl, dpo.jsonl, eval.jsonl from your PC:")
up = files.upload()
for raw in list(up.keys()):
    clean = re.sub(r"\\s*\\(\\d+\\)(?=\\.)", "", raw)   # 'sft (1).jsonl' -> 'sft.jsonl'
    shutil.move(raw, f"data/generated/{clean}")
print("In data/generated:", sorted(os.listdir("data/generated")))
"""))

cells.append(md("## 4. SFT training (stage 1, ~15-40 min)"))
cells.append(code("""
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"   # fallback: unsloth/Llama-3.2-3B-Instruct-bnb-4bit
MAXLEN = 4096

model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=MAXLEN, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

ds = load_dataset("json", data_files="data/generated/sft.jsonl", split="train")
ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)})

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds,
    args=SFTConfig(dataset_text_field="text", max_seq_length=MAXLEN,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=2, learning_rate=2e-4, logging_steps=10,
        output_dir="outputs/sft", optim="adamw_8bit"))
trainer.train()
model.save_pretrained("outputs/sft-adapter"); tok.save_pretrained("outputs/sft-adapter")
print("SFT done")
"""))

cells.append(md("## 5. DPO training (stage 2, ~15-30 min)"))
cells.append(code("""
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

dpo_ds = load_dataset("json", data_files="data/generated/dpo.jsonl", split="train")

trainer = DPOTrainer(
    model=model, tokenizer=tok, train_dataset=dpo_ds,
    args=DPOConfig(beta=0.1, per_device_train_batch_size=1,
        gradient_accumulation_steps=4, num_train_epochs=1, learning_rate=5e-5,
        logging_steps=10, output_dir="outputs/dpo", optim="adamw_8bit",
        max_length=4096, max_prompt_length=1024))
trainer.train()
model.save_pretrained("outputs/dpo-adapter"); tok.save_pretrained("outputs/dpo-adapter")
print("DPO done")
"""))

cells.append(md("## 6. Quick test — generate a plan for an injured user"))
cells.append(code("""
from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)
msgs = [{"role":"user","content":"35yo, 80kg, lose fat, home dumbbells, knee pain, 3 days/week. Return the JSON plan."}]
ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
out = model.generate(input_ids=ids, max_new_tokens=1200, temperature=0.7)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
"""))

cells.append(md("## 7. Evaluate the fine-tuned model (numbers for your resume)"))
cells.append(code("""
import json
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

def avoids_injury(plan, injury):
    if not injury or injury == "None": return True
    bad = CONTRA.get(injury, [])
    for d in plan.get("weekly_workouts", []):
        for e in d.get("exercises", []):
            if any(b in e.get("name","").lower() for b in bad): return False
    return True

def respects_equipment(plan, eq):
    if eq != "bodyweight only": return True
    banned = ["barbell","machine","cable","leg press"]
    for d in plan.get("weekly_workouts", []):
        for e in d.get("exercises", []):
            if any(b in e.get("name","").lower() for b in banned): return False
    return True

def gen(p):
    u = (f"Profile: {p['age']}yo, {p['weight_kg']}kg, goal: {p['goal']}, "
         f"equipment: {p['equipment']}, diet: {p['diet']}, experience: {p['experience']}, "
         f"injury: {p['injury']}, {p['days_per_week']} days/week. Return the JSON plan.")
    ids = tok.apply_chat_template([{"role":"user","content":u}], return_tensors="pt",
                                  add_generation_prompt=True).to(model.device)
    out = model.generate(input_ids=ids, max_new_tokens=1200, temperature=0.7)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

rows = [json.loads(l) for l in open("data/generated/eval.jsonl")][:50]  # 50 for speed; raise to 150 for full
valid = eq = inj = 0
for r in rows:
    p = r["profile"]; plan = extract_json(gen(p))
    if plan is None: continue
    valid += 1
    eq += int(respects_equipment(plan, p["equipment"]))
    inj += int(avoids_injury(plan, p["injury"]))
n = len(rows)
print(f"Fine-tuned model on {n} held-out profiles:")
print(f"  valid JSON:             {valid}/{n} = {valid/n:.0%}")
print(f"  equipment satisfaction: {eq}/{n} = {eq/n:.0%}")
print(f"  injury safety:          {inj}/{n} = {inj/n:.0%}")
print("\\nPaste these numbers to Claude to fill docs/RESULTS.md")
"""))

cells.append(md("## 8. Save the model (download to your PC)"))
cells.append(code("""
!zip -qr dpo-adapter.zip outputs/dpo-adapter
from google.colab import files
files.download("dpo-adapter.zip")
# Optional: push to your Hugging Face account
# from huggingface_hub import login; login()   # paste HF token
# model.push_to_hub("your-username/fitness-coach-qwen2.5-7b")
"""))

notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {OUT}")
