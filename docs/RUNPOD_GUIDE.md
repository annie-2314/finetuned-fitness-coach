# RunPod Training Guide — Qwen2.5-7B (Unsloth)

Trains the full 7B model. RunPod's GPU images have matched torch/CUDA, so Unsloth
installs cleanly (none of the Colab torchao issues).

## 1. Deploy a pod
1. runpod.io → **Pods → Deploy**.
2. **GPU:** pick **RTX 3090 (24 GB)** or **RTX A5000 (24 GB)** (~$0.22–0.35/hr). 24 GB is plenty for 7B.
3. **Template:** search **"Unsloth"** and select it if present (Unsloth pre-installed).
   If not, choose **"RunPod PyTorch 2.x"** (we'll pip-install Unsloth in Cell 1).
4. **Container/Volume disk:** ~30 GB.
5. **Deploy**, wait ~1–2 min until it's *Running*.
6. Click **Connect → Jupyter Lab** (HTTP 8888). Open a new **Notebook (Python 3)**.

## 2. Upload the data (Jupyter drag-drop)
In Jupyter Lab's left file browser, **drag these 3 files** from your PC into the working folder:
- `sft.jsonl`, `dpo.jsonl`, `eval.jsonl`
(from `C:\Users\TEAMAPEX-003\Downloads\DB_CHECK\ai-fitness-coach\data\generated\`)

Then run each cell below in order.

---

### Cell 1 — (skip if template already has Unsloth) install
```python
!pip install -q unsloth
```

### Cell 2 — confirm GPU + files
```python
import torch, os
print("GPU:", torch.cuda.get_device_name(0))
print("jsonl files here:", [f for f in os.listdir(".") if f.endswith(".jsonl")])
```
*Should list sft.jsonl, dpo.jsonl, eval.jsonl.*

### Cell 3 — SFT training (7B, ~15–25 min)
```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAXLEN = 4096

model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=MAXLEN, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

ds = load_dataset("json", data_files="sft.jsonl", split="train")
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
```

### Cell 4 — DPO training (~10–20 min)
```python
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

dpo_ds = load_dataset("json", data_files="dpo.jsonl", split="train")

trainer = DPOTrainer(
    model=model, tokenizer=tok, train_dataset=dpo_ds,
    args=DPOConfig(beta=0.1, per_device_train_batch_size=1,
        gradient_accumulation_steps=4, num_train_epochs=1, learning_rate=5e-5,
        logging_steps=10, output_dir="outputs/dpo", optim="adamw_8bit",
        max_length=4096, max_prompt_length=1024))
trainer.train()
model.save_pretrained("outputs/dpo-adapter"); tok.save_pretrained("outputs/dpo-adapter")
print("DPO done")
```

### Cell 5 — quick test
```python
from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)
msgs = [{"role":"user","content":"35yo, 80kg, lose fat, home dumbbells, knee pain, 3 days/week. Return the JSON plan."}]
ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
out = model.generate(input_ids=ids, max_new_tokens=1200, temperature=0.7)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

### Cell 6 — evaluate (resume numbers)
```python
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

rows = [json.loads(l) for l in open("eval.jsonl")][:100]
valid = eq = inj = 0
for r in rows:
    p = r["profile"]; plan = extract_json(gen(p))
    if plan is None: continue
    valid += 1
    eq += int(respects_equipment(plan, p["equipment"]))
    inj += int(avoids_injury(plan, p["injury"]))
n = len(rows)
print(f"Fine-tuned Qwen2.5-7B on {n} held-out profiles:")
print(f"  valid JSON:             {valid}/{n} = {valid/n:.0%}")
print(f"  equipment satisfaction: {eq}/{n} = {eq/n:.0%}")
print(f"  injury safety:          {inj}/{n} = {inj/n:.0%}")
```

### Cell 7 — save the model
```python
# Zip the adapter, then download it from the Jupyter file browser (right-click > Download)
!zip -qr dpo-adapter.zip outputs/dpo-adapter
print("Download outputs/../dpo-adapter.zip from the Jupyter file browser.")
# Optional: push to Hugging Face
# from huggingface_hub import login; login()   # paste HF token
# model.push_to_hub("your-username/fitness-coach-qwen2.5-7b")
```

---

## Notes
- **Stop the pod when done** (RunPod → Pods → Stop/Terminate) so you're not billed for idle time.
- Total training ≈ 25–45 min on a 3090.
- If a TRL/Unsloth arg errors, paste it to me — but on RunPod's matched env this is rare.
- After Cell 6, **paste the numbers to me** and I'll fill `docs/RESULTS.md`.
