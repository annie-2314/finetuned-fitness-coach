# Colab Training Guide — turn the dataset into a fine-tuned model

You'll do this in **Google Colab** (free T4 GPU). You only need to upload **2 files**:
`data/generated/sft.jsonl` and `data/generated/dpo.jsonl` (from your project folder).

## 0. Set up Colab
1. Go to https://colab.research.google.com → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save.**

Then paste each block below into its own cell and run in order.

---

### Cell 1 — Install
```python
!pip install -q unsloth trl peft transformers datasets bitsandbytes accelerate
```

### Cell 2 — Upload the two data files
```python
import os
from google.colab import files
os.makedirs("data/generated", exist_ok=True)
print("Select sft.jsonl AND dpo.jsonl from your PC:")
up = files.upload()
for name in up:
    os.replace(name, f"data/generated/{name}")
print(os.listdir("data/generated"))
```

### Cell 3 — SFT training (stage 1)
```python
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
```

### Cell 4 — DPO training (stage 2)
```python
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
```

### Cell 5 — Quick sanity test
```python
FastLanguageModel.for_inference(model)
msgs = [{"role":"user","content":"35yo, 80kg, lose fat, home dumbbells, knee pain, 3 days/week. JSON plan."}]
ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
out = model.generate(input_ids=ids, max_new_tokens=1200, temperature=0.7)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

### Cell 6 — Save the model (download or push to HF)
```python
# Option A: zip + download the adapter to your PC
!zip -r dpo-adapter.zip outputs/dpo-adapter
from google.colab import files; files.download("dpo-adapter.zip")

# Option B (optional): push to your Hugging Face account
# from huggingface_hub import login; login()  # paste your HF token
# model.push_to_hub("your-username/fitness-coach-qwen2.5-7b")
```

---

## Notes
- **Time:** SFT ~15–40 min, DPO ~15–30 min on a free T4 (with 600/200 examples).
- **If you hit out-of-memory or it's too slow:** change `MODEL` in Cell 3 to the 3B
  fallback (`unsloth/Llama-3.2-3B-Instruct-bnb-4bit`) and re-run.
- **Library API drift:** Unsloth/TRL update often. If a cell errors on an argument,
  copy the error to me and I'll give you the exact fix — this is the one step where
  we may need to iterate together.
- After training, we run the **eval** (base vs fine-tuned) to get the numbers for
  `docs/RESULTS.md`.
