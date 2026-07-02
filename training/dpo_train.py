# Run on Colab/Kaggle GPU. Continues from the SFT adapter.
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

model, tok = FastLanguageModel.from_pretrained(
    "outputs/sft-adapter", max_seq_length=4096, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"])

ds = load_dataset("json", data_files="data/generated/dpo.jsonl", split="train")
# TRL expects columns: prompt, chosen, rejected (already the case)

trainer = DPOTrainer(
    model=model, tokenizer=tok, train_dataset=ds, beta=0.1,
    args=DPOConfig(
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_train_epochs=1, learning_rate=5e-5, logging_steps=10,
        output_dir="outputs/dpo", optim="adamw_8bit",
        max_length=4096, max_prompt_length=1024))
trainer.train()
model.save_pretrained("outputs/dpo-adapter")
tok.save_pretrained("outputs/dpo-adapter")
print("DPO done -> outputs/dpo-adapter")
