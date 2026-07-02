# Run on Colab/Kaggle GPU: pip install unsloth trl peft transformers datasets
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"   # fallback: unsloth/Llama-3.2-3B-Instruct-bnb-4bit
MAXLEN = 4096

model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=MAXLEN, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"])

ds = load_dataset("json", data_files="data/generated/sft.jsonl", split="train")


def fmt(ex):
    return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}


ds = ds.map(fmt)

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds,
    args=SFTConfig(
        dataset_text_field="text", max_seq_length=MAXLEN,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=2, learning_rate=2e-4, logging_steps=10,
        output_dir="outputs/sft", optim="adamw_8bit"))
trainer.train()
model.save_pretrained("outputs/sft-adapter")
tok.save_pretrained("outputs/sft-adapter")
print("SFT done -> outputs/sft-adapter")
