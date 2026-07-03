# Serve your fine-tuned model locally via Ollama

This makes the backend generate plans with **your fine-tuned model** (free, local),
instead of a hosted base model.

## 1. Export to GGUF (one-time, on a GPU — RunPod/Colab)
Load your saved SFT adapter and export a quantized GGUF:
```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("sft-adapter", max_seq_length=4096, load_in_4bit=True)
model.save_pretrained_gguf("fitness-coach", tok, quantization_method="q4_k_m")
# -> fitness-coach/*.Q4_K_M.gguf  (~4.5 GB). Download it to your PC.
```

## 2. Install Ollama
Download from https://ollama.com/download (Windows). It serves at http://localhost:11434.

## 3. Register the model
Place the `.gguf` next to `backend/Modelfile` (provided), then:
```
ollama create fitness-coach -f backend/Modelfile
ollama run fitness-coach "give me a quick test plan"   # verify
```
(Edit the `FROM` line in the Modelfile to match your actual .gguf filename.)

## 4. Point the backend at Ollama
In `backend/.env`:
```
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=fitness-coach
LLM_API_KEY=ollama
```
Restart the backend. Now `/plan/generate` and `/plan/adapt` use YOUR fine-tuned model.

## Notes
- 7B on CPU is slow (~1–3 min/plan) but free & local. For fast demos, host on RunPod+vLLM
  instead and point `LLM_BASE_URL` at that endpoint (same one-line `.env` change).
- If Ollama's OpenAI endpoint rejects `response_format`, the backend still works — the
  system prompt already forces JSON, and generation is validated + retried.
