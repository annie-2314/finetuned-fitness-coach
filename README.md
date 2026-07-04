# 🏋️ Adaptive AI Fitness Coach

A **fine-tuned LLM** that generates structured, injury-aware workout + nutrition plans —
wrapped in a **full-stack app** that logs workouts and **adapts the plan each week**.

The point of the project is to demonstrate **model fine-tuning and rigorous evaluation**
(QLoRA · SFT → DPO), then ship it end-to-end behind a swappable serving layer.

## 🎯 Headline result
Fine-tuned **Qwen2.5-7B** (QLoRA, SFT→DPO) → **97% valid-schema output** and **84% overall
accuracy** (valid + constraint-compliant + injury-safe) on a held-out eval set, after
diagnosing a **train/serve prompt skew** (schema 0%→97%) and a generation-truncation issue.

See the numbers in [`docs/RESULTS.md`](docs/RESULTS.md), the full write-up (with code) in
[`implementation.md`](implementation.md), and the **executed** eval in
[`training/03_evaluation_results.ipynb`](training/03_evaluation_results.ipynb).

## 🧠 The fine-tuning (in one paragraph)
QLoRA (PEFT) fine-tune of Qwen2.5-7B-Instruct in 4-bit: freeze the base, train rank-16 LoRA
adapters (**0.53% of params**) in two stages — **SFT** (imitate ideal JSON plans; loss
0.92→0.09) then **DPO** (prefer safe over reckless plans; RLHF-free). Data is real-grounded
synthetic (free-exercise-db + Mifflin-St Jeor), schema-validated. Nutrition macros and exercise
images are **looked up from real sources**, never hallucinated.

## 📓 Notebooks (`training/`)
| Notebook | What it does |
|---|---|
| `01_finetune_qlora_sft_dpo.ipynb` | Fine-tune Qwen2.5-7B (QLoRA: SFT → DPO) |
| `02_evaluate.ipynb` | Evaluation harness (no-Unsloth, transformers+peft) |
| `03_evaluation_results.ipynb` | **Executed eval with the real 97% / 84% results** |
| `04_export_to_gguf.ipynb` | Export the adapter to GGUF for local serving (Ollama) |
| `sft_train.py` / `dpo_train.py` | Standalone training scripts |

## 🗂️ Repo structure
```
ai-fitness-coach/
├── training/     # fine-tuning + eval + export notebooks/scripts
├── src/          # core modules: schema, nutrition tool, exercise lookup, metrics, data gen
├── scripts/      # data download + dataset generation
├── eval/         # eval harness
├── backend/      # FastAPI + PostgreSQL + JWT + plan generation + adaptive loop
├── frontend/     # React + Vite + TypeScript + Tailwind (dashboard UI)
└── docs/         # spec, plan, results, Ollama & Colab/RunPod guides
```

## 🚀 Run the full-stack app
**1. Database** (once):
```
psql -U postgres -c "CREATE DATABASE fitness_coach;"
```
**2. Backend:**
```
cd backend
pip install -r requirements.txt
copy .env.example .env      # fill DATABASE_URL + LLM_API_KEY
uvicorn app.main:app --reload --port 8000
```
**3. Frontend:**
```
cd frontend
npm install
npm run dev                 # http://localhost:5173
```
Sign up → onboarding → **Generate my plan** → log workouts → **Adapt next week**.

## 🔌 Which model serves the plans? (swappable — one `.env` line)
The backend calls the model through an **OpenAI-compatible endpoint**, so it's provider-agnostic:
- **Hosted API (Groq/OpenRouter)** — default; free, instant, no GPU.
- **Your fine-tuned model via Ollama** — export to GGUF, run locally, set
  `LLM_BASE_URL=http://localhost:11434/v1`. See [`docs/OLLAMA_SETUP.md`](docs/OLLAMA_SETUP.md).
- **vLLM on a GPU** — for production.

## 🧰 Tech stack
**ML:** Qwen2.5-7B, QLoRA/PEFT, TRL (SFT/DPO), Unsloth, transformers, GGUF/Ollama.
**Backend:** FastAPI, SQLAlchemy 2, PostgreSQL, JWT (argon2).
**Frontend:** React 18, Vite, TypeScript, Tailwind, Recharts.

## 📝 Resume bullet
> Fine-tuned Qwen2.5-7B (QLoRA, SFT→DPO) into a structured fitness-coaching model with
> tool-grounded nutrition and injury-aware safety; built a held-out eval harness reaching
> **97% valid-schema output and 84% overall accuracy**, and diagnosed a train/serve prompt
> skew (schema 0%→97%). Shipped it in a full-stack app (FastAPI + PostgreSQL + React) with a
> swappable model-serving layer (Ollama / vLLM / hosted API).

_General fitness guidance, not medical advice._
