# Adaptive AI Fitness Coach

Fine-tuned (QLoRA, SFT→DPO) fitness coach that outputs structured JSON plans.
See `docs/spec.md` for the design and `docs/plans/` for the build plan.

## Setup
1. `pip install -r requirements.txt`
2. `cp .env.example .env` and fill in keys.
3. Run tests: `pytest -q`

## Build order (scripts run as modules from the project root)
1. `python -m scripts.download_data` — fetch real exercise data
2. `python -m scripts.generate_sft --n 50` (pilot) → spot-check → `--n 1200`
3. `python -m scripts.generate_dpo --n 400`
4. `python -m scripts.make_eval_set`
5. On Colab/Kaggle GPU: run `training/sft_train.py`, then `training/dpo_train.py`
6. On Colab: run the eval harness (`eval/run_eval.py`) → `eval/results.json`
7. Serve locally: build the FastAPI app with the trained model (see `app/`)

## Full-stack app (v2/v3)
- `backend/` — FastAPI + PostgreSQL + JWT auth + plan generation + adaptive loop
- `frontend/` — React + Vite + TypeScript + Tailwind (auth, onboarding, plan view with
  exercise images, workout logging, progress, "adapt next week")

### Run it
**1. Database:** create the Postgres DB (once):
```
psql -U postgres -c "CREATE DATABASE fitness_coach;"
```
**2. Backend:**
```
cd backend
python -m venv venv && venv\Scripts\activate      # optional
pip install -r requirements.txt
copy .env.example .env    # then fill DATABASE_URL + LLM_API_KEY + USDA_API_KEY
uvicorn app.main:app --reload --port 8000
```
**3. Frontend:**
```
cd frontend
npm install
npm run dev            # http://localhost:5173
```
Open http://localhost:5173 → sign up → onboarding → generate a plan → log workouts → adapt.

### Which model generates the plans?
- **Default:** a hosted base model (Groq/OpenRouter) via `LLM_BASE_URL` + your system prompt.
- **Your fine-tuned model:** follow `docs/OLLAMA_SETUP.md` — export to GGUF, run in Ollama,
  and set `LLM_BASE_URL=http://localhost:11434/v1`. One `.env` change, no code edits.

## Results
See `docs/RESULTS.md`. Fine-tuning reached **97% valid-schema output and 84% overall accuracy**
on a held-out set (after fixing a train/serve prompt skew + a truncation issue).

## Resume bullet
Built a QLoRA fine-tuning pipeline (SFT → DPO) for Qwen2.5-7B into a structured
fitness-coaching model with tool-grounded nutrition and injury-aware safety. Via an
**ablation** identified the SFT checkpoint as the best model and, after fixing a train/serve
prompt skew (schema 0%→97%) and a generation-truncation issue, reached **97% valid-schema
output and 84% overall accuracy** on a held-out eval set.

_(Note: SFT and DPO were both implemented; the 97% figure is the evaluated/shipped SFT model.)_
