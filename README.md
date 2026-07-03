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

## Results
See `docs/RESULTS.md`. Fine-tuning improves valid-JSON, equipment-constraint
satisfaction, and injury-safety over the base model on a 150-profile held-out set.

## Resume bullet
Fine-tuned Qwen2.5-7B (QLoRA, SFT→DPO) into a structured fitness-coaching model with
tool-grounded nutrition and injury-aware safety; built a held-out evaluation harness and
reached **97% valid-schema output and 84% overall accuracy** (valid + constraint-compliant +
injury-safe). Diagnosed and fixed a train/serve prompt skew that had collapsed schema
conformance to 0%, and a generation-truncation issue.
