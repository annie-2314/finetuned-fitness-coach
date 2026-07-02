# Implementation Log — Adaptive AI Fitness Coach (v1)

> A running technical record of **what** was built, **how**, and **why**.
> Updated as work progresses. Newest status at the bottom.

**Project:** Fine-tune Qwen2.5-7B (QLoRA, SFT→DPO) into a fitness coach that
outputs structured JSON plans with tool-grounded macros and safe coaching.
**Method stack:** QLoRA (PEFT) · SFT · DPO · tool-use (nutrition/media lookup).
**Related docs:** `docs/spec.md` (design), `docs/plans/2026-07-02-fitness-coach-v1.md` (plan).

---

## Environment

- **OS:** Windows 11 · **Python:** 3.12.10 · **Git:** 2.53 (local repo, branch `main`, no remote).
- **Local deps** (`requirements.txt`): pydantic 2, requests, python-dotenv, openai,
  datasets, fastapi, uvicorn, httpx, truststore, pytest.
- **Training deps** (Colab/Kaggle only): unsloth, trl, peft, transformers, bitsandbytes.
- **Test runner:** `pytest -q` (config in `pytest.ini`, `pythonpath = .`).

### Deviations from the plan (with reasons)
1. **Added `pytest.ini`** (`pythonpath = .`) — the plan's `from src...` imports need the
   project root on `sys.path`; this makes `pytest` and `python -m scripts.*` resolve cleanly.
2. **Added `truststore` + `src/ssl_setup.py`** — this machine sits behind a corporate
   SSL-inspection proxy, so Python's bundled CA list rejected `raw.githubusercontent.com`
   (`CERTIFICATE_VERIFY_FAILED`). `truststore.inject_into_ssl()` delegates to the Windows
   trust store (which trusts the proxy CA). Chosen over `verify=False` to avoid disabling security.
3. **Scripts run as modules** (`python -m scripts.download_data`) instead of
   `python scripts/download_data.py`, so package-relative `src` imports work.

---

## Architecture (data → train → eval → serve)

```
free-exercise-db + USDA  →  scripts/  →  data/generated/{sft,dpo,eval}.jsonl
                          →  training/ (Colab) QLoRA SFT → DPO → adapter
                          →  eval/ metrics vs base model
                          →  app/ FastAPI: profile → model + nutrition tool → JSON plan
```

Design principle enforced everywhere: **the model never generates URLs or nutrition
numbers** — those come from real-data lookups (exercise DB / USDA), avoiding hallucination.

---

## Task-by-task record

### Task 1 — Project scaffold  ✅
- **What:** Folder layout, `requirements.txt`, `.env.example`, `.gitignore`, `README.md`,
  `pytest.ini`, package `__init__.py`s, `data/` dirs; `git init` on `main`.
- **How:** Wrote config files; `git config` set to Annie's identity; first commit.
- **Commit:** `chore: scaffold ai-fitness-coach project`.

### Task 2 — Plan JSON schema (TDD)  ✅
- **What:** Pydantic v2 models in `src/schema.py`: `FitnessPlan` with `daily_schedule`
  (wake/meals/workout/sleep timing), `weekly_workouts` (exercises w/ `demo_image`, `why`),
  `nutrition` (macros + example day + grocery list), `experience` enum, `disclaimer`.
- **How:** TDD — `tests/test_schema.py` (valid parse, missing-field raises, bad-enum raises)
  written first, confirmed failing, then implemented. 3 tests pass.
- **Why this shape:** JSON output = auto-validatable + easy to eval and render; timing fields
  and `why` support beginners; `disclaimer` enforces the "not medical advice" safety rule.
- **Commit:** `feat: add FitnessPlan JSON schema`.

### Task 3 — USDA nutrition tool (TDD)  ✅
- **What:** `src/nutrition.py::lookup_nutrition(food, grams, api_key, session)` — queries
  USDA FoodData Central, scales per-100g values to requested grams, returns
  `{food, grams, calories, protein_g}`.
- **How:** TDD with an injected fake `session` (no network in tests): verifies per-100g
  scaling and zero-fallback for missing foods. 2 tests pass.
- **Why:** This is the **tool-use** grounding — real macros instead of hallucinated numbers.
- **Commit:** `feat: add USDA nutrition lookup tool`.

### Task 4 — Exercise DB + media lookup (TDD)  ✅
- **What:** `src/exercise_db.py::ExerciseDB.lookup(name)` — loads free-exercise-db JSON,
  case-insensitive lookup, returns real demo-image URL + muscles + equipment.
- **How:** TDD against `tests/fixtures/exercises_sample.json` (exact, case-insensitive,
  missing→None). 3 tests pass.
- **Why:** Supplies **verified** exercise media so the model only names exercises (no fake links).
- **Commit:** `feat: add exercise DB lookup with media`.

### Task 5 — Real dataset downloader  ✅ (executed)
- **What:** `scripts/download_data.py` fetches free-exercise-db (`dist/exercises.json`).
- **How:** `requests` + `enable_os_trust_store()`; run as module. **Ran successfully →
  873 exercises saved to `data/raw/exercises.json`.**
- **Commit:** `feat: real dataset downloader + OS trust-store SSL helper`.

### Task 6 — SFT data generation + profiles (TDD)  ✅ (code) / ⏳ (run needs API key)
- **What:** `src/profiles.py::sample_profiles(n, seed)` (diverse, deterministic user profiles);
  `scripts/generate_sft.py` prompts the LLM to emit schema-valid JSON plans, drops invalid ones,
  writes chat-format JSONL.
- **How:** TDD for profile sampling (determinism + diversity, 1 test pass); generation script
  syntax-verified. Pilot (`--n 50`) → scale (`--n 1200`) **pending your `LLM_API_KEY`**.
- **Commit:** `feat: SFT data generation grounded in profiles + schema`.

### Task 7 — Curation + media enrichment (TDD)  ✅
- **What:** `src/curate.py`: `validate_example` (drops non-schema-valid records),
  `enrich_media` (fills `demo_image` from the exercise DB).
- **How:** TDD (valid accepted, bad JSON rejected, media filled). 3 tests pass.
- **Commit:** `feat: add curation + media enrichment`.

### Task 8 — DPO pair generation  ✅ (code) / ⏳ (run needs API key)
- **What:** `scripts/generate_dpo.py` — generates `{prompt, chosen, rejected}` pairs;
  `chosen` = safe/injury-aware coach, `rejected` = reckless coach (for safety contrast).
- **How:** Syntax-verified; run pending `LLM_API_KEY`.
- **Commit:** `feat: DPO preference pair generation`.

### Task 9 — Held-out eval set  ✅ (code) / ⏳ (run needs USDA key)
- **What:** `scripts/make_eval_set.py` — 150 profiles with a **distinct seed (424242)** so
  there is no overlap with training data; stores nutrition ground-truth from USDA.
- **How:** Syntax-verified; run pending `USDA_API_KEY`.
- **Commit:** `feat: build held-out eval set`.

### Task 10 — QLoRA SFT training (Colab)  ✅ (code) / ⏳ (run on GPU)
- **What:** `training/sft_train.py` — Unsloth loads `Qwen2.5-7B-Instruct` 4-bit, LoRA
  (r=16) on attention+MLP projections, SFTTrainer over `sft.jsonl` (2 epochs, lr 2e-4).
- **How:** Syntax-verified. Runs on free Colab/Kaggle T4; 3B fallback documented.
- **Commit:** `feat: QLoRA SFT training script (Colab)`.

### Task 11 — QLoRA DPO training (Colab)  ✅ (code) / ⏳ (run on GPU)
- **What:** `training/dpo_train.py` — continues from the SFT adapter, DPOTrainer over
  `dpo.jsonl` (beta 0.1, 1 epoch, lr 5e-5) → `outputs/dpo-adapter`.
- **How:** Syntax-verified. Runs after SFT on GPU.
- **Commit:** `feat: QLoRA DPO training script (Colab)`.

### Task 12 — Eval metrics (TDD)  ✅
- **What:** `src/metrics.py`: `valid_json_rate`, `macro_close`, `respects_equipment`,
  `avoids_injury` (contraindication table per injury).
- **How:** TDD (json rate, macro tolerance, injury contraindication). 3 tests pass.
- **Commit:** `feat: add eval metric functions`.

### Task 13 — Eval harness (base vs fine-tuned)  ✅ (code) / ⏳ (run on GPU)
- **What:** `eval/run_eval.py::main(base_generate, tuned_generate)` — runs both models over
  the held-out set, computes valid-JSON / equipment-satisfaction / injury-safety, writes
  `eval/results.json`.
- **How:** Syntax-verified. Executed in Colab where both models are loaded.
- **Commit:** `feat: eval harness comparing base vs fine-tuned`.

### Task 14 — FastAPI demo app (TDD)  ✅
- **What:** `app/main.py::create_app(generate_fn, exercises_path)` exposes `POST /plan`:
  validates model output against the schema (422 on invalid) and enriches media.
  `app/inference.py` loads the trained model for real serving.
- **How:** TDD with a **stub generator** (no GPU needed) via FastAPI `TestClient`; asserts a
  200 response with media-enriched output. 1 test pass.
- **Commit:** `feat: FastAPI /plan endpoint with schema validation + media enrichment`.

### Task 15 — Results documentation  ✅
- **What:** `docs/RESULTS.md` (metrics table, numbers to fill post-training) + README build
  order + resume bullet.
- **Commit:** `docs: v1 results, build order, and resume bullet`.

---

## Current status

- **Code:** v1 complete. **Tests:** 16/16 passing. **Commits:** 15 on local `main`.
- **Executed:** real exercise data downloaded (873 items); all unit tests.
- **Pending (needs you):** fill `.env` keys → run data generation → run SFT+DPO on Colab →
  run eval → fill `docs/RESULTS.md` numbers. GitHub push optional (repo is local-only).

## How to resume
See README "Build order". Everything after data generation needs your API keys (Groq/OpenRouter
+ USDA) and a free Colab/Kaggle GPU for training/eval.
