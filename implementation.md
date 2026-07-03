# Implementation Log — Adaptive AI Fitness Coach (v1)

> A running **technical** record: what was built, and *how each part works step by step*.
> Updated as work progresses.

**Project:** Fine-tune Qwen2.5-7B (QLoRA, SFT→DPO) into a fitness coach that outputs
structured JSON plans with tool-grounded macros and safe coaching.
**Method stack:** QLoRA (PEFT) · SFT · DPO · tool-use (nutrition/media lookup).
**Docs:** `docs/spec.md` (design), `docs/plans/2026-07-02-fitness-coach-v1.md` (plan).

---

## End-to-end flow — what we actually did (step by step)

1. **Scoped the project** — chose a fine-tuning *showcase* (fitness coach that outputs
   structured JSON plans) to prove model-training skill; wrote `docs/spec.md`.
2. **Built v1 code with TDD** — JSON schema, USDA nutrition tool, exercise-DB media lookup,
   curation, eval metrics, FastAPI app, training + eval scripts. **22 unit tests, all green.**
3. **Built the dataset** — first tried LLM generation via Groq → hit the **free-tier daily
   token limit** → pivoted to a **keyless generator** grounded in real data (free-exercise-db +
   Mifflin-St Jeor macros). Produced **600 SFT + 200 DPO + 150 held-out eval**, all validated.
4. **Trained the model** — Colab failed with **Unsloth/torchao dependency conflicts** → moved
   to **RunPod (RTX 3090)** → QLoRA **SFT** (loss 0.92 → 0.09), then **DPO**. Both adapters saved.
5. **Evaluated (round 1)** — SFT+DPO scored **0% schema match** → looked broken.
6. **Diagnosed via ablation** — evaluated **SFT-only vs SFT+DPO**; both 0% → **DPO wasn't the
   cause.** Real bug: a **train/serve prompt skew** — the eval dropped the system prompt the
   model was trained with.
7. **Fix 1 (system prompt)** — restored it at inference → schema **0% → 38%**; revealed a
   second issue: **truncation** (long plans cut off at 1200 tokens).
8. **Fix 2 (token limit)** — raised generation to **2500 tokens** → valid-JSON **38% → 97%**.
9. **Final eval** — ran on **free Colab** with a **no-Unsloth** stack (transformers + peft +
   bitsandbytes) to avoid the earlier dependency hell → **97% valid / 97% schema / 84% overall
   accuracy** on held-out profiles. (The evaluated/shipped model is the **SFT checkpoint**;
   DPO was implemented but not the finalized one.)
10. **Hardened serving** — baked both fixes into `app/inference.py` (system prompt always
    included; 2500-token generation). Recorded results in `docs/RESULTS.md` + resume bullet.

**Key lessons:** environment/dependency matching matters (Colab vs RunPod); train/serve
parity is critical (the system prompt); low training loss ≠ good output (held-out eval + the
truncation catch); ablation isolates the real cause.

## Environment

- **OS:** Windows 11 · **Python:** 3.12.10 · **Git:** 2.53 (local repo, branch `main`, no remote).
- **Local deps** (`requirements.txt`): pydantic 2, requests, python-dotenv, openai, datasets,
  fastapi, uvicorn, httpx, truststore, pytest.
- **Training deps** (Colab/Kaggle only): unsloth, trl, peft, transformers, bitsandbytes.
- **Tests:** `pytest -q` (config `pytest.ini`, `pythonpath = .`). 16/16 passing.

### Deviations from the plan (and why)
1. **`pytest.ini` with `pythonpath = .`** — so `from src...` imports resolve for both pytest
   and `python -m scripts.*`.
2. **`truststore` + `src/ssl_setup.py`** — the machine is behind a corporate SSL-inspecting
   proxy; Python's bundled CA list rejected GitHub (`CERTIFICATE_VERIFY_FAILED`). truststore
   uses the Windows trust store (which trusts the proxy CA). Chosen over `verify=False`.
3. **Scripts run as modules** (`python -m scripts.download_data`) so package imports work.
4. **`reps` int→str coercion (bug fix)** — the LLM emits `reps` as a number (`12`) or a
   range string (`"10-12"`). The strict `reps: str` schema rejected 100% of the first pilot
   (all 50 dropped). Added a Pydantic `field_validator(mode="before")` that normalizes reps
   to `str`. After the fix the pilot passed 20/20.

## Run log
- **Pilot #1 (n=50):** 0/50 valid — uncovered the `reps` type bug (see deviation 4).
- **Pilot #2 (n=20, post-fix):** 20/20 valid (100%). Sample plan verified: correct
  schedule/meals/sleep timing, sets/reps/rest, macros, disclaimer; `why` populated only for
  beginners (as designed).
- **Scale decision:** lean first pass — **600 SFT + 200 DPO + 150 eval** (LIMA principle;
  safer on Groq free-tier daily limits; expandable later). Generation running in background.
- **Data-gen provider:** Groq `llama-3.3-70b-versatile` via OpenAI-compatible client;
  single-call latency ~1s, full-plan generation ~3–4s each through the proxy.
- **Groq free-tier limit hit:** during the 600-run we exhausted Groq's **daily**
  token cap (TPD 100,000 — "Used 99,492"); only 4 SFT / 0 DPO were written before
  every call returned HTTP 429. LLM generation is blocked until the daily reset.
- **Pivot → keyless generator** (`src/local_gen.py`, `scripts/generate_local.py`):
  deterministic, real-data-grounded (free-exercise-db + Mifflin-St Jeor macros),
  no API/keys/limits. **Generated 600 SFT + 200 DPO + 150 eval, all validated:**
  600/600 SFT schema-valid; DPO chosen-plan avoids contraindications 154/154 on
  injury cases (rejected plans include them → clean safety signal).
  *Optional later:* when Groq resets, mix in LLM examples for language variety.
- **Test suite:** 22 passing.

---

## How each part works (technical walkthrough)

### 1. Fetching the exercise data — `scripts/download_data.py`
Goal: pull the real exercise database onto disk.
1. **Source:** free-exercise-db is a public GitHub repo; its file is served raw at
   `https://raw.githubusercontent.com/yuhonas/free-exercise-db/main/dist/exercises.json`.
2. **HTTP GET:** `requests.get(url, timeout=60)` opens an HTTPS connection and downloads the bytes.
3. **SSL through the proxy:** the corporate proxy re-signs HTTPS with the company's own cert,
   which Python's default CA list doesn't trust → handshake failed. `truststore.inject_into_ssl()`
   (called via `enable_os_trust_store()`) switches Python to the **Windows cert store**, which
   trusts that cert → handshake succeeds.
4. **`resp.raise_for_status()`** ensures HTTP 200 (else raise).
5. **Write:** `out.write_bytes(resp.content)` saves to `data/raw/exercises.json`.
6. **Verify:** `len(resp.json())` → **873 exercises** confirmed.
So fetching = open HTTPS → download bytes → check status → write file, with the trust-store tweak.

### 2. Nutrition lookup (the "tool") — `src/nutrition.py`
Goal: accurate macros, not model guesses.
1. **API:** USDA FoodData Central REST endpoint
   `GET /fdc/v1/foods/search?query=<food>&api_key=<key>&pageSize=1`.
2. **Call:** `session.get(URL, params=...)`; USDA returns JSON with a `foods` list, each having
   a `foodNutrients` array.
3. **Parse:** scan `foodNutrients` for `nutrientName == "Energy"`/`unitName == "KCAL"` (calories)
   and `"Protein"`/`"G"`. USDA values are **per 100 g**.
4. **Scale:** multiply by `grams / 100`. E.g. chicken 165 kcal/100 g → 200 g = 330 kcal.
5. **Return** `{food, grams, calories, protein_g}`.
6. **Role:** the model is fine-tuned to *call this* rather than invent numbers (grounding).

### 3. Exercise media lookup — `src/exercise_db.py`
1. **Load once:** parse `exercises.json` into a dict `{name.lower(): record}` → O(1),
   case-insensitive lookups.
2. **Lookup:** returns `{name, demo_image, primary_muscles, equipment}`.
3. **Image URL:** the dataset stores a relative path (`GobletSquat/0.jpg`); we prepend the
   GitHub raw base to form a **real, verified** URL.
4. **Rule:** the model never writes URLs — the app attaches them from here (no broken links).

### 4. User profiles — `src/profiles.py`
1. **Deterministic:** `random.Random(seed)` → same seed = same profiles (reproducible).
   The eval set uses a *different* seed (424242) so it never overlaps training.
2. **Diverse:** random combos of age, weight, goal, equipment, diet, experience, injury,
   days/week → the model generalizes instead of memorizing one profile type.

### 5. SFT data generation — `scripts/generate_sft.py`
1. **Client:** `OpenAI(base_url, api_key)` — an OpenAI-*compatible* client pointed at
   **Groq/OpenRouter** (free tiers).
2. **Prompts:** a strict **system prompt** (output only schema JSON, respect injuries,
   `demo_image=null`, `why` for beginners, include disclaimer) + a **user prompt** = one profile.
3. **JSON mode:** `response_format={"type": "json_object"}` forces valid JSON out.
4. **Quality gate:** each response is validated with `FitnessPlan.model_validate_json(...)`;
   invalid ones are dropped (curation at generation time).
5. **Output:** kept examples written as **chat-format JSONL** — `{"messages": [system, user, assistant]}`,
   the exact shape the trainer consumes.

### 6. DPO pair generation — `scripts/generate_dpo.py`
1. For each profile, generate **two** responses: `chosen` from a *safe, injury-aware* system
   prompt, `rejected` from a *reckless* system prompt.
2. Write `{prompt, chosen, rejected}` JSONL — the preference signal DPO learns from.

### 7. Held-out eval set — `scripts/make_eval_set.py`
1. 150 profiles with seed **424242** (distinct from training seeds 0/99 → no leakage).
2. Store USDA ground-truth nutrition per profile for macro-accuracy checks.

### 8. Fine-tuning — `training/sft_train.py`, `training/dpo_train.py`
**SFT (stage 1):**
1. **Unsloth** loads `Qwen2.5-7B-Instruct` in **4-bit** (quantized → fits a free GPU).
2. **LoRA** (`get_peft_model`, rank 16) injects small trainable adapters on attention+MLP
   projections; only these (~1% of params) train — this is QLoRA.
3. **`SFTTrainer`** applies the chat template and trains via next-token prediction
   (cross-entropy) for 2 epochs → `outputs/sft-adapter`.

**DPO (stage 2):**
1. Loads the SFT adapter, trains with **`DPOTrainer`** on the preference pairs (beta 0.1).
2. DPO's loss raises the probability of `chosen` over `rejected` → learns safe/correct
   coaching → `outputs/dpo-adapter`.

### 9. Eval harness — `eval/run_eval.py`
1. Runs **both** base and fine-tuned models over the 150 held-out profiles.
2. Computes: valid-JSON rate, equipment-satisfaction, injury-safety (via `src/metrics.py`).
3. Writes `eval/results.json` (the before→after numbers for the resume).

### 10. Metrics — `src/metrics.py`
- `valid_json_rate`: fraction of outputs that parse as JSON.
- `macro_close`: is a predicted macro within tolerance of USDA truth.
- `respects_equipment`: bodyweight-only plans must not require machines/barbells.
- `avoids_injury`: checks exercises against a per-injury contraindication table.

### 11. Serving app — `app/main.py`, `app/inference.py`
1. **`create_app(generate_fn, exercises_path)`** builds FastAPI with the model injected
   (so tests can pass a stub — no GPU needed).
2. **`POST /plan`:** profile in → `generate_fn` → **validate** against schema (HTTP 422 if
   invalid) → **enrich** exercises with real `demo_image` → return JSON.
3. `app/inference.py` loads the trained adapter for real serving (Unsloth inference).

### 12. Testing approach (TDD)
For each module: write the test first → run it to confirm it *fails* → implement → confirm it
*passes*. Network is faked (stub `session`/`generate_fn`) so tests are offline and fast.
Result: **16 tests passing.**

---

## Training & evaluation walkthrough (RunPod, what each step does + its output)

Training was run on **RunPod (RTX 3090, 24 GB)** via `training/fitness_coach_runpod.ipynb`,
after Colab's Unsloth/torchao dependency conflicts made it unusable.

- **Cell 1 — GPU + files check.** Prints the GPU name and lists the 3 `.jsonl` files.
  *Output:* `GPU: NVIDIA GeForce RTX 3090` + `['sft.jsonl','dpo.jsonl','eval.jsonl']`.
- **Cell 2 — install Unsloth.** On RunPod's matched torch/CUDA image this installs cleanly.
- **Cell 3 — SFT (stage 1).** Unsloth loads Qwen2.5-7B in 4-bit, attaches LoRA adapters, and
  trains on the 600 SFT examples (2 epochs, 150 steps).
  *Output:* `Trainable parameters = 40,370,176 of 7,655,986,688 (0.53%)` (proves QLoRA — only
  0.53% of weights train), a falling training loss, and `SFT done`.
  *Result:* loss fell **0.92 → 0.09** — the model learned the plan format well.
- **Cell 4 — DPO (stage 2).** Continues from the SFT model, training on the 200 chosen/rejected
  pairs so it prefers safe, injury-aware plans. *Output:* a loss line then `DPO done`.
  *Gotcha:* a "kernel Idle but cell `[*]`" state means the cell was queued/interrupted, not
  running — fix by Stop then re-running **only Cell 4** (never Restart Kernel — the in-memory
  SFT model would be lost; it is, however, saved on disk at `outputs/sft-adapter`).
- **Cell 5 — quick test.** Generates a plan for a knee-pain user. *Observation:* the plan
  contained **no squats/lunges** (safety learned ✅). Using an abbreviated prompt here (not the
  training format) made the model improvise a different JSON shape — expected for out-of-format prompts.
- **Cell 6 — evaluation.** Runs the model over 100 held-out profiles (distinct seed) using the
  exact training-format prompt, and reports valid-JSON %, schema-match %, equipment-satisfaction %,
  and injury-safety %. The repeated `max_new_tokens/max_length` lines are harmless (one per
  generation). The metric functions were hardened to tolerate inconsistent output shapes
  (`'str' object has no attribute 'get'` came from assuming a fixed structure).
- **Cell 7 — save.** Zips `outputs/dpo-adapter` for download (or push to HF Hub).

### Observed model-quality note
The fine-tuned model sometimes drifts from the exact `FitnessPlan` schema (e.g. a `day1/day2`
shape with `workouts`/`exercise` instead of `weekly_workouts`/`exercises`/`name`). Likely causes:
the keyless training data is highly uniform/templated, and/or DPO over-optimization. If schema-match
comes out low, the fix is one of: reduce DPO strength (lower `beta`/epochs), add more varied SFT
data, or constrain decoding to the JSON schema at serve time.

## Task status

| Task | Component | Status |
|---|---|---|
| 1 | Scaffold + git | ✅ done |
| 2 | JSON schema (TDD) | ✅ done |
| 3 | Nutrition tool (TDD) | ✅ done |
| 4 | Exercise DB (TDD) | ✅ done |
| 5 | Dataset downloader | ✅ done (873 exercises fetched) |
| 6 | SFT generation + profiles | ✅ code / ⏳ run needs LLM key |
| 7 | Curation + media (TDD) | ✅ done |
| 8 | DPO generation | ✅ code / ⏳ run needs LLM key |
| 9 | Eval set | ✅ code / ⏳ run needs USDA key |
| 10 | QLoRA SFT (Colab) | ✅ code / ⏳ run on GPU |
| 11 | QLoRA DPO (Colab) | ✅ code / ⏳ run on GPU |
| 12 | Metrics (TDD) | ✅ done |
| 13 | Eval harness | ✅ code / ⏳ run on GPU |
| 14 | FastAPI app (TDD) | ✅ done |
| 15 | Results docs | ✅ done (numbers pending training) |

**Overall:** v1 code complete, 16/16 tests passing, 16 commits on local `main`.

## How to resume
1. `cp .env.example .env` → add Groq/OpenRouter + USDA keys.
2. `python -m scripts.generate_sft --n 50` (pilot) → check → `--n 1200`.
3. `python -m scripts.generate_dpo --n 400`; `python -m scripts.make_eval_set`.
4. Colab/Kaggle: run `training/sft_train.py` → `training/dpo_train.py`.
5. Run eval → fill numbers into `docs/RESULTS.md`. (Then optional GitHub push.)
