# Adaptive AI Fitness Coach — Design Spec (v1)

**Date:** 2026-07-02
**Author:** Annie Siri Kakumanu
**Type:** Personal portfolio project — fine-tuning showcase

---

## 1. Summary

A fine-tuned LLM that acts as a personal fitness coach: given a user's profile
(age, weight, goal, equipment, injuries, diet), it generates a structured
workout + nutrition plan with **accurate** macros (via tool-use) and
**safe** coaching (via preference alignment).

The point of the project is to demonstrate **model fine-tuning and rigorous
evaluation** — not just LLM orchestration. This fills the gap in an
orchestration-heavy resume by proving the ability to *train* and *evaluate* a model.

## 2. Goals / Non-goals

**Goals (v1)**
- Fine-tune an open model (Qwen2.5-7B) to generate JSON fitness plans.
- Teach tool-use so nutrition numbers are looked up, not hallucinated.
- Align for safety (injury-aware refusals) via DPO.
- Build an evaluation harness with measurable before→after metrics.
- Provide a thin API/demo to show it working.

**Deferred to later versions (still planned — just not in v1)**
- User accounts, database, workout logging → **v2**.
- Weekly adaptation loop → **v2**.
- Progress charts, on-demand swaps, polished UI, public deployment → **v3**.

_Note: these are out of scope for v1 only. They WILL be built, in v2 and v3,
after the v1 model + eval are working._

## 3. Approach (chosen)

- **Base model:** Qwen2.5-7B-Instruct (fallback: Llama-3.2-3B-Instruct if compute-limited).
- **Method:** QLoRA (4-bit, PEFT) via Unsloth — trains on a free Colab/Kaggle GPU.
- **Recipe:** two stages —
  1. **SFT** (supervised fine-tuning): learn JSON plan format + tool-use.
  2. **DPO** (direct preference optimization): learn safe/quality coaching.
- Rejected alternative: full RLHF (reward model + PPO) — too complex/unstable for the value.

## 4. Architecture

```
Real data (free-exercise-db + USDA FoodData Central / Recipe1M)
  → [scripts/] generate + curate → data/generated/sft.jsonl, dpo.jsonl
  → [training/] Qwen2.5-7B QLoRA: SFT → DPO → fine-tuned adapter
  → [eval/] metrics vs. base model (before→after)
  → [app/] FastAPI: profile in → model + nutrition tool → JSON plan out
```

## 5. Folder layout (`DB_CHECK/ai-fitness-coach/`)

```
ai-fitness-coach/
  docs/spec.md          # this file
  data/raw/             # downloaded real datasets
  data/generated/       # sft.jsonl, dpo.jsonl, eval.jsonl
  scripts/              # data download, generation, curation
  training/             # sft + dpo notebooks (run on Colab/Kaggle)
  eval/                 # eval harness + metrics
  app/                  # FastAPI serving + nutrition tool
  README.md
```

## 6. Data pipeline

- **Sources (real, open):** free-exercise-db (public domain exercises),
  USDA FoodData Central (macros, free API), Recipe1M (meals).
- **Method:** load real facts → prompt a free LLM (Groq / OpenRouter / Ollama)
  to synthesize diverse examples grounded in those facts → curate/filter.
- **Diversity axes:** age, goal (cut/bulk/maintain), equipment (gym/home/none),
  injuries, diet (veg/vegan/keto), experience level.
- **Output format:** structured JSON plans (chat-format JSONL). Includes a
  `daily_schedule` block (wake, meal times + focus, workout time, sleep window)
  so beginners know *when* to eat/train/sleep — general wellness guidance, not
  clinical prescription.
- **Beginner mode:** an experience flag; for beginners the model adds short
  "why" explanations to plan items (content feature, learned via SFT).
- **Exercise demo media:** each exercise carries a real image/GIF URL. The model
  only names the exercise; the app looks up the verified media from the exercise
  dataset (free-exercise-db / ExerciseDB) — the model NEVER generates URLs
  (avoids hallucinated/broken links). Same "ground in real data" pattern as the
  nutrition tool. URL lookup = v1; rich visual display = v3.
- **Sizes (v1):** ~1,200 SFT examples, ~400 DPO pairs, ~150 held-out eval examples.
- **Process:** generate a ~50-example pilot first → user spot-checks → then scale.
- **Principle:** quality + diversity over volume (LIMA principle).

## 7. Fine-tuning

- **Stage 1 — SFT:** chat-format examples (system/user/assistant, + tool role for
  tool-use). Teaches valid JSON output and nutrition-tool calls.
- **Stage 2 — DPO:** preference pairs (prompt / chosen / rejected). Rejected =
  base-model weaker output or deliberately unsafe answer. Teaches safe, correct coaching.
- **Tooling:** Unsloth + TRL (SFTTrainer, DPOTrainer) + PEFT/LoRA + bitsandbytes.
- **Compute:** free Kaggle (30h/week) or Colab T4.

## 8. Evaluation harness (the resume deliverable)

Measured on the held-out set, **base vs. fine-tuned**:
- Valid-JSON rate (schema-conformant output %)
- Macro accuracy (vs. USDA ground truth)
- Tool-call accuracy (correct function + args)
- Constraint-satisfaction % (respects equipment/diet/days)
- Injury-safety: contraindication-avoidance %
- DPO win-rate via LLM-as-judge (fine-tuned vs. base)

## 9. Serving (thin demo)

- FastAPI endpoint: user profile → model (+ USDA nutrition tool) → JSON plan.
- Model served from Colab/Kaggle GPU for demo, or exported to GGUF and run
  locally via Ollama. (Decision deferred to build time; does not affect training.)
- Unit tests: nutrition tool + JSON schema validation.

## 10. Cost

- v1 is **free**: open model, free GPU (Kaggle/Colab), open data, free LLM API
  tiers for data generation, local app.
- Optional paid (skippable): higher-quality data via paid API (~$2–10),
  faster training via RunPod, always-on public deploy.

## 11. Version roadmap

- **v1** — model + core plan generation + eval + thin demo (this spec). Resume-ready.
- **v2** — user DB, workout logging, weekly adaptation loop.
- **v3** — progress charts, on-demand swaps, Next.js UI, public deployment,
  and a **visual daily-schedule timeline** (wake → meals → workout → sleep),
  especially helpful for beginners.

## 12. Risks & mitigations

- **Free GPU slow / disconnects** → use Kaggle (30h/week); 3B fallback.
- **Low-quality synthetic data** → real-data grounding + curation + pilot check.
- **Model too big to serve locally** → quantized GGUF via Ollama, or serve from Colab.
- **"Toy" perception** → lead with the ML (QLoRA SFT→DPO + eval), not "fitness".

## 13. Success criteria (v1)

- Fine-tuned model beats the base model on the eval metrics (measurable gap).
- Valid-JSON rate near 100%, macros accurate via tool-use, unsafe requests refused.
- A working demo that turns a profile into a structured plan.
- A resume bullet backed by real numbers.

## 14. Prior art (cite in interviews)

- **Purrfessor** (arXiv 2411.14925) — fine-tuned diet chatbot on USDA + Recipe1M.
- **FormCoach** (arXiv 2508.07501) — LLM strength coaching.
- **LIMA** — "Less Is More for Alignment" (data-quality-over-quantity principle).


Task 1: Project scaffold + git init + deps

Task 2: Plan JSON schema (TDD)

Task 3: USDA nutrition tool (TDD)

Task 4: Exercise DB + media lookup (TDD)

Task 5: Download real datasets script

Task 6: SFT data generation + profiles (TDD)

Task 7: Curation + media enrichment (TDD)

Task 8: DPO pair generation

Task 9: Held-out eval set builder

Task 10: QLoRA SFT training script (Colab)

Task 11: QLoRA DPO training script (Colab)

Task 12: Eval metrics (TDD)

Task 13: Eval harness runner

Task 14: FastAPI demo app (TDD)

Task 15: Results documentation