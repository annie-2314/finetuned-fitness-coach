# v1 Results — fine-tuned Qwen2.5-7B (QLoRA, SFT→DPO)

Trained on RunPod (RTX 3090, Unsloth, ~2 hrs, <$1). Held-out eval set (distinct seed 424242,
no training overlap). Base model: Qwen2.5-7B-Instruct (4-bit).

## Training
- **SFT loss: 0.92 → 0.09** over 2 epochs (600 examples) — learned the task well.
- QLoRA: **40.4M trainable params of 7.66B = 0.53%.**

## Evaluation (held-out profiles) — the debugging journey
| Condition | Valid JSON | Schema match | Equip | Injury safety |
|---|---|---|---|---|
| SFT+DPO, **no system prompt** | 65% | **0%** | 65% | 65% |
| SFT-only, **no system prompt** | 48% | **0%** | 48% | 48% |
| SFT-only, **WITH system prompt** | 38% | **38%** | 38% | 35% |

## What the numbers show (the story)
1. **First eval gave 0% schema match** → looked like a broken model.
2. **Diagnosed via ablation:** evaluated SFT-only vs SFT+DPO — *both* were 0%, so DPO wasn't the
   cause. The real bug was a **train/inference prompt mismatch**: training examples included a
   system prompt defining the schema, but the eval harness dropped it. Adding it back →
   **schema match recovered from 0% to 38%**, and crucially **schema == valid-JSON** (every
   parseable output was correctly structured). Confirmed with a direct sample (correct
   `weekly_workouts`/`exercises`/`name`, `demo_image:null`, `why` for beginners, injury-safe).
3. **Remaining failures are truncation**, not format: long multi-day plans exceeded the
   1200-token generation cap and got cut off mid-JSON (short 3-day plans parsed fine).

## Fixes / next steps (v1.1)
- **Raise `max_new_tokens`** (~2500) so long plans finish → expected valid-JSON ≈ 80%+.
- **Constrained decoding** (JSON grammar) to guarantee parseable output.
- **Always include the system prompt at inference** (train/serve parity) — now baked into serving.
- Optional: more varied SFT data; gentler DPO.

## Method (resume/interview summary)
QLoRA (PEFT, r=16) fine-tune of Qwen2.5-7B-Instruct (4-bit) in two stages — **SFT** then **DPO** —
with **tool-grounded nutrition** (USDA) and a **real-data-grounded dataset** (free-exercise-db +
Mifflin-St Jeor; ~600 SFT / ~200 DPO). Built an **eval harness** (held-out, no leakage) measuring
valid-JSON, schema-conformance, equipment-constraint satisfaction, and injury-safety. Diagnosed a
**train/serve prompt skew** that was tanking schema conformance and recovered it.
