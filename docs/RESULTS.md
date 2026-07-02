# v1 Results — fine-tuned Qwen2.5-7B (QLoRA, SFT→DPO)

Trained on RunPod (RTX 3090, Unsloth). Held-out eval set (distinct seed, no training overlap).

## SFT→DPO model (40 held-out profiles)
| Metric | Score |
|---|---|
| Valid JSON | 26/40 = **65%** |
| Schema match (our exact format) | 0/40 = **0%** |
| Equipment-constraint satisfaction | 26/40 = **65%** |
| Injury safety (avoids contraindicated) | 26/40 = **65%** |

**Training signal:** SFT loss fell **0.92 → 0.09** (learned the task well).

## Interpretation
- The model produces plausible, **injury-safe** fitness plans (every valid plan avoided
  contraindicated exercises).
- **But it drifts from the target JSON schema** (0% exact-schema match; it emits a
  `day1/day2 + workouts/exercise` shape instead of `weekly_workouts/exercises/name`).
- Root cause (hypothesis): **DPO over-optimization degraded format adherence.** SFT loss of
  0.09 implies the SFT model matched the schema; DPO likely pushed it off-format.

## Next diagnostic
Evaluate the **SFT-only** adapter (before DPO) with the same harness. Expectation:
substantially higher schema-match, confirming DPO caused the drift.

## Fixes if confirmed
1. Ship the **SFT-only** model (if its schema-match is high). OR
2. **Redo DPO more gently** (lower `beta`, fewer steps, lower LR). OR
3. **Constrained decoding** at inference (enforce the JSON schema via a grammar). OR
4. **More varied SFT data** (the keyless data is uniform/templated).

## Method (for the resume/interview)
QLoRA (PEFT, r=16) on Qwen2.5-7B-Instruct (4-bit), two stages: SFT then DPO, with tool-grounded
nutrition (USDA) and a real-data-grounded dataset (~600 SFT / ~200 DPO). Evaluated on a
held-out set (no leakage) with objective metrics (valid-JSON, schema, constraint, safety).
