# v1 Results — fine-tuned Qwen2.5-7B (QLoRA, SFT→DPO)

Base model: Qwen2.5-7B-Instruct (4-bit). Trained on RunPod (RTX 3090, Unsloth).
Held-out eval set (distinct seed 424242, no training overlap).

## FINAL — SFT model, correct inference (system prompt + 2500 tokens)
Held-out profiles (37 evaluated):

| Metric | Score |
|---|---|
| **Valid JSON** | **97%** |
| **Schema match (exact format)** | **97%** |
| **Equipment-constraint satisfaction** | **97%** |
| **Injury safety (avoids contraindicated)** | **84%** |
| **OVERALL ACCURACY (all checks pass)** | **31/37 = 84%** |

**Training signal:** SFT loss fell **0.92 → 0.09**. QLoRA trained **40.4M of 7.66B params = 0.53%**.

## The journey to that result (diagnostic story)
| Condition | Valid JSON | Schema |
|---|---|---|
| SFT+DPO, no system prompt, 1200 tok | 65% | 0% |
| SFT-only, no system prompt, 1200 tok | 48% | 0% |
| SFT-only, **+system prompt**, 1200 tok | 38% | 38% |
| SFT-only, **+system prompt + 2500 tok** | **97%** | **97%** |

Two bugs were diagnosed and fixed via ablation:
1. **Train/serve prompt skew** — training used a system prompt that defined the schema; the
   eval/serving harness dropped it, collapsing schema conformance to 0%. Evaluated SFT-only vs
   SFT+DPO to prove DPO wasn't the cause, then restored the system prompt → schema recovered.
2. **Truncation** — long multi-day plans exceeded the 1200-token generation cap and were cut
   off mid-JSON. Raising to 2500 tokens → valid-JSON 38% → 97%.

Both fixes are now baked into `app/inference.py` (system prompt always included; 2500 tokens).

## Method (resume/interview summary)
QLoRA (PEFT, r=16) fine-tune of Qwen2.5-7B-Instruct (4-bit), stages SFT → DPO, with
tool-grounded nutrition (USDA) and a real-data-grounded dataset (free-exercise-db +
Mifflin-St Jeor; ~600 SFT / ~200 DPO). Built a held-out eval harness (no leakage) measuring
valid-JSON, schema-conformance, equipment-constraint satisfaction, and injury-safety.
Diagnosed and fixed a train/serve prompt skew and a truncation issue to reach 97% valid-schema
output and 84% overall accuracy.

## Notes
- Free-Colab eval (bitsandbytes 4-bit, no Unsloth) is much slower for generation than the
  RunPod 3090 + Unsloth — a serving-stack lesson, not a model-quality issue.
- Model artifacts: `sft-adapter` (the shipped model) and `dpo-adapter` (comparison).
