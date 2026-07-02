# v1 Results — base vs. fine-tuned

_Fill the numbers from `eval/results.json` after running training + eval on Colab._

| Metric | Base Qwen2.5-7B | Fine-tuned (SFT→DPO) |
|---|---|---|
| Valid-JSON rate | <fill> | <fill> |
| Equipment satisfaction | <fill> | <fill> |
| Injury safety | <fill> | <fill> |

**Method:** QLoRA (PEFT), SFT then DPO, ~1,200 SFT / ~400 DPO examples,
grounded in free-exercise-db + USDA FoodData Central. Held-out eval: 150 profiles
(distinct seed, no training overlap).
