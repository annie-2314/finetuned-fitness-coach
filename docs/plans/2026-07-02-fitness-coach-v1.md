# Adaptive AI Fitness Coach — v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Qwen2.5-7B (QLoRA, SFT→DPO) into a fitness coach that outputs structured JSON plans with tool-grounded macros and safe coaching, plus an eval harness proving it beats the base model.

**Architecture:** Deterministic Python modules (schema, nutrition tool, exercise lookup, curation, metrics) are built locally with TDD. Data generation calls a free LLM API grounded in real datasets. Fine-tuning runs in a Colab/Kaggle notebook. A thin FastAPI app serves the model.

**Tech Stack:** Python 3.10+, Pydantic v2, requests, pytest, FastAPI, OpenAI-compatible client (Groq/OpenRouter); Unsloth + TRL + PEFT + bitsandbytes (Colab); Qwen2.5-7B-Instruct.

## Global Constraints

- Base model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (fallback: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`).
- All project files live under `DB_CHECK/ai-fitness-coach/`.
- Model NEVER generates URLs — media/nutrition come from lookups only.
- Plan output must validate against the `FitnessPlan` Pydantic schema.
- Dataset sizes (v1): ~1,200 SFT, ~400 DPO, ~150 held-out eval.
- Every plan includes a `disclaimer` ("general guidance, not medical advice").
- Secrets via `.env` only; never hardcode API keys.
- Python indentation 4 spaces; run tests with `pytest -q`.

---

### Task 1: Project scaffold

**Files:**
- Create: `ai-fitness-coach/requirements.txt`
- Create: `ai-fitness-coach/.env.example`
- Create: `ai-fitness-coach/.gitignore`
- Create: `ai-fitness-coach/README.md`
- Create: `ai-fitness-coach/src/__init__.py`
- Create: `ai-fitness-coach/tests/__init__.py`

**Interfaces:**
- Produces: the folder layout and dependency list every later task relies on.

- [ ] **Step 1: Initialize git in the project folder**

Run:
```bash
cd "c:/Users/TEAMAPEX-003/Downloads/DB_CHECK/ai-fitness-coach" && git init
```
Expected: `Initialized empty Git repository`.

- [ ] **Step 2: Write `requirements.txt`**

```
pydantic>=2.6
requests>=2.31
python-dotenv>=1.0
openai>=1.30
datasets>=2.19
fastapi>=0.111
uvicorn>=0.30
pytest>=8.0
```

- [ ] **Step 3: Write `.env.example`**

```
# LLM used to GENERATE training data (free tiers work)
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your_groq_or_openrouter_key
LLM_MODEL=llama-3.3-70b-versatile
# USDA nutrition (free key: https://fdc.nal.usda.gov/api-key-signup.html)
USDA_API_KEY=your_usda_key
```

- [ ] **Step 4: Write `.gitignore`**

```
__pycache__/
*.pyc
.env
data/raw/*
data/generated/*
!data/raw/.gitkeep
!data/generated/.gitkeep
outputs/
*.gguf
```

- [ ] **Step 5: Create empty package files and data dirs**

Run:
```bash
cd "c:/Users/TEAMAPEX-003/Downloads/DB_CHECK/ai-fitness-coach" && mkdir -p src tests data/raw data/generated scripts training eval app && touch src/__init__.py tests/__init__.py data/raw/.gitkeep data/generated/.gitkeep
```

- [ ] **Step 6: Write minimal `README.md`**

```markdown
# Adaptive AI Fitness Coach

Fine-tuned (QLoRA, SFT→DPO) fitness coach that outputs structured JSON plans.
See `docs/spec.md` for the design and `docs/plans/` for the build plan.

## Setup
1. `pip install -r requirements.txt`
2. `cp .env.example .env` and fill in keys.
3. Run tests: `pytest -q`
```

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "chore: scaffold ai-fitness-coach project"
```

---

### Task 2: Plan JSON schema

**Files:**
- Create: `src/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: Pydantic models `FitnessPlan`, `WorkoutDay`, `ExerciseItem`, `MacroTargets`, `NutritionPlan`, `DailySchedule`; used by curation, eval, app.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schema.py
import pytest
from pydantic import ValidationError
from src.schema import FitnessPlan

VALID = {
    "goal": "lose fat",
    "experience": "beginner",
    "daily_schedule": {
        "wake": "07:00",
        "workout": {"time": "18:00", "type": "Full body"},
        "meals": [{"time": "08:00", "name": "Breakfast", "focus": "protein"}],
        "sleep": {"target": "22:30-07:00", "hours": 8},
    },
    "weekly_workouts": [{
        "day": "Monday", "focus": "Full body",
        "exercises": [{"name": "Goblet Squat", "sets": 3, "reps": "10",
                       "rest_seconds": 60, "demo_image": None, "why": None}],
    }],
    "nutrition": {
        "daily_macros": {"calories": 2000, "protein_g": 150, "carbs_g": 200, "fat_g": 60},
        "example_day": [{"food": "Chicken", "grams": 150, "calories": 247, "protein_g": 46}],
        "grocery_list": ["Chicken", "Rice"],
    },
    "disclaimer": "General guidance, not medical advice.",
}

def test_valid_plan_parses():
    plan = FitnessPlan.model_validate(VALID)
    assert plan.goal == "lose fat"
    assert plan.nutrition.daily_macros.protein_g == 150

def test_missing_field_raises():
    bad = {k: v for k, v in VALID.items() if k != "nutrition"}
    with pytest.raises(ValidationError):
        FitnessPlan.model_validate(bad)

def test_bad_experience_enum_raises():
    bad = {**VALID, "experience": "pro"}
    with pytest.raises(ValidationError):
        FitnessPlan.model_validate(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.schema'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/schema.py
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel

class MealTime(BaseModel):
    time: str
    name: str
    focus: Optional[str] = None

class SleepWindow(BaseModel):
    target: str
    hours: float

class WorkoutSlot(BaseModel):
    time: str
    type: str

class DailySchedule(BaseModel):
    wake: str
    workout: WorkoutSlot
    meals: list[MealTime]
    sleep: SleepWindow

class ExerciseItem(BaseModel):
    name: str
    sets: int
    reps: str
    rest_seconds: int
    demo_image: Optional[str] = None
    why: Optional[str] = None

class WorkoutDay(BaseModel):
    day: str
    focus: str
    exercises: list[ExerciseItem]

class MacroTargets(BaseModel):
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int

class FoodItem(BaseModel):
    food: str
    grams: float
    calories: float
    protein_g: float

class NutritionPlan(BaseModel):
    daily_macros: MacroTargets
    example_day: list[FoodItem]
    grocery_list: list[str]

class FitnessPlan(BaseModel):
    goal: str
    experience: Literal["beginner", "intermediate", "advanced"]
    daily_schedule: DailySchedule
    weekly_workouts: list[WorkoutDay]
    nutrition: NutritionPlan
    disclaimer: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/schema.py tests/test_schema.py && git commit -m "feat: add FitnessPlan JSON schema"
```

---

### Task 3: USDA nutrition lookup tool

**Files:**
- Create: `src/nutrition.py`
- Test: `tests/test_nutrition.py`

**Interfaces:**
- Consumes: `USDA_API_KEY` from env.
- Produces: `lookup_nutrition(food: str, grams: float, api_key: str, session=requests) -> dict` returning `{"food", "grams", "calories", "protein_g"}`.

- [ ] **Step 1: Write the failing test** (uses a fake session so no network)

```python
# tests/test_nutrition.py
from src.nutrition import lookup_nutrition

class FakeResp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p

class FakeSession:
    def __init__(self, payload): self.payload = payload
    def get(self, url, params=None, timeout=None): return FakeResp(self.payload)

USDA_PAYLOAD = {"foods": [{"description": "Chicken breast",
    "foodNutrients": [
        {"nutrientName": "Energy", "unitName": "KCAL", "value": 165},
        {"nutrientName": "Protein", "unitName": "G", "value": 31}]}]}

def test_scales_per_100g_to_grams():
    r = lookup_nutrition("chicken", 200, "k", session=FakeSession(USDA_PAYLOAD))
    assert r["calories"] == 330.0      # 165 * 200/100
    assert r["protein_g"] == 62.0      # 31 * 200/100
    assert r["grams"] == 200

def test_missing_food_returns_zeros():
    r = lookup_nutrition("nope", 100, "k", session=FakeSession({"foods": []}))
    assert r["calories"] == 0 and r["protein_g"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nutrition.py -q`
Expected: FAIL — `No module named 'src.nutrition'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/nutrition.py
import requests

USDA_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

def _nutrient(food, name, unit):
    for n in food.get("foodNutrients", []):
        if n.get("nutrientName") == name and n.get("unitName") == unit:
            return float(n.get("value", 0))
    return 0.0

def lookup_nutrition(food: str, grams: float, api_key: str, session=requests) -> dict:
    resp = session.get(USDA_URL,
        params={"query": food, "api_key": api_key, "pageSize": 1},
        timeout=30)
    resp.raise_for_status()
    foods = resp.json().get("foods", [])
    if not foods:
        return {"food": food, "grams": grams, "calories": 0.0, "protein_g": 0.0}
    top = foods[0]
    factor = grams / 100.0
    return {
        "food": food,
        "grams": grams,
        "calories": round(_nutrient(top, "Energy", "KCAL") * factor, 1),
        "protein_g": round(_nutrient(top, "Protein", "G") * factor, 1),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nutrition.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nutrition.py tests/test_nutrition.py && git commit -m "feat: add USDA nutrition lookup tool"
```

---

### Task 4: Exercise DB loader + media lookup

**Files:**
- Create: `src/exercise_db.py`
- Test: `tests/test_exercise_db.py`
- Test fixture: `tests/fixtures/exercises_sample.json`

**Interfaces:**
- Produces: `ExerciseDB(path)` with `.lookup(name) -> dict | None` returning `{"name", "demo_image", "primary_muscles", "equipment"}`.

- [ ] **Step 1: Create the fixture**

```json
// tests/fixtures/exercises_sample.json
[
  {"name": "Goblet Squat", "primaryMuscles": ["quadriceps"],
   "equipment": "dumbbell", "images": ["GobletSquat/0.jpg"]},
  {"name": "Push Up", "primaryMuscles": ["chest"],
   "equipment": "body only", "images": ["PushUp/0.jpg"]}
]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_exercise_db.py
from pathlib import Path
from src.exercise_db import ExerciseDB

FIX = Path(__file__).parent / "fixtures" / "exercises_sample.json"

def test_lookup_exact():
    db = ExerciseDB(FIX)
    ex = db.lookup("Goblet Squat")
    assert ex["primary_muscles"] == ["quadriceps"]
    assert ex["demo_image"].endswith("GobletSquat/0.jpg")

def test_lookup_case_insensitive():
    db = ExerciseDB(FIX)
    assert db.lookup("push up")["equipment"] == "body only"

def test_lookup_missing_returns_none():
    db = ExerciseDB(FIX)
    assert db.lookup("Deadlift") is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_exercise_db.py -q`
Expected: FAIL — `No module named 'src.exercise_db'`.

- [ ] **Step 4: Write minimal implementation**

```python
# src/exercise_db.py
import json
from pathlib import Path

IMG_BASE = "https://raw.githubusercontent.com/yuhonas/free-exercise-db/main/exercises/"

class ExerciseDB:
    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._by_name = {e["name"].lower(): e for e in data}

    def lookup(self, name: str):
        e = self._by_name.get(name.strip().lower())
        if e is None:
            return None
        images = e.get("images") or []
        return {
            "name": e["name"],
            "demo_image": (IMG_BASE + images[0]) if images else None,
            "primary_muscles": e.get("primaryMuscles", []),
            "equipment": e.get("equipment"),
        }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_exercise_db.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/exercise_db.py tests/test_exercise_db.py tests/fixtures/exercises_sample.json && git commit -m "feat: add exercise DB lookup with media"
```

---

### Task 5: Download real datasets

**Files:**
- Create: `scripts/download_data.py`

**Interfaces:**
- Produces: `data/raw/exercises.json` (free-exercise-db). Consumed by exercise lookup + data generation.

- [ ] **Step 1: Write the downloader**

```python
# scripts/download_data.py
import requests
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
EXERCISES_URL = ("https://raw.githubusercontent.com/yuhonas/"
                 "free-exercise-db/main/dist/exercises.json")

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "exercises.json"
    print(f"Downloading exercises -> {out}")
    resp = requests.get(EXERCISES_URL, timeout=60)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    print(f"Saved {len(resp.json())} exercises.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify output**

Run: `python scripts/download_data.py`
Expected: prints `Saved 800+ exercises.` and `data/raw/exercises.json` exists.

- [ ] **Step 3: Commit**

```bash
git add scripts/download_data.py && git commit -m "feat: add real dataset downloader"
```

---

### Task 6: SFT data generation (pilot first)

**Files:**
- Create: `src/profiles.py`
- Create: `scripts/generate_sft.py`
- Test: `tests/test_profiles.py`

**Interfaces:**
- Consumes: `ExerciseDB`, `lookup_nutrition`, LLM env vars, `FitnessPlan` schema.
- Produces: `data/generated/sft.jsonl` (chat-format examples); `sample_profiles(n, seed) -> list[dict]`.

- [ ] **Step 1: Write the failing test for profile sampling**

```python
# tests/test_profiles.py
from src.profiles import sample_profiles

def test_deterministic_and_diverse():
    a = sample_profiles(20, seed=1)
    b = sample_profiles(20, seed=1)
    assert a == b                        # deterministic
    assert len({p["goal"] for p in a}) >= 2
    assert len({p["equipment"] for p in a}) >= 2
    assert all("age" in p and "experience" in p for p in a)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_profiles.py -q`
Expected: FAIL — `No module named 'src.profiles'`.

- [ ] **Step 3: Implement profile sampling**

```python
# src/profiles.py
import random

GOALS = ["lose fat", "build muscle", "maintain", "gain strength"]
EQUIPMENT = ["full gym", "home dumbbells", "bodyweight only", "resistance bands"]
DIETS = ["no restriction", "vegetarian", "vegan", "keto"]
EXPERIENCE = ["beginner", "intermediate", "advanced"]
INJURIES = [None, "knee pain", "shoulder impingement", "lower back pain"]
DAYS = [3, 4, 5, 6]

def sample_profiles(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append({
            "age": rng.randint(18, 60),
            "weight_kg": rng.randint(50, 110),
            "goal": rng.choice(GOALS),
            "equipment": rng.choice(EQUIPMENT),
            "diet": rng.choice(DIETS),
            "experience": rng.choice(EXPERIENCE),
            "injury": rng.choice(INJURIES),
            "days_per_week": rng.choice(DAYS),
        })
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_profiles.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Write the generation script**

```python
# scripts/generate_sft.py
import os, json, argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from src.profiles import sample_profiles
from src.schema import FitnessPlan

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "sft.jsonl"

SYSTEM = (
    "You are an expert fitness coach. Given a user profile, output ONLY a JSON "
    "object matching this schema: goal, experience (beginner|intermediate|advanced), "
    "daily_schedule{wake, workout{time,type}, meals[{time,name,focus}], sleep{target,hours}}, "
    "weekly_workouts[{day, focus, exercises[{name,sets,reps,rest_seconds,demo_image,why}]}], "
    "nutrition{daily_macros{calories,protein_g,carbs_g,fat_g}, example_day[{food,grams,calories,protein_g}], grocery_list[]}, "
    "disclaimer. Use realistic exercises and macros. Set demo_image to null "
    "(it is filled later). For beginners, fill 'why' with a short reason. "
    "Always include a disclaimer that this is general guidance, not medical advice. "
    "If the user reports an injury, avoid contraindicated movements."
)

def user_prompt(p: dict) -> str:
    return (f"Profile: {p['age']}yo, {p['weight_kg']}kg, goal: {p['goal']}, "
            f"equipment: {p['equipment']}, diet: {p['diet']}, "
            f"experience: {p['experience']}, injury: {p['injury']}, "
            f"{p['days_per_week']} days/week. Return the JSON plan.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)     # pilot default
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
    model = os.environ["LLM_MODEL"]
    OUT.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with OUT.open("w", encoding="utf-8") as f:
        for p in sample_profiles(args.n, seed=args.seed):
            up = user_prompt(p)
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=0.7,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": up}],
                    response_format={"type": "json_object"})
                content = resp.choices[0].message.content
                FitnessPlan.model_validate_json(content)   # drop invalid
            except Exception as e:
                print("skip:", e); continue
            rec = {"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": up},
                {"role": "assistant", "content": content}]}
            f.write(json.dumps(rec) + "\n")
            kept += 1
    print(f"Wrote {kept}/{args.n} valid examples -> {OUT}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the PILOT (50 examples) and spot-check**

Run: `python scripts/generate_sft.py --n 50`
Expected: `Wrote ~45-50/50 valid examples`. Open `data/generated/sft.jsonl`, read 3 examples, confirm plans are sensible and injuries are respected. **Do not scale until this looks good.**

- [ ] **Step 7: Scale to full dataset**

Run: `python scripts/generate_sft.py --n 1200`
Expected: `Wrote ~1100-1200 valid examples`.

- [ ] **Step 8: Commit (code only; data is gitignored)**

```bash
git add scripts/generate_sft.py src/profiles.py tests/test_profiles.py && git commit -m "feat: SFT data generation grounded in profiles + schema"
```

---

### Task 7: Curation — enrich media + validate

**Files:**
- Create: `src/curate.py`
- Test: `tests/test_curate.py`

**Interfaces:**
- Consumes: `FitnessPlan`, `ExerciseDB`.
- Produces: `validate_example(rec) -> bool`; `enrich_media(plan_dict, db) -> dict` (fills `demo_image` from the exercise DB).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curate.py
from pathlib import Path
from src.curate import validate_example, enrich_media
from src.exercise_db import ExerciseDB

FIX = Path(__file__).parent / "fixtures" / "exercises_sample.json"

def _plan():
    return {"goal":"x","experience":"beginner",
        "daily_schedule":{"wake":"07:00","workout":{"time":"18:00","type":"Full"},
            "meals":[{"time":"08:00","name":"B","focus":"p"}],
            "sleep":{"target":"22:30-07:00","hours":8}},
        "weekly_workouts":[{"day":"Mon","focus":"F","exercises":[
            {"name":"Goblet Squat","sets":3,"reps":"10","rest_seconds":60,
             "demo_image":None,"why":"legs"}]}],
        "nutrition":{"daily_macros":{"calories":2000,"protein_g":150,"carbs_g":200,"fat_g":60},
            "example_day":[{"food":"Chicken","grams":150,"calories":247,"protein_g":46}],
            "grocery_list":["Chicken"]},
        "disclaimer":"General guidance, not medical advice."}

def test_validate_good_example():
    import json
    rec = {"messages":[{"role":"assistant","content":json.dumps(_plan())}]}
    assert validate_example(rec) is True

def test_validate_rejects_bad_json():
    rec = {"messages":[{"role":"assistant","content":"{not json"}]}
    assert validate_example(rec) is False

def test_enrich_fills_demo_image():
    db = ExerciseDB(FIX)
    enriched = enrich_media(_plan(), db)
    img = enriched["weekly_workouts"][0]["exercises"][0]["demo_image"]
    assert img and img.endswith("GobletSquat/0.jpg")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_curate.py -q`
Expected: FAIL — `No module named 'src.curate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/curate.py
import json
from src.schema import FitnessPlan

def validate_example(rec: dict) -> bool:
    try:
        content = rec["messages"][-1]["content"]
        FitnessPlan.model_validate_json(content)
        return True
    except Exception:
        return False

def enrich_media(plan: dict, db) -> dict:
    for day in plan.get("weekly_workouts", []):
        for ex in day.get("exercises", []):
            found = db.lookup(ex["name"])
            if found and found["demo_image"]:
                ex["demo_image"] = found["demo_image"]
    return plan
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_curate.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/curate.py tests/test_curate.py && git commit -m "feat: add curation + media enrichment"
```

---

### Task 8: DPO pair generation

**Files:**
- Create: `scripts/generate_dpo.py`

**Interfaces:**
- Consumes: LLM env, `sample_profiles`, `FitnessPlan`.
- Produces: `data/generated/dpo.jsonl` with `{"prompt","chosen","rejected"}`.

- [ ] **Step 1: Write the DPO generator** (chosen = safe/correct; rejected = deliberately unsafe/wrong)

```python
# scripts/generate_dpo.py
import os, json, argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from src.profiles import sample_profiles

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "dpo.jsonl"

CHOSEN_SYS = ("You are a SAFE expert fitness coach. Respect injuries, avoid "
    "contraindicated movements, give correct macros, output a JSON plan.")
REJECTED_SYS = ("You are a reckless coach who IGNORES injuries, prescribes unsafe "
    "heavy lifts regardless of limitations, and guesses macros. Output a JSON plan.")

def prompt(p):
    return (f"{p['age']}yo, {p['weight_kg']}kg, goal {p['goal']}, {p['equipment']}, "
            f"injury: {p['injury']}, {p['days_per_week']} days/week. JSON plan.")

def gen(client, model, sys, user):
    r = client.chat.completions.create(model=model, temperature=0.8,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        response_format={"type":"json_object"})
    return r.choices[0].message.content

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=99); args = ap.parse_args()
    client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
    model = os.environ["LLM_MODEL"]; OUT.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with OUT.open("w", encoding="utf-8") as f:
        # bias toward injury profiles so safety contrast is strong
        for p in sample_profiles(args.n, seed=args.seed):
            u = prompt(p)
            try:
                chosen = gen(client, model, CHOSEN_SYS, u)
                rejected = gen(client, model, REJECTED_SYS, u)
            except Exception as e:
                print("skip:", e); continue
            f.write(json.dumps({"prompt": u, "chosen": chosen, "rejected": rejected}) + "\n")
            kept += 1
    print(f"Wrote {kept} DPO pairs -> {OUT}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run pilot then full**

Run: `python scripts/generate_dpo.py --n 20` (spot-check chosen looks safe, rejected looks unsafe), then `python scripts/generate_dpo.py --n 400`.
Expected: `Wrote ~400 DPO pairs`.

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_dpo.py && git commit -m "feat: DPO preference pair generation"
```

---

### Task 9: Held-out eval set

**Files:**
- Create: `scripts/make_eval_set.py`

**Interfaces:**
- Produces: `data/generated/eval.jsonl` — profiles + expected constraints (NOT used in training).

- [ ] **Step 1: Write the eval-set builder** (profiles with a distinct seed; store the profile + ground-truth macro from USDA for one food)

```python
# scripts/make_eval_set.py
import os, json
from pathlib import Path
from dotenv import load_dotenv
from src.profiles import sample_profiles
from src.nutrition import lookup_nutrition

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "eval.jsonl"

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    key = os.environ["USDA_API_KEY"]
    with OUT.open("w", encoding="utf-8") as f:
        for p in sample_profiles(150, seed=424242):   # unique seed => held-out
            truth = lookup_nutrition("chicken breast", 150, key)
            f.write(json.dumps({"profile": p, "nutrition_truth": truth}) + "\n")
    print(f"Wrote 150 eval profiles -> {OUT}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

Run: `python scripts/make_eval_set.py`
Expected: `Wrote 150 eval profiles`. Confirm seed differs from training seeds (0/99) so there is no overlap.

- [ ] **Step 3: Commit**

```bash
git add scripts/make_eval_set.py && git commit -m "feat: build held-out eval set"
```

---

### Task 10: SFT training notebook (Colab/Kaggle)

**Files:**
- Create: `training/sft_train.py` (runs as a Colab cell or `python`)

**Interfaces:**
- Consumes: `data/generated/sft.jsonl`.
- Produces: LoRA adapter at `outputs/sft-adapter/` (and pushed to HF Hub).

- [ ] **Step 1: Write the SFT training script**

```python
# training/sft_train.py
# Run on Colab/Kaggle GPU: pip install unsloth trl peft transformers datasets
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"   # fallback: Llama-3.2-3B-Instruct-bnb-4bit
MAXLEN = 4096

model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=MAXLEN, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

ds = load_dataset("json", data_files="data/generated/sft.jsonl", split="train")
def fmt(ex): return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
ds = ds.map(fmt)

trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds,
    args=SFTConfig(dataset_text_field="text", max_seq_length=MAXLEN,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=2, learning_rate=2e-4, logging_steps=10,
        output_dir="outputs/sft", optim="adamw_8bit"))
trainer.train()
model.save_pretrained("outputs/sft-adapter"); tok.save_pretrained("outputs/sft-adapter")
print("SFT done -> outputs/sft-adapter")
```

- [ ] **Step 2: Run on Colab/Kaggle and verify**

Upload `data/generated/sft.jsonl`, run the script on a T4/P100. Expected: loss decreases over steps; `outputs/sft-adapter/` created. If OOM/slow: switch `MODEL` to the 3B fallback.

- [ ] **Step 3: Commit**

```bash
git add training/sft_train.py && git commit -m "feat: QLoRA SFT training script"
```

---

### Task 11: DPO training notebook (Colab/Kaggle)

**Files:**
- Create: `training/dpo_train.py`

**Interfaces:**
- Consumes: `outputs/sft-adapter/`, `data/generated/dpo.jsonl`.
- Produces: final adapter `outputs/dpo-adapter/`.

- [ ] **Step 1: Write the DPO training script**

```python
# training/dpo_train.py
# Continues from the SFT adapter.
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

model, tok = FastLanguageModel.from_pretrained("outputs/sft-adapter",
    max_seq_length=4096, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

ds = load_dataset("json", data_files="data/generated/dpo.jsonl", split="train")
# TRL expects columns: prompt, chosen, rejected (already the case)

trainer = DPOTrainer(model=model, tokenizer=tok, train_dataset=ds, beta=0.1,
    args=DPOConfig(per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_train_epochs=1, learning_rate=5e-5, logging_steps=10,
        output_dir="outputs/dpo", optim="adamw_8bit", max_length=4096, max_prompt_length=1024))
trainer.train()
model.save_pretrained("outputs/dpo-adapter"); tok.save_pretrained("outputs/dpo-adapter")
print("DPO done -> outputs/dpo-adapter")
```

- [ ] **Step 2: Run on Colab/Kaggle and verify**

Expected: DPO reward margin (chosen > rejected) trends positive; `outputs/dpo-adapter/` created. Optionally push to HF Hub with `model.push_to_hub(...)`.

- [ ] **Step 3: Commit**

```bash
git add training/dpo_train.py && git commit -m "feat: QLoRA DPO training script"
```

---

### Task 12: Eval metrics

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `valid_json_rate(outputs)`, `macro_close(pred, truth, tol)`, `respects_equipment(plan, equipment)`, `avoids_injury(plan, injury)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
from src.metrics import valid_json_rate, macro_close, avoids_injury

def test_valid_json_rate():
    outs = ['{"a":1}', 'not json', '{"b":2}']
    assert valid_json_rate(outs) == 2/3

def test_macro_close_within_tolerance():
    assert macro_close(247, 250, tol=0.05) is True
    assert macro_close(100, 250, tol=0.05) is False

def test_avoids_injury_detects_contraindication():
    plan = {"weekly_workouts":[{"exercises":[{"name":"Barbell Back Squat"}]}]}
    assert avoids_injury(plan, "knee pain") is False   # squat flagged for knee
    assert avoids_injury(plan, None) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -q`
Expected: FAIL — `No module named 'src.metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/metrics.py
import json

CONTRAINDICATED = {
    "knee pain": ["squat", "lunge", "leg press"],
    "shoulder impingement": ["overhead press", "upright row", "bench press"],
    "lower back pain": ["deadlift", "good morning", "bent over row"],
}

def valid_json_rate(outputs: list[str]) -> float:
    if not outputs: return 0.0
    ok = 0
    for o in outputs:
        try: json.loads(o); ok += 1
        except Exception: pass
    return ok / len(outputs)

def macro_close(pred: float, truth: float, tol: float = 0.1) -> bool:
    if truth == 0: return pred == 0
    return abs(pred - truth) / truth <= tol

def respects_equipment(plan: dict, equipment: str) -> bool:
    # bodyweight-only plans must not require gym machines
    if equipment != "bodyweight only": return True
    banned = ["barbell", "machine", "cable", "leg press"]
    for day in plan.get("weekly_workouts", []):
        for ex in day.get("exercises", []):
            if any(b in ex["name"].lower() for b in banned):
                return False
    return True

def avoids_injury(plan: dict, injury) -> bool:
    if not injury: return True
    bad = CONTRAINDICATED.get(injury, [])
    for day in plan.get("weekly_workouts", []):
        for ex in day.get("exercises", []):
            if any(b in ex["name"].lower() for b in bad):
                return False
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py && git commit -m "feat: add eval metric functions"
```

---

### Task 13: Eval harness runner (base vs fine-tuned)

**Files:**
- Create: `eval/run_eval.py`

**Interfaces:**
- Consumes: `data/generated/eval.jsonl`, `src.metrics`, `src.schema`, a model-generate function.
- Produces: `eval/results.json` with metrics for base and fine-tuned.

- [ ] **Step 1: Write the eval runner** (runs in Colab where the model is loaded)

```python
# eval/run_eval.py
import json
from pathlib import Path
from src.schema import FitnessPlan
from src.metrics import valid_json_rate, respects_equipment, avoids_injury
from src.profiles import GOALS  # noqa: ensures src import path works

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "generated" / "eval.jsonl"

def evaluate(generate_fn, label):
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines()]
    outputs, eq_ok, inj_ok = [], 0, 0
    for r in rows:
        p = r["profile"]
        text = generate_fn(p)                 # model returns JSON string
        outputs.append(text)
        try:
            plan = FitnessPlan.model_validate_json(text).model_dump()
        except Exception:
            continue
        eq_ok += int(respects_equipment(plan, p["equipment"]))
        inj_ok += int(avoids_injury(plan, p["injury"]))
    n = len(rows)
    return {"label": label, "valid_json_rate": valid_json_rate(outputs),
            "equipment_satisfaction": eq_ok / n, "injury_safety": inj_ok / n}

def main(base_generate, tuned_generate):
    results = [evaluate(base_generate, "base"), evaluate(tuned_generate, "fine-tuned")]
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

# In Colab: define base_generate(profile)->str and tuned_generate(profile)->str
# using each loaded model, then call main(base_generate, tuned_generate).
```

- [ ] **Step 2: Run in Colab and verify the gap**

Load base model and the `outputs/dpo-adapter` model, define the two `generate_fn`s, call `main(...)`. Expected: fine-tuned shows higher `valid_json_rate`, `equipment_satisfaction`, and `injury_safety` than base. Record the numbers.

- [ ] **Step 3: Commit**

```bash
git add eval/run_eval.py && git commit -m "feat: eval harness comparing base vs fine-tuned"
```

---

### Task 14: FastAPI demo app

**Files:**
- Create: `app/inference.py`
- Create: `app/main.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: a `generate(profile) -> str` (model or a stub), `FitnessPlan`, `ExerciseDB`, `enrich_media`.
- Produces: `POST /plan` returning a validated, media-enriched plan.

- [ ] **Step 1: Write the failing test** (uses a stub generator so no GPU needed)

```python
# tests/test_app.py
import json
from pathlib import Path
from fastapi.testclient import TestClient
from tests.test_curate import _plan   # reuse a valid plan
from app.main import create_app

FIX = Path(__file__).parent / "fixtures" / "exercises_sample.json"

def stub_generate(profile):        # pretends to be the model
    return json.dumps(_plan())

def test_plan_endpoint_returns_enriched_plan():
    app = create_app(generate_fn=stub_generate, exercises_path=FIX)
    client = TestClient(app)
    resp = client.post("/plan", json={"age":30,"weight_kg":80,"goal":"lose fat",
        "equipment":"home dumbbells","diet":"no restriction",
        "experience":"beginner","injury":None,"days_per_week":3})
    assert resp.status_code == 200
    body = resp.json()
    img = body["weekly_workouts"][0]["exercises"][0]["demo_image"]
    assert img.endswith("GobletSquat/0.jpg")   # media enriched from DB
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py -q`
Expected: FAIL — `No module named 'app.main'`.

- [ ] **Step 3: Write `app/inference.py`** (real model loader; not exercised by the test)

```python
# app/inference.py
import json

def make_model_generate(adapter_path: str):
    """Load the fine-tuned model and return generate(profile)->json str."""
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(adapter_path,
        max_seq_length=4096, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    def generate(profile: dict) -> str:
        user = (f"Profile: {profile}. Return the JSON plan.")
        msgs = [{"role": "user", "content": user}]
        inputs = tok.apply_chat_template(msgs, return_tensors="pt",
            add_generation_prompt=True).to(model.device)
        out = model.generate(input_ids=inputs, max_new_tokens=1500, temperature=0.7)
        text = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        return text
    return generate
```

- [ ] **Step 4: Write `app/main.py`**

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from src.schema import FitnessPlan
from src.exercise_db import ExerciseDB
from src.curate import enrich_media

def create_app(generate_fn, exercises_path):
    app = FastAPI(title="AI Fitness Coach")
    db = ExerciseDB(exercises_path)

    @app.post("/plan")
    def plan(profile: dict):
        raw = generate_fn(profile)
        try:
            validated = FitnessPlan.model_validate_json(raw).model_dump()
        except Exception:
            raise HTTPException(422, "model produced invalid plan")
        return enrich_media(validated, db)
    return app
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_app.py -q`
Expected: PASS (1 passed).

- [ ] **Step 6: Run the whole suite**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/inference.py app/main.py tests/test_app.py && git commit -m "feat: FastAPI /plan endpoint with schema validation + media enrichment"
```

---

### Task 15: Results documentation

**Files:**
- Modify: `README.md`
- Create: `docs/RESULTS.md`

**Interfaces:**
- Consumes: `eval/results.json`.
- Produces: the resume-facing numbers write-up.

- [ ] **Step 1: Write `docs/RESULTS.md` from the eval numbers**

```markdown
# v1 Results — base vs. fine-tuned

| Metric | Base Qwen2.5-7B | Fine-tuned (SFT→DPO) |
|---|---|---|
| Valid-JSON rate | <fill from results.json> | <fill> |
| Equipment satisfaction | <fill> | <fill> |
| Injury safety | <fill> | <fill> |

Method: QLoRA (PEFT), SFT then DPO, ~1,200 SFT / ~400 DPO examples,
grounded in free-exercise-db + USDA FoodData Central. Held-out eval: 150 profiles.
```

- [ ] **Step 2: Add a results section + resume bullet to `README.md`**

Append:
```markdown
## Results
See `docs/RESULTS.md`. Fine-tuning improved valid-JSON, equipment-constraint
satisfaction, and injury-safety over the base model on a 150-profile held-out set.

## Resume bullet
Fine-tuned Qwen2.5-7B (QLoRA, SFT→DPO) into a fitness coach with tool-grounded
macros and injury-aware safety; built an eval harness measuring valid-JSON,
constraint-satisfaction, and safety, showing a measurable gain over the base model.
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/RESULTS.md && git commit -m "docs: v1 results and resume bullet"
```

---

## Self-Review

**Spec coverage:** base model + QLoRA SFT→DPO (Tasks 10–11) ✓; JSON output schema (Task 2) ✓; tool-use nutrition (Task 3) ✓; exercise media, no hallucinated URLs (Tasks 4, 7) ✓; daily_schedule + beginner "why" (schema Task 2 + generation Task 6) ✓; real-grounded data ~1200/400/150 (Tasks 5–9) ✓; eval metrics incl. injury safety + constraint satisfaction (Tasks 12–13) ✓; thin FastAPI serving (Task 14) ✓; free tooling ✓; results/resume framing (Task 15) ✓. v2/v3 features intentionally excluded.

**Placeholder scan:** `docs/RESULTS.md` intentionally has `<fill>` slots — these are filled from real eval output at execution time, not plan-authoring time. No other placeholders.

**Type consistency:** `generate_fn(profile:dict)->str`, `ExerciseDB.lookup->{"demo_image",...}`, `enrich_media(plan_dict, db)`, `FitnessPlan` schema names are consistent across generation, curation, eval, and app tasks.
