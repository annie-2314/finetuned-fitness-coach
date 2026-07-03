# IMPORTANT: the model was trained WITH this system prompt. Serving must include it
# (train/serve parity) — dropping it collapses schema conformance to ~0%.
SYSTEM = (
    "You are an expert fitness coach. Given a user profile, output ONLY a JSON "
    "object matching this schema: goal, experience (beginner|intermediate|advanced), "
    "daily_schedule{wake, workout{time,type}, meals[{time,name,focus}], sleep{target,hours}}, "
    "weekly_workouts[{day, focus, exercises[{name,sets,reps,rest_seconds,demo_image,why}]}], "
    "nutrition{daily_macros{calories,protein_g,carbs_g,fat_g}, "
    "example_day[{food,grams,calories,protein_g}], grocery_list[]}, disclaimer. "
    "Use realistic exercises and macros. Set demo_image to null (filled later). "
    "For beginners, fill 'why' with a short reason. Always include a disclaimer that "
    "this is general guidance, not medical advice. If the user reports an injury, "
    "avoid contraindicated movements."
)


def build_user_prompt(p: dict) -> str:
    return (f"Profile: {p['age']}yo, {p['weight_kg']}kg, goal: {p['goal']}, "
            f"equipment: {p['equipment']}, diet: {p['diet']}, "
            f"experience: {p['experience']}, injury: {p.get('injury')}, "
            f"{p['days_per_week']} days/week. Return the JSON plan.")


def make_model_generate(adapter_path: str):
    """Load the fine-tuned adapter and return generate(profile)->json str.

    Runs on a GPU. Not exercised by the unit tests (they pass a stub).
    max_new_tokens is high (2500) because full multi-day plans are long; a lower
    cap truncated the JSON mid-plan and made outputs unparseable.
    """
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        adapter_path, max_seq_length=4096, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def generate(profile: dict) -> str:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user_prompt(profile)}]
        inputs = tok.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
        out = model.generate(input_ids=inputs, max_new_tokens=2500, temperature=0.7,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    return generate
