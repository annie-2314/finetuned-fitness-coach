def make_model_generate(adapter_path: str):
    """Load the fine-tuned model and return generate(profile)->json str.

    Used on a GPU (Colab/local Ollama export). Not exercised by the unit tests,
    which pass a stub generate function instead.
    """
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        adapter_path, max_seq_length=4096, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    def generate(profile: dict) -> str:
        user = f"Profile: {profile}. Return the JSON plan."
        msgs = [{"role": "user", "content": user}]
        inputs = tok.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
        out = model.generate(input_ids=inputs, max_new_tokens=1500, temperature=0.7)
        return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    return generate
