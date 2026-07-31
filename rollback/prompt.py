def encode_prompt(prompt, tokenizer):
    """Chat-template the prompt (DSP-V2 is a chat model); fall back to raw."""
    # wrap in the model’s chat format and then converts that entire formatted text into numerical token IDs
    try:
        out = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                            add_generation_prompt=True,
                                            return_tensors="pt")
        return out["input_ids"] if isinstance(out, dict) else \
               (out.input_ids if hasattr(out, "input_ids") else out)
    # converts only the exact raw text
    except Exception:
        return tokenizer(prompt, return_tensors="pt").input_ids
