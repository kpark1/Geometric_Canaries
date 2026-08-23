from typing import Any


def encode_prompt(prompt: str, tokenizer: Any) -> Any:
    """Chat-template the prompt (DSP-V2 is a chat model); fall back to raw."""
    # wrap in the model’s chat format and then converts that entire formatted text into numerical token IDs
    try:
        out = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return (
            out["input_ids"]
            if isinstance(out, dict)
            else (out.input_ids if hasattr(out, "input_ids") else out)
        )
    # converts only the exact raw text
    except Exception:
        return tokenizer(prompt, return_tensors="pt").input_ids


def encode_messages(messages: list[dict[str, str]], tokenizer: Any) -> Any:
    """Chat-template a full conversation (multi-turn agent retries); fall back
    to a plain role-prefixed concatenation for tokenizers with no chat template."""
    try:
        out = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return (
            out["input_ids"]
            if isinstance(out, dict)
            else (out.input_ids if hasattr(out, "input_ids") else out)
        )
    except Exception:
        joined = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return tokenizer(joined, return_tensors="pt").input_ids
