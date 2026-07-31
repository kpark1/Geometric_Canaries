import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_runtime(project_cache, dry_run=True, device_request="auto"):
    os.environ["HF_HOME"] = os.path.join(project_cache, "huggingface")
    os.environ["MPLCONFIGDIR"] = os.path.join(project_cache, "matplotlib")
    huggingface_cache = os.path.join(project_cache, "huggingface", "hub")

    device_request = device_request.lower()
    if device_request not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "GEOMETRIC_CANARIES_DEVICE must be one of: auto, cpu, cuda"
        )

    cuda_available = torch.cuda.is_available()
    if device_request == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested but is unavailable. Check nvidia-smi and the "
            "installed PyTorch CUDA build."
        )
    device_map = (
        "cuda"
        if device_request == "cuda"
        else ("auto" if device_request == "auto" and cuda_available else "cpu")
    )
    use_cuda = device_map != "cpu"
    model_dtype = (
        torch.bfloat16
        if use_cuda and torch.cuda.is_bf16_supported()
        else (torch.float16 if use_cuda else torch.float32)
    )
    print(f"CUDA: {cuda_available}; device: {device_map}; dtype: {model_dtype}")
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {properties.name}, {properties.total_memory/1e9:.1f} GB")
    elif not dry_run:
        print("!! no GPU — set DRY_RUN = True or switch runtime")

    if dry_run:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=model_dtype, device_map=device_map,
            cache_dir=huggingface_cache,
        )
    else:

        model_id = "deepseek-ai/DeepSeek-Prover-V2-7B"
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map=device_map,
            dtype=model_dtype,
            cache_dir=huggingface_cache,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=huggingface_cache
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    layer_count = model.config.num_hidden_layers
    capture_layer = int(layer_count * 0.65)
    print(f"{model_id}: {layer_count} layers, capturing layer {capture_layer}")

    return model, tokenizer, capture_layer
