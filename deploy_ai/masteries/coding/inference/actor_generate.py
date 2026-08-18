"""
PACE Actor Inference Node
Task LE-1: Loads the trained Actor Model, generates multiple code fixes based on a prompt.
"""

import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_fixes(
    prompt: str,
    model_dir: str = "masteries/coding/models/actor_v1",
    num_return_sequences: int = 3,
    fallback_model: str = "bigcode/tiny_starcoder_py",
) -> list[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hf_repo = "kiritox07/pace-models"

    if os.path.exists(model_dir):
        # Load local
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
        except Exception:
            tokenizer = None
            model = None
    else:
        # Load from Hugging Face Hub subfolder
        print(f"[SYSTEM] Local model not found. Downloading {model_dir} from HF Hub ({hf_repo})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_repo, subfolder=model_dir)
            model = AutoModelForCausalLM.from_pretrained(hf_repo, subfolder=model_dir).to(device)
        except Exception:
            tokenizer = None
            model = None

    if tokenizer is None or model is None:
        print(f"[SYSTEM] Failed to load from {hf_repo}. Falling back to {fallback_model}...")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model = AutoModelForCausalLM.from_pretrained(fallback_model).to(device)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=num_return_sequences,
        )

    decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # VRAM PURGE: Destroy tensors and clear cache
    del model, tokenizer, inputs, generated_ids
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(
            f"[VRAM] Actor Purged. Idling Footprint: {torch.cuda.memory_allocated() / (1024**2):.2f} MB"
        )

    return decoded_texts


if __name__ == "__main__":
    print("[SYSTEM] Testing Actor Inference Engine")
    test_prompt = "def linear_search(arr):"
    print(generate_fixes(test_prompt, num_return_sequences=1))