"""
PACE Critic Inference Node
Task LE-2: Loads the trained 125M CodeBERT Critic, evaluates a batch of code strings.
"""

import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def evaluate_syntax_batch(
    code_snippets: list[str],
    model_dir: str = "masteries/coding/models/critic_v3",
    fallback_model: str = "microsoft/codebert-base",
) -> list[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hf_repo = "kiritox07/pace-models"

    if os.path.exists(model_dir):
        # Load local
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2).to(device)
        except Exception:
            tokenizer = None
            model = None
    else:
        # Load from Hugging Face Hub subfolder
        print(f"[SYSTEM] Local model not found. Downloading {model_dir} from HF Hub ({hf_repo})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_repo, subfolder=model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(hf_repo, subfolder=model_dir, num_labels=2).to(device)
        except Exception:
            tokenizer = None
            model = None

    if tokenizer is None or model is None:
        print(f"[SYSTEM] Failed to load from {hf_repo}. Falling back to {fallback_model}...")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model = AutoModelForSequenceClassification.from_pretrained(fallback_model, num_labels=2).to(device)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        code_snippets,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert raw logits to percentages (0.0 to 1.0)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Extract the probability of Class 1 (Bug). Lower is better (closer to 0 / Clean).
    bug_probs = probabilities[:, 1].tolist()

    # VRAM PURGE: Destroy tensors and clear cache
    del model, tokenizer, inputs, outputs, probabilities
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(
            f"[VRAM] Critic Purged. Idling Footprint: {torch.cuda.memory_allocated() / (1024**2):.2f} MB"
        )

    return bug_probs


if __name__ == "__main__":
    print("[SYSTEM] Testing Critic Inference Engine...")
    bad_code = ["def calculate_sum(a, b) return a + b"]
    print("Testing Critic Inference Engine...")
    print(evaluate_syntax_batch(bad_code))
