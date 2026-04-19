from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick Gemma generation test.")
    parser.add_argument("--model-id", default="google/gemma-2-2b-it")
    parser.add_argument("--prompt", default="Write a short poem about a cat in space.")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    local_only = args.local_files_only or os.getenv("HF_HUB_OFFLINE") == "1"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=local_only)

    dtype = torch.bfloat16
    if not _supports_bfloat16():
        dtype = "auto"

    model_kwargs = {
        "dtype": dtype,
        "local_files_only": local_only,
        "low_cpu_mem_usage": True,
    }

    if _supports_bfloat16():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    if not _supports_bfloat16():
        model.to("cpu")

    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def _supports_bfloat16() -> bool:
    if torch.cuda.is_available():
        return True
    mps = getattr(torch.backends, "mps", None)
    return bool(mps and mps.is_available())


if __name__ == "__main__":
    main()
