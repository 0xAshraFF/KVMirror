from __future__ import annotations

from dataclasses import asdict

from .config import RunConfig
from .traces import TokenTrace, TraceSummary


def capture_attention_trace(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    config: RunConfig,
    device: str = "cpu",
) -> TraceSummary:
    """
    Capture a lightweight prompt-token attention summary from a Hugging Face
    causal LM generation call.

    This is the first honest step toward KV-cache-aware benchmarking:
    we trace which prompt tokens receive attention during generation rather than
    pretending prompt deduplication is the same thing.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "transformers and torch are required for capture_attention_trace()"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_token_count = int(encoded["input_ids"].shape[1])

    with torch.no_grad():
        generation = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=True,
            output_scores=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    token_ids = encoded["input_ids"][0].tolist()
    token_texts = tokenizer.convert_ids_to_tokens(token_ids)
    per_token_attention = [0.0 for _ in range(prompt_token_count)]

    attentions = generation.attentions or []
    for step in attentions:
        for layer_attention in step:
            if layer_attention is None:
                continue
            # Shape: [batch, heads, q_len, kv_len]
            attn = layer_attention[0].mean(dim=0)
            last_query = attn[-1]
            limit = min(prompt_token_count, int(last_query.shape[0]))
            for idx in range(limit):
                per_token_attention[idx] += float(last_query[idx].item())

    token_traces = [
        TokenTrace(
            index=idx,
            token_text=token_texts[idx],
            attention_received=per_token_attention[idx],
            generated_attention_received=per_token_attention[idx],
            prompt_role=_infer_prompt_role(idx, prompt_token_count, config.sink_tokens),
        )
        for idx in range(prompt_token_count)
    ]

    generated_token_count = int(generation.sequences.shape[1] - prompt_token_count)
    return TraceSummary(
        model_name=model_name,
        prompt_token_count=prompt_token_count,
        generated_token_count=generated_token_count,
        bytes_per_token=config.bytes_per_token,
        token_traces=token_traces,
    )


def trace_to_dict(summary: TraceSummary) -> dict:
    return {
        "model_name": summary.model_name,
        "prompt_token_count": summary.prompt_token_count,
        "generated_token_count": summary.generated_token_count,
        "bytes_per_token": summary.bytes_per_token,
        "token_traces": [asdict(token) for token in summary.token_traces],
    }


def _infer_prompt_role(index: int, prompt_token_count: int, sink_tokens: int) -> str:
    if index < sink_tokens:
        return "sink"
    if index >= prompt_token_count - sink_tokens:
        return "recent_context"
    return "content"
