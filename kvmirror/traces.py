from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenTrace:
    index: int
    token_text: str
    attention_received: float
    generated_attention_received: float
    prompt_role: str


@dataclass
class TraceSummary:
    model_name: str
    prompt_token_count: int
    generated_token_count: int
    bytes_per_token: int
    token_traces: list[TokenTrace]

