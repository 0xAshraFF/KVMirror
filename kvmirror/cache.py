from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVToken:
    index: int
    attention_mass: float
    novelty: float
    recency: float
    span_role: str


@dataclass
class CacheStats:
    kept_tokens: int
    dropped_tokens: int
    kept_ratio: float
    estimated_bytes: int
    estimated_saved_bytes: int
    heavy_hitter_recall: float
    sink_recall: float

