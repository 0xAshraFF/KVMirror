from __future__ import annotations

from dataclasses import dataclass
import random

from .cache import CacheStats, KVToken
from .config import RunConfig
from .policies import RetentionPolicy


@dataclass
class SimulationResult:
    policy_name: str
    config: RunConfig
    stats: CacheStats


def run_simulation(config: RunConfig, policy: RetentionPolicy) -> SimulationResult:
    tokens = _build_sequence(config)
    keep = policy.select(tokens, sink_tokens=config.sink_tokens)
    kept_tokens = len(keep)
    total_tokens = len(tokens)
    dropped_tokens = total_tokens - kept_tokens
    estimated_bytes = kept_tokens * config.bytes_per_token
    full_bytes = total_tokens * config.bytes_per_token

    heavy_hitter_ids = {
        token.index
        for token in sorted(tokens, key=lambda t: t.attention_mass, reverse=True)[: config.heavy_hitter_topk]
    }
    sink_ids = {token.index for token in tokens[: config.sink_tokens]}

    stats = CacheStats(
        kept_tokens=kept_tokens,
        dropped_tokens=dropped_tokens,
        kept_ratio=kept_tokens / total_tokens if total_tokens else 0.0,
        estimated_bytes=estimated_bytes,
        estimated_saved_bytes=full_bytes - estimated_bytes,
        heavy_hitter_recall=_recall(heavy_hitter_ids, keep),
        sink_recall=_recall(sink_ids, keep),
    )
    return SimulationResult(policy_name=policy.name, config=config, stats=stats)


def _build_sequence(config: RunConfig) -> list[KVToken]:
    rng = random.Random(config.seed)
    tokens: list[KVToken] = []
    for idx in range(config.sequence_length):
        span_role = _span_role(idx, config)
        recency = idx / max(1, config.sequence_length - 1)

        base_attention = 0.1 + 0.4 * recency
        if idx < config.sink_tokens:
            base_attention += 0.7
        if idx % config.heavy_hitter_stride == 0:
            base_attention += 0.9
        if span_role == "anchor":
            base_attention += 0.35
        if span_role == "boilerplate":
            base_attention -= 0.18

        novelty = 0.2 + rng.random() * 0.6
        if span_role == "boilerplate":
            novelty *= 0.35
        elif span_role == "anchor":
            novelty = min(1.0, novelty + 0.3)

        tokens.append(
            KVToken(
                index=idx,
                attention_mass=max(0.0, base_attention + rng.random() * 0.1),
                novelty=novelty,
                recency=recency,
                span_role=span_role,
            )
        )
    return tokens


def _span_role(index: int, config: RunConfig) -> str:
    if index < config.sink_tokens:
        return "sink"
    if index % config.heavy_hitter_stride == 0:
        return "anchor"
    if index % 11 in {0, 1, 2, 3}:
        return "boilerplate"
    return "content"


def _recall(target_ids: set[int], kept_ids: set[int]) -> float:
    if not target_ids:
        return 1.0
    return len(target_ids & kept_ids) / len(target_ids)

