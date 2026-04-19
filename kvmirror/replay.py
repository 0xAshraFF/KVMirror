from __future__ import annotations

from dataclasses import dataclass

from .cache import CacheStats, KVToken
from .config import RunConfig
from .policies import RetentionPolicy
from .traces import TraceSummary


@dataclass
class ReplayResult:
    policy_name: str
    kept_ratio: float
    heavy_hitter_recall: float
    sink_recall: float
    estimated_saved_bytes: int


def replay_trace(summary: TraceSummary, policy: RetentionPolicy, config: RunConfig) -> ReplayResult:
    tokens = [
        KVToken(
            index=trace.index,
            attention_mass=trace.attention_received,
            novelty=1.0 if trace.prompt_role == "content" else 0.4,
            recency=trace.index / max(1, summary.prompt_token_count - 1),
            span_role=trace.prompt_role,
        )
        for trace in summary.token_traces
    ]
    keep = policy.select(tokens, sink_tokens=config.sink_tokens)
    total_tokens = len(tokens)
    kept_tokens = len(keep)

    heavy = {
        token.index
        for token in sorted(tokens, key=lambda t: t.attention_mass, reverse=True)[: config.heavy_hitter_topk]
    }
    sinks = {token.index for token in tokens[: config.sink_tokens]}

    full_bytes = total_tokens * summary.bytes_per_token
    kept_bytes = kept_tokens * summary.bytes_per_token
    return ReplayResult(
        policy_name=policy.name,
        kept_ratio=kept_tokens / total_tokens if total_tokens else 0.0,
        heavy_hitter_recall=_recall(heavy, keep),
        sink_recall=_recall(sinks, keep),
        estimated_saved_bytes=full_bytes - kept_bytes,
    )


def _recall(target_ids: set[int], kept_ids: set[int]) -> float:
    if not target_ids:
        return 1.0
    return len(target_ids & kept_ids) / len(target_ids)

