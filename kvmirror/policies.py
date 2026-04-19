from __future__ import annotations

from dataclasses import dataclass

from .cache import KVToken


class RetentionPolicy:
    name: str = "base"

    def select(self, tokens: list[KVToken], sink_tokens: int) -> set[int]:
        raise NotImplementedError


@dataclass
class KeepAllPolicy(RetentionPolicy):
    name: str = "keep_all"

    def select(self, tokens: list[KVToken], sink_tokens: int) -> set[int]:
        return {token.index for token in tokens}


@dataclass
class RecentWindowPolicy(RetentionPolicy):
    window_size: int = 256
    name: str = "recent_window"

    def select(self, tokens: list[KVToken], sink_tokens: int) -> set[int]:
        if not tokens:
            return set()
        max_index = tokens[-1].index
        keep = {token.index for token in tokens if token.index >= max_index - self.window_size + 1}
        keep.update(token.index for token in tokens[:sink_tokens])
        return keep


@dataclass
class HeavyHitterPolicy(RetentionPolicy):
    topk: int = 128
    name: str = "heavy_hitter"

    def select(self, tokens: list[KVToken], sink_tokens: int) -> set[int]:
        ranked = sorted(tokens, key=lambda t: (t.attention_mass, t.novelty), reverse=True)
        keep = {token.index for token in ranked[: self.topk]}
        keep.update(token.index for token in tokens[:sink_tokens])
        return keep


@dataclass
class HybridPolicy(RetentionPolicy):
    window_size: int = 256
    topk: int = 96
    name: str = "hybrid"

    def select(self, tokens: list[KVToken], sink_tokens: int) -> set[int]:
        if not tokens:
            return set()
        max_index = tokens[-1].index
        recent = {
            token.index
            for token in tokens
            if token.index >= max_index - self.window_size + 1
        }
        ranked = sorted(
            tokens,
            key=lambda t: (t.attention_mass * 0.75 + t.novelty * 0.25, t.recency),
            reverse=True,
        )
        heavy = {token.index for token in ranked[: self.topk]}
        sinks = {token.index for token in tokens[:sink_tokens]}
        return recent | heavy | sinks

