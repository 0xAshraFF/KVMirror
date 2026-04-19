from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    sequence_length: int = 4096
    head_count: int = 8
    head_dim: int = 128
    layer_count: int = 24
    bytes_per_value: int = 2
    sink_tokens: int = 16
    heavy_hitter_stride: int = 128
    recent_window: int = 256
    heavy_hitter_topk: int = 96
    seed: int = 42

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.layer_count * self.head_count * self.head_dim * self.bytes_per_value

