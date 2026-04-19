from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RunConfig
from .policies import (
    HeavyHitterPolicy,
    HybridPolicy,
    KeepAllPolicy,
    RecentWindowPolicy,
)
from .simulator import run_simulation


def main() -> None:
    args = _parse_args()
    config = RunConfig(
        sequence_length=args.sequence_length,
        head_count=args.head_count,
        head_dim=args.head_dim,
        layer_count=args.layer_count,
        bytes_per_value=args.bytes_per_value,
        sink_tokens=args.sink_tokens,
        heavy_hitter_stride=args.heavy_hitter_stride,
        recent_window=args.window_size,
        heavy_hitter_topk=args.topk,
        seed=args.seed,
    )
    policy = _build_policy(args)
    result = run_simulation(config, policy)

    payload = {
        "policy": result.policy_name,
        "sequence_length": result.config.sequence_length,
        "bytes_per_token": result.config.bytes_per_token,
        "stats": result.stats.__dict__,
    }
    print(json.dumps(payload, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a KVMirror KV-cache retention simulation.")
    parser.add_argument("--policy", choices=["keep_all", "recent_window", "heavy_hitter", "hybrid"], default="hybrid")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--head-count", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--layer-count", type=int, default=24)
    parser.add_argument("--bytes-per-value", type=int, default=2)
    parser.add_argument("--sink-tokens", type=int, default=16)
    parser.add_argument("--heavy-hitter-stride", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _build_policy(args: argparse.Namespace):
    if args.policy == "keep_all":
        return KeepAllPolicy()
    if args.policy == "recent_window":
        return RecentWindowPolicy(window_size=args.window_size)
    if args.policy == "heavy_hitter":
        return HeavyHitterPolicy(topk=args.topk)
    return HybridPolicy(window_size=args.window_size, topk=args.topk)


if __name__ == "__main__":
    main()

