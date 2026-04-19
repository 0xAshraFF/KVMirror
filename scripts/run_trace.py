#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvmirror.config import RunConfig
from kvmirror.hooks import capture_attention_trace, trace_to_dict
from kvmirror.policies import HybridPolicy, KeepAllPolicy, RecentWindowPolicy
from kvmirror.replay import replay_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a real transformer attention trace.")
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt", default="Summarize the main idea of this paragraph in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", default="reports/trace_run.json")
    args = parser.parse_args()

    config = RunConfig()
    trace = capture_attention_trace(
        model_name=args.model_name,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        config=config,
    )
    policies = [
        KeepAllPolicy(),
        RecentWindowPolicy(window_size=256),
        HybridPolicy(window_size=192, topk=96),
    ]
    replay_results = [
        replay_trace(trace, policy, config).__dict__
        for policy in policies
    ]

    payload = {
        "trace": trace_to_dict(trace),
        "replay_results": replay_results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
