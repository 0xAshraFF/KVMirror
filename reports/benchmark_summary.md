# Benchmark Summary

This file tracks the benchmark outputs that are worth keeping in Git, even when
raw JSON artifacts remain ignored.

## Recommended local benchmark model

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Why: it runs locally on this machine, produces usable prompt-attention traces,
  and is practical for repeated policy experiments.

## Low-setting simulator comparison

Command family:

```bash
python3 -m kvmirror.runner --policy <policy> --sequence-length 1024
```

Results:

| Policy | Kept tokens | Kept ratio | Saved bytes | Heavy-hitter recall | Sink recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `keep_all` | 1024 / 1024 | 1.0000 | 0 | 1.0000 | 1.0000 |
| `recent_window` | 272 / 1024 | 0.2656 | 73,924,608 | 0.9479 | 1.0000 |
| `heavy_hitter` | 96 / 1024 | 0.0938 | 91,226,112 | 1.0000 | 1.0000 |
| `hybrid` | 228 / 1024 | 0.2227 | 78,249,984 | 0.9896 | 1.0000 |

Takeaway:

- `heavy_hitter` gives the strongest raw memory reduction in the simulator.
- `hybrid` looks like the safer product default because it preserves nearly all
  heavy hitters while keeping only about `22%` of tokens.

## Qwen trace smoke test

Command:

```bash
HF_HUB_OFFLINE=1 python3 scripts/run_trace.py --model-name Qwen/Qwen2.5-0.5B-Instruct --output reports/qwen_0_5b_trace.json
```

Observed output:

- prompt tokens: `31`
- generated tokens: `24`
- estimated bytes per token: `12,288`
- replay result on this short prompt: all policies kept `100%` of prompt tokens

Interpretation:

- The real-model trace path works on Qwen 0.5B.
- This specific prompt was too short to force meaningful eviction, so it should
  be treated as a smoke test, not a full retention benchmark.

## Gemma status

- `google/gemma-2-2b-it` downloads and loads successfully.
- On this CPU-only machine it remains too slow for practical repeated
  benchmarks.
- Gemma is useful here as a compatibility check, but Qwen is the better default
  benchmark target for roadmap work on this hardware.
