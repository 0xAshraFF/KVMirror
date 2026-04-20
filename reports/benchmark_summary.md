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

## Qwen practical-context benchmark

Command:

```bash
HF_HUB_OFFLINE=1 python3 - <<'PY'
# generates a 516-token prompt with repeated boilerplate plus important facts
# and writes reports/qwen_practical_context_benchmark.json
PY
```

Observed output:

- prompt tokens: `516`
- generated tokens: `1`
- estimated bytes per token: `12,288`

Replay results:

| Policy | Kept ratio | Saved bytes | Heavy-hitter recall | Sink recall |
| --- | ---: | ---: | ---: | ---: |
| `keep_all` | 1.0000 | 0 | 1.0000 | 1.0000 |
| `recent_window` | 0.5271 | 2,998,272 | 0.6354 | 1.0000 |
| `hybrid` | 0.5058 | 3,133,440 | 1.0000 | 1.0000 |

Interpretation:

- Yes, the product direction looks real in this benchmark.
- On a real Qwen attention trace, `hybrid` cut retained prompt state by about
  half while preserving all measured heavy hitters and sink tokens.
- `recent_window` also reduced memory, but it lost too much important-token
  signal to be the safer default.
- The current evidence supports `hybrid` as the better product candidate for
  memory reduction without obvious quality-risk regression.

## Gemma status

- `google/gemma-2-2b-it` downloads and loads successfully.
- On this CPU-only machine it remains too slow for practical repeated
  benchmarks.
- Gemma is useful here as a compatibility check, but Qwen is the better default
  benchmark target for roadmap work on this hardware.
