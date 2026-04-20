# KVMirror

`KVMirror` is a research repo for LLM KV-cache reduction.

It is a starting point for this product problem:

- keep fewer KV entries during long-context inference,
- preserve important context,
- measure memory saved versus quality risk,
- grow toward real transformer-layer integration.

## Current state

The repo currently contains:

- a KV-cache simulator for long sequences,
- policy interfaces for retention / eviction strategies,
- baseline policies (`keep_all`, `recent_window`, `heavy_hitter`, `hybrid`),
- a CLI runner that reports memory retention and proxy recall metrics,
- a first transformer attention-trace capture path and replay evaluator.

It does **not yet** hook into a live transformer runtime. The simulator is a
foundation for policy design and evaluation before wiring into actual model
internals.

It now also includes a first real-model trace path that captures prompt-token
attention summaries during Hugging Face generation. This is still an early
instrumentation layer, not full live KV eviction.

The current recommended benchmark model on local hardware is
`Qwen/Qwen2.5-0.5B-Instruct`. It is light enough to run practical experiments on
CPU-only machines while still giving a more realistic trace signal than tiny
toy checkpoints.

## Benchmark results

### Simulator sweep (sequence length 1024)

| Policy | Kept tokens | Kept ratio | Saved bytes | Heavy-hitter recall | Sink recall |
|---|---:|---:|---:|---:|---:|
| `keep_all` | 1024 / 1024 | 1.0000 | 0 | 1.0000 | 1.0000 |
| `recent_window` | 272 / 1024 | 0.2656 | 73,924,608 | 0.9479 | 1.0000 |
| `heavy_hitter` | 96 / 1024 | 0.0938 | 91,226,112 | 1.0000 | 1.0000 |
| `hybrid` | 228 / 1024 | 0.2227 | 78,249,984 | 0.9896 | 1.0000 |

### Qwen practical-context benchmark (516 prompt tokens, real attention trace)

Scenario: a 516-token prompt with repeated boilerplate and embedded important facts,
run on `Qwen/Qwen2.5-0.5B-Instruct` locally.

| Policy | Kept ratio | Saved bytes | Heavy-hitter recall | Sink recall |
|---|---:|---:|---:|---:|
| `keep_all` | 1.0000 | 0 | 1.0000 | 1.0000 |
| `recent_window` | 0.5271 | 2,998,272 | 0.6354 | 1.0000 |
| `hybrid` | 0.5058 | 3,133,440 | 1.0000 | 1.0000 |

**Key finding:** on a real Qwen attention trace, `hybrid` cut retained prompt state by
~50% while preserving **all** measured heavy hitters and sink tokens.
`recent_window` achieved similar memory savings but lost ~36% of important-token
signal — a meaningful quality risk.

**Current recommendation:** `hybrid` is the safer product-default policy. It matches
or beats `recent_window` on memory while maintaining perfect heavy-hitter recall.

## Repo layout

```text
kvmirror/
  __init__.py
  cache.py
  config.py
  policies.py
  simulator.py
  runner.py
  hooks.py
  traces.py
  replay.py
reports/
  benchmark_summary.md        # tracked benchmark notes
  qwen_practical_context_benchmark.json  # latest real-model trace artifact
scripts/
```

## Quick start

```bash
python3 -m kvmirror.runner --policy keep_all
python3 -m kvmirror.runner --policy recent_window --window-size 256
python3 -m kvmirror.runner --policy hybrid --window-size 192 --topk 96
HF_HUB_OFFLINE=1 python3 scripts/run_trace.py --model-name Qwen/Qwen2.5-0.5B-Instruct
```

See `reports/benchmark_summary.md` for the full benchmark history.

## What comes next

1. Expand transformer hooks from prompt-token tracing to full per-layer KV accounting.
2. Replay real prompts through a local model and compare policies on memory and answer quality.
3. Add quantized-value support so `KVMirror` can mix retention and compression.
4. Run `hybrid` against longer contexts (2K–8K tokens) to validate the recall advantage holds at scale.
