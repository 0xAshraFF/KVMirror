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
scripts/
```

## Quick start

```bash
python3 -m kvmirror.runner --policy keep_all
python3 -m kvmirror.runner --policy recent_window --window-size 256
python3 -m kvmirror.runner --policy hybrid --window-size 192 --topk 96
HF_HUB_OFFLINE=1 python3 scripts/run_trace.py --model-name Qwen/Qwen2.5-0.5B-Instruct
```

See `reports/benchmark_summary.md` for the latest tracked simulator and
real-model benchmark notes.

## What comes next

1. Expand transformer hooks from prompt-token tracing to full per-layer KV accounting.
2. Replay real prompts through a local model and compare policies on memory and answer quality.
3. Add quantized-value support so `KVMirror` can mix retention and compression.
