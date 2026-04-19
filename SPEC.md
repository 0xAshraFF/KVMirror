# KVMirror Spec

## Goal

Build a real KV-cache reduction system for transformer inference that can:

1. hook into a transformer's attention layers during inference
2. detect KV-cache entries that are redundant or low-activation
3. evict or quantize them without breaking output quality
4. benchmark memory savings against answer-quality degradation

## Non-goals

- Do not present prompt deduplication as KV-cache optimization.
- Do not couple this repo to MemoirAI classification logic.
- Do not claim product readiness before live transformer benchmarks exist.

## Build phases

### 1. Transformer hook path

Needed:

- capture per-layer attention summaries during generation
- capture token positions and prompt roles
- estimate live KV bytes by layer, head, and token count
- expose traces in a reusable format

Initial deliverables:

- `kvmirror/hooks.py`
- `kvmirror/replay.py`
- trace output JSON in `reports/`

### 2. Retention / compression policy engine

Needed:

- score tokens by sink importance, recency, and attention mass
- support retention policies:
  - keep-all
  - recent-window
  - heavy-hitter
  - hybrid
- prepare interface for future quantization policies

Initial deliverables:

- shared token score format
- policy interface that works for both simulation and transformer traces
- quantization placeholder interface

### 3. Benchmark harness

Needed:

- replay real prompts through a local transformer
- run the same prompt under multiple policies
- compare:
  - retained token count
  - estimated / actual memory use
  - latency
  - text quality proxies

Initial deliverables:

- deterministic benchmark runner
- report JSON
- `state.md` updated after every benchmark

## Features to build next

### Feature 1. Real attention trace capture

- Add Hugging Face generation wrapper with `output_attentions=True`
- Summarize attention received by each input token across generation steps
- Record sink-token and heavy-hitter candidates from real traces

### Feature 2. Trace replay evaluation

- Replay captured traces through retention policies
- Report token retention and heavy-hitter recall
- Add a per-policy comparison table

### Feature 3. Quality-preserving benchmark loop

- Run baseline and policy variants on the same prompts
- Compare output text using exact match / overlap style proxies
- Report whether memory savings come with unacceptable quality drop

## Definition of done for v0

KVMirror v0 is done when:

- one real local transformer can be traced during inference
- at least three policies can be compared on the same prompt set
- the benchmark reports memory proxy and output-quality proxy together
- the project can show one policy that reduces retained KV state materially
  without obvious output collapse

