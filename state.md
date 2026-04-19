# KVMirror State

This file tracks the current state of the repo and should be updated every time
we run a new simulation, replay, or transformer-hook benchmark.

## Current status

- Repo created as a separate project from MemoirAI.
- KV-cache simulator is working.
- Baseline retention policies are implemented:
  - `keep_all`
  - `recent_window`
  - `heavy_hitter`
  - `hybrid`
- First smoke simulation completed successfully.
- Real transformer hook path is scaffolded but not yet benchmarked on a full
  model-quality task.

## Latest run

- Date: 2026-04-20
- Mode: transformer trace smoke test
- Command:
  `python3 scripts/run_trace.py --model-name sshleifer/tiny-gpt2 --prompt 'Repeat the key point of this short note in one sentence.' --max-new-tokens 8 --output reports/trace_run.json`
- Result:
  - prompt tokens: `12`
  - generated tokens: `8`
  - attention trace captured without crashing
  - all prompt-token attention values were `0.0` for this tiny model path
  - replay policies all kept `100%` of tokens
  - estimated saved bytes: `0`
  - conclusion: the hook path works structurally, but this model is not a good
    trace-quality target for KV policy evaluation

## Previous run

- Date: 2026-04-20
- Mode: simulator smoke test
- Command:
  `python3 -m kvmirror.runner --policy hybrid --sequence-length 1024 --output reports/hybrid_smoke.json`
- Result:
  - kept tokens: `285 / 1024`
  - kept ratio: `0.2783`
  - estimated saved bytes: `72,646,656`
  - heavy-hitter recall: `1.0`
  - sink recall: `1.0`

## Update rule

After every run, append:

1. date
2. command
3. model or simulation mode
4. key memory metrics
5. quality metrics or proxy metrics
6. short conclusion
