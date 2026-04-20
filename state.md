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
- Mode: Qwen practical-context benchmark
- Command:
  `HF_HUB_OFFLINE=1 python3 <inline benchmark script writing reports/qwen_practical_context_benchmark.json>`
- Result:
  - prompt tokens: `516`
  - generated tokens: `1`
  - `keep_all`: kept ratio `1.0`, saved bytes `0`, heavy-hitter recall `1.0`, sink recall `1.0`
  - `recent_window`: kept ratio `0.5271`, saved bytes `2,998,272`, heavy-hitter recall `0.6354`, sink recall `1.0`
  - `hybrid`: kept ratio `0.5058`, saved bytes `3,133,440`, heavy-hitter recall `1.0`, sink recall `1.0`
- Conclusion:
  - Qwen produced the first practical real-model benchmark with actual retention pressure
  - `hybrid` cut retained prompt state by about half while preserving all measured heavy hitters and sink tokens
  - current evidence supports `hybrid` as the safer product-default direction over `recent_window`

- Date: 2026-04-20
- Mode: roadmap benchmark target decision
- Commands:
  - `HF_HUB_OFFLINE=1 python3 scripts/run_trace.py --model-name Qwen/Qwen2.5-0.5B-Instruct --output reports/qwen_0_5b_trace.json`
  - `HF_HUB_OFFLINE=1 python3 run_gemma_persistent.py --local-files-only --max-input-tokens 8 --max-new-tokens 1`
- Result:
  - Qwen `0.5B` trace path completed successfully on local hardware
  - prompt tokens: `31`
  - generated tokens: `24`
  - replay policies all kept `100%` on the short smoke prompt
  - Gemma `2B` loads and generates in a persistent session, but is too slow for practical repeated benchmark runs on this CPU-only machine
- Conclusion:
  - `Qwen/Qwen2.5-0.5B-Instruct` is the default benchmark model for current roadmap work
  - Gemma remains a compatibility check, not the primary local benchmark target

- Date: 2026-04-20
- Mode: low-setting simulator benchmark sweep
- Commands:
  - `python3 -m kvmirror.runner --policy keep_all --sequence-length 1024 --output reports/keep_all_low.json`
  - `python3 -m kvmirror.runner --policy recent_window --sequence-length 1024 --window-size 256 --output reports/recent_window_low.json`
  - `python3 -m kvmirror.runner --policy heavy_hitter --sequence-length 1024 --topk 96 --output reports/heavy_hitter_low.json`
  - `python3 -m kvmirror.runner --policy hybrid --sequence-length 1024 --window-size 192 --topk 96 --output reports/hybrid_low_compact.json`
- Result:
  - `keep_all`: kept `1024 / 1024`, saved bytes `0`, heavy-hitter recall `1.0`, sink recall `1.0`
  - `recent_window`: kept `272 / 1024`, saved bytes `73,924,608`, heavy-hitter recall `0.9479`, sink recall `1.0`
  - `heavy_hitter`: kept `96 / 1024`, saved bytes `91,226,112`, heavy-hitter recall `1.0`, sink recall `1.0`
  - `hybrid`: kept `228 / 1024`, saved bytes `78,249,984`, heavy-hitter recall `0.9896`, sink recall `1.0`
- Conclusion:
  - low-setting runs completed successfully
  - `heavy_hitter` gives the strongest memory reduction in this simulator
  - `hybrid` preserves almost all heavy hitters while keeping only `22.3%` of tokens, which looks like the safer product candidate

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

## Previous run

## Update rule

After every run, append:

1. date
2. command
3. model or simulation mode
4. key memory metrics
5. quality metrics or proxy metrics
6. short conclusion
