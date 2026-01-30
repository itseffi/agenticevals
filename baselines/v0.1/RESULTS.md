# v0.1 Real Account Smoke Results

Dataset: `tau-retail-smoke`  
Environment: `examples.tau_retail_env:TauRetailEnv`  
Generated: `2026-05-12T14:14:42Z`

| Agent | Account mode | Items | Trials | Pass@1 | Pass^k | 95% CI | Cost/success | Mean seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| claude-code | local CLI account | 3 | 1 | 1.000 | 1.000 | [1.000, 1.000] | n/a | 18.9 |
| codex | local CLI account | 3 | 1 | 1.000 | 1.000 | [1.000, 1.000] | n/a | 56.1 |

Scope: smoke result only. Do not treat this as a public benchmark result.

This file is intentionally a smoke baseline, not an upstream τ²/τ³ claim. The official upstream retail data lives in `sierra-research/tau2-bench` under `data/tau2/domains/retail`.
