# v0.1 Smoke Result Artifact

This directory contains a checked-in real-account smoke result generated on 2026-05-12. It is useful for validating artifact shape; it is not a public benchmark result.

Included:

- `baselines.json`: machine-readable baseline rows with bootstrap CIs, pass@1, pass^k, cost-per-success fields, calibration metadata, and limitations.
- `RESULTS.md`: human-readable smoke-result table.
- `gate.json`: release-gate output.

Generated `*.calibration.json` files are intentionally not checked in; regenerate them from the labeled JSONL corpus when running a release gate.

Reproduce the same smoke shape with local CLI accounts:

```bash
AGENTICEVALS_CODEX_COMMAND='codex exec --dangerously-bypass-approvals-and-sandbox {prompt}' \
AGENTICEVALS_CLAUDE_COMMAND='claude -p --permission-mode acceptEdits --add-dir={workspace} {prompt}' \
python3 -m agenticevals env-baselines examples.tau_retail_env:TauRetailEnv \
  --agents codex,claude-code \
  --max-items 3 \
  --trials 1 \
  --backend local \
  --output baselines/v0.1
```

For provider-native cost accounting, run `openai`, `claude`, and `gemini` agents with API keys instead of CLI-account adapters.
