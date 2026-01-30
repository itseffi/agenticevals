# AgentSmoke

`AgentSmoke` is a deterministic 100-item environment for smoke-testing agent adapters and computer interfaces across common agent work:

- file transformation
- shell reasoning
- Python debugging
- data cleaning
- browser-visible app validation

Run the first ten items:

```bash
python3 -m agenticevals evaluate examples.agent_smoke_env:AgentSmokeEnv --agent scripted --max-items 10 --backend local
```

Compare against a no-op agent:

```bash
python3 -m agenticevals evaluate examples.agent_smoke_env:AgentSmokeEnv --agent noop --max-items 10 --backend local
```

Fixtures are generated deterministically under `workspace/agent-smoke-fixtures`.
