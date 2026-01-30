# Tau Retail Smoke Adapter

This adapter exercises the tau-retail evaluation shape:

- customer request
- retail policy
- mutable order database
- agent actions over the workspace
- reward by final database-state comparison and policy compliance

Run the bundled smoke set:

```bash
python3 -m agenticevals evaluate examples.tau_retail_env:TauRetailEnv --agent scripted --backend local
```

Use an external adapter-compatible task export:

```bash
AGENTICEVALS_TAU_RETAIL_TASKS=/path/to/tau-retail/tasks.json \
python3 -m agenticevals evaluate examples.tau_retail_env:TauRetailEnv --agent openai --backend docker --image auto
```

The external JSON must contain items with `id`, `request` or `instruction`, `db` or `initial_state`, and `expected_db` or `goal_state` fields. The bundled items validate the harness. This is not the canonical upstream tau2 simulator adapter.
