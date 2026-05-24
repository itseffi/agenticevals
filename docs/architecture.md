# Architecture

`agenticevals` evaluates AI agents, not isolated model responses.

The core unit is a trajectory:

```text
task -> agent action -> computer observation -> agent action -> ... -> final state
```

## Components

### Environment

The primary API is a Python environment class:

```python
setup()
get_next_item()
format_prompt(item)
fixture_path(item)
compute_reward(item, result, ctx)
evaluate(options)
```

An environment defines the task distribution and reward logic. JSON task files remain useful for smoke tests, but the environment API is the main path for serious evals.

### Agent

The AI agent receives a task prompt and operates in a workspace. Agents are pluggable:

- Codex
- Claude Code
- OpenAI Responses API with native function calls
- Gemini generateContent with native function calls
- Anthropic Messages API with native tool_use blocks
- custom command-line agents
- HTTP agents
- direct text-only model calls
- no-op baseline agent
- deterministic scripted agents for test fixtures

Command-line agents receive a control packet in the workspace:

- `.agenticevals/prompt.txt`
- `.agenticevals/task.json`
- `.agenticevals/result.json`
- `.agenticevals/agent-trace.jsonl`

The runner also sets `AGENTICEVALS_WORKSPACE`, `AGENTICEVALS_TASK_PATH`, `AGENTICEVALS_RESULT_PATH`, `AGENTICEVALS_TRACE_PATH`, and, when enabled, `AGENTICEVALS_SANDBOX_URL`. This lets shell-invoked agents run against the same task contract without being linked into this Python package.

HTTP agents receive the same contract as a JSON request. The adapter posts task metadata, workspace path, sandbox URL, and limits to `agent.command` or `AGENTICEVALS_HTTP_AGENT_URL`. The response can include a final message and normalized trace events.

### Controlled Computer Environment

The environment gives the agent access to computer interfaces:

- filesystem
- shell
- git
- browser sessions with navigation, DOM snapshots, form actions, and captured artifacts
- local dev servers
- test runners
- declarative HTTP tools backed by mock services

Execution goes through a backend. The current backends are:

- `local`: copied workspace on the host.
- `sandbox-http`: copied workspace controlled through the persistent sandbox HTTP interface.
- `docker`: copied workspace mounted into a container for command execution.

Backend isolation is part of the environment contract. Reward code should use `ComputerContext`, not direct host paths.

The sandbox HTTP interface exposes `/exec`, `/read`, `/write`, `/edit`, `/glob`, `/grep`, `/download`, `/browser/goto`, and `/browser/check`. For JSON task runs, `--sandbox-server` exposes the same interface to external command agents.

### Reward Context

Reward code receives a `ComputerContext` scoped to the same rollout workspace the agent used. Reward functions can run commands, read files, inspect git diffs, and copy artifacts from that state.

### Declarative Tool Dispatch

JSON tasks can define tool schemas and endpoint routes. The dispatcher validates required arguments, calls the endpoint, records latency and response data, and writes `tool.dispatch` events to the trajectory.

### Mock Services

Mock services provide resettable task state. Services expose health, reset, and audit endpoints. Audit logs are collected after the agent run and saved as `audit.json`, so graders can inspect actions the agent actually performed.

### Trace

Every run produces:

- `trajectory.jsonl` raw append-only event stream
- `trajectory.json` typed semantic trajectory with steps, tool calls, observations, metrics, final metrics, and a stable semantic hash
- `diff.patch`
- `reward.json`
- `reward-details.json`
- `score.json`
- `report.json`
- `report.html`
- browser snapshots when browser state is inspected
- `audit.json` when services are used
- `dimensions.json` when dimension scoring is used
- `snapshots/` when post-run environment snapshots are configured

Environment rollouts additionally produce `rollout.json`.

The typed trajectory is the stable research artifact. The raw JSONL records setup, agent execution, verification, changed files, and score items for debugging and compatibility.

### Typed Trajectory Schema

`trajectory.json` uses `agenticevals.trajectory.v1`. It contains:

- `task`: task id, title, and prompt
- `agent`: agent kind, model, provider, and adapter metadata
- `steps`: user messages, agent messages, tool calls, tool results, observations, and outcomes
- `final_metrics`: aggregate tool-call, token, cost, and latency counts
- `semantic_hash`: a deterministic hash that excludes run-specific ids and raw event indices

Deterministic agents such as `noop`, `scripted`, and fixture-backed model agents should produce the same semantic hash for the same task behavior. Live LLM agents are replayable through captured fixtures, not guaranteed hash-deterministic.

### Canonical Trace Schema

Raw trajectory events are normalized into canonical rows with:

- `run_id`
- `task_id`
- `index`
- `timestamp`
- `actor`
- `action_type`
- `name`
- `status`
- `summary`
- `data`

Use `agenticevals normalize <run-dir>` or `agenticevals export-data <run-dir> --format normalized`.

### Model Loop And Tool Parsers

The `model-loop` agent runs a multi-turn loop: model response, parse tool calls, execute actions, append observations, and stop on a final answer. The parser registry supports JSON, XML-style `<tool_call>...</tool_call>`, and fenced markdown JSON tool calls.

### Suites

Suites are JSON files under `configs/suites/`. They run multiple task configs, optional per-task trials, optional parallel workers, and checkpoint resume. Suite runs write `checkpoint.jsonl`, `suite.json`, `results.json`, and `failures.json`. Result rows include bootstrap confidence intervals for pass rate and mean score. Failure clustering uses stable categories: `agent_runtime_error`, `missing_or_invalid_artifact`, `policy_violation`, `browser_state_failure`, `command_check_failed`, `dimension_failure`, `tool_failure`, and `unknown`.

### Viewer

`agenticevals view <run-dir>` writes a self-contained HTML trajectory viewer from canonical trace rows. `agenticevals review <suite-run-dir>` writes a suite review UI with aggregate metrics, failure clusters, task rows, and browser-local reviewer notes.

`agenticevals review <suite-run-dir> --filter status=failed` prints filtered cross-run summaries for command-line failure inspection.

### Release Gate

The v0.1 release gate is explicit:

- at least three agent baselines on one suite or dataset
- bootstrap confidence intervals present for baseline rows
- judge calibration report present
- Cohen's kappa at or above the configured threshold
- for binary calibrations, judge TPR and TNR at or above 0.70 (when reported)

If the judge kappa is weak, or its TPR/TNR show it is lenient on one class, do not use LLM judge scores as headline results.

### Verifiers And Rewards

Task runs are graded by verifier components that consume the typed trajectory and final environment state. The built-in verifier types are:

- `programmatic`: tests, linters, benchmark scripts, or precomputed command results
- `state_check`: file checks, browser checks, git policy, service audits, and expected final state
- `tool_calls`: required tools, forbidden tools, argument-schema checks, dispatch success, and tool safety policy
- `trajectory_check`: final-message, step-budget, tool-call-budget, and status checks over `trajectory.json`
- `llm_rubric`: optional rubric judging with fixture mode for deterministic tests

Verifier output is written to `reward.json` and `reward-details.json`. `score.json` is still emitted as a compatibility view of the same verifier results.

`dimensions.json` is retained for standardized completion, robustness, communication, and safety summaries when tool or service audit fields are present.

### Data Export

Run directories can be exported as:

- full rollout trajectories
- ShareGPT-style conversations
- action and observation rows
- reward component rows
- RL-oriented trajectory rows with prompt, observations, actions, final state, reward, reward components, failure category, hard-negative tags, and metadata
- preference pairs for repeated attempts on the same task
- hard-negative rows for failed or unsafe trajectories
- dataset manifests and dataset cards
- reward recomputation artifacts when a saved task workspace contains `task.json`
- normalized canonical rows
- training rows
- gzip-compressed JSONL

`agenticevals export-dataset <run-dir>` writes `rl.jsonl`, `preferences.jsonl`, `hard_negatives.jsonl`, `manifest.json`, and `DATASET.md`.

`agenticevals improve-loop <run-dir>` mines hard negatives into failure clusters and candidate regression eval records. The intended loop is: run agents, collect failures, promote useful candidate tasks, rerun, then export a new dataset.

Direct model calls use a small cost, rate-limit, and cache layer controlled by `AGENTICEVALS_CACHE_DIR`, `AGENTICEVALS_USE_CACHE`, and `AGENTICEVALS_MIN_REQUEST_INTERVAL_SECONDS`.

### Benchmark Adapters

`examples.tau_retail_env:TauRetailEnv` provides a tau-retail shaped smoke adapter: customer request, retail policy, mutable order database, and reward by database-state comparison plus policy compliance. `AGENTICEVALS_TAU_RETAIL_TASKS` can point at an adapter-compatible JSON export with `id`, `request` or `instruction`, `db` or `initial_state`, and `expected_db` or `goal_state` fields. It is not a canonical upstream tau2 simulator adapter yet.

## Why This Exists

Most eval harnesses grade responses. Agentic systems need evals that inspect whether the agent completed work in an environment. The key question is:

> Did the AI agent move from user intent to verified useful result?
