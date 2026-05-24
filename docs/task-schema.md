# Task Schema

Task specs are JSON files.

## Required Fields

- `id`: stable machine-readable ID
- `title`: human-readable title
- `prompt`: the instruction given to the AI agent
- `workspace.fixture_path`: source directory copied into an isolated run workspace

## Workspace

```json
{
  "workspace": {
    "fixture_path": "fixtures/app",
    "setup": ["npm install"],
    "dev_server": {
      "command": "npm run dev",
      "url": "http://localhost:3000"
    }
  }
}
```

When browser checks are configured, `dev_server.command` is started after the agent finishes and stopped after browser verification. This makes the browser layer a final-state verifier for UI and HTTP-observable behavior.

## Agent Commands

Command agents run an external shell command from the copied workspace:

```json
{
  "agent": {
    "kind": "command",
    "command": "my-agent --task {task_path} --result {result_path}"
  }
}
```

Templates can use `{prompt_path}`, `{task_path}`, `{result_path}`, `{trace_path}`, and `{workspace}`. The runner also sets matching `AGENTICEVALS_*` environment variables. Command agents should write:

- `.agenticevals/result.json`: `{"ok": true, "final_message": "..."}`
- `.agenticevals/agent-trace.jsonl`: optional JSONL events for commands, edits, tool calls, and observations

Built-in adapter names `codex` and `claude-code` are command adapters configured through environment variables. Provider-native tool-loop adapters are `openai`, `gemini`, and `claude`.

HTTP agents use the same task contract over JSON:

```json
{
  "agent": {
    "kind": "http",
    "command": "http://127.0.0.1:8000/run"
  }
}
```

If `command` is omitted, the adapter reads `AGENTICEVALS_HTTP_AGENT_URL`.

`model-loop` runs a parsed tool-use loop. For offline tests, set `agent.model` to `fixture` and provide assistant responses in `agent.script`:

```json
{
  "agent": {
    "kind": "model-loop",
    "model": "fixture",
    "script": [
      {
        "content": "<tool_call>{\"tool_name\":\"write_file\",\"arguments\":{\"path\":\"answer.txt\",\"content\":\"done\\n\"}}</tool_call>"
      },
      {
        "content": "Created answer.txt."
      }
    ]
  }
}
```

The same loop can execute built-in computer calls such as `write_file`, `read_file`, and `terminal`, or declared task tools through the dispatcher.

## Checks

```json
{
  "checks": {
    "commands": ["npm test"],
    "files": [
      { "path": "src/app.ts", "contains": "validateUser" }
    ],
    "browser": [
      { "path": "/", "status": 200, "contains": "Dashboard" },
      {
        "path": "/login",
        "actions": [
          { "action": "fill", "name": "email", "value": "agent@example.com" },
          { "action": "submit", "form_index": 0 }
        ],
        "contains": "Signed in"
      }
    ]
  }
}
```

Browser checks use the same stateful browser session for navigation, DOM snapshots, simple form actions, and snapshot artifacts.

## Policies

```json
{
  "policies": {
    "forbidden_paths": ["tests"],
    "max_changed_files": 5,
    "require_changed_files": ["src/app.ts"]
  }
}
```

Policies catch cases where an agent gets a result by changing the wrong state.

## Declarative Tools

Tasks can declare tools and the HTTP endpoints that execute them:

```json
{
  "tools": [
    {
      "name": "gmail_save_draft",
      "description": "Save a Gmail draft.",
      "input_schema": {
        "type": "object",
        "properties": {
          "to": { "type": "string" },
          "subject": { "type": "string" },
          "body": { "type": "string" }
        },
        "required": ["to", "subject", "body"]
      }
    }
  ],
  "tool_endpoints": [
    {
      "tool_name": "gmail_save_draft",
      "url": "http://127.0.0.1:9100/gmail/drafts/save",
      "method": "POST"
    }
  ]
}
```

Tool dispatches are recorded in `trajectory.jsonl` as `tool.dispatch` events and in `trajectory.json` as typed tool-call/tool-result steps.

## Score Weights

The optional `score` block sets the point budget for each default verifier dimension. Magnitudes are relative — the reward is the weight-normalized average across all emitted criteria.

```json
{
  "score": {
    "command_checks": 40,
    "file_checks": 20,
    "browser_checks": 20,
    "git_policy": 20,
    "expected_actions": 25,
    "audit_safety": 25,
    "tool_dispatch": 25,
    "tool_argument": 25,
    "tool_safety": 25
  }
}
```

All nine are configurable; the values above are the defaults. Raise `tool_*` relative to `file_checks`, for example, to weight tool correctness more heavily.

## Verifiers

Runs write `reward.json` and `reward-details.json` from verifier components. If `verifiers` is omitted, the runner builds default verifiers from `checks`, `policies`, `tools`, `expected_actions`, and `safety_checks`.

Explicit verifiers replace the defaults:

```json
{
  "verifiers": [
    {
      "type": "programmatic",
      "name": "tests",
      "weight": 1,
      "commands": ["python -m pytest"]
    },
    {
      "type": "tool_calls",
      "weight": 1,
      "required_tools": ["gmail_save_draft"],
      "forbidden_tools": ["gmail_send_message"]
    },
    {
      "type": "trajectory_check",
      "weight": 1,
      "require_final_message": true,
      "max_tool_calls": 8
    },
    {
      "type": "llm_rubric",
      "name": "helpfulness",
      "weight": 1,
      "rubric": "Score whether the final response accurately describes the completed work."
    }
  ]
}
```

Verifier types:

- `programmatic`: shell commands or precomputed command results
- `state_check`: files, browser checks, git policy, service audits, and expected actions
- `tool_calls`: required tools, forbidden tools, JSON-schema argument checks, dispatch success, and tool safety checks
- `trajectory_check`: checks over the typed `trajectory.json` steps and metrics
- `llm_rubric`: optional LLM judge; fixture scores can be used for deterministic tests

## Sandbox HTTP Interface

`agenticevals run --sandbox-server` starts a persistent workspace-scoped HTTP server and exposes its URL to command agents as `AGENTICEVALS_SANDBOX_URL`. Environment rollouts can use the same interface with `--backend sandbox-http`.

The interface supports:

- `POST /exec`
- `POST /read`
- `POST /write`
- `POST /edit`
- `POST /glob`
- `POST /grep`
- `POST /download`
- `POST /browser/goto`
- `POST /browser/check`

## Services And Audits

Mock services are started before the agent runs, reset between trials, and audited after the agent finishes:

```json
{
  "services": [
    {
      "name": "gmail",
      "command": "python mock_services/gmail/server.py",
      "port": 9100,
      "health_check": "http://127.0.0.1:9100/health",
      "reset_endpoint": "http://127.0.0.1:9100/gmail/reset",
      "audit_endpoint": "http://127.0.0.1:9100/gmail/audit"
    }
  ]
}
```

Audit data is written to `audit.json` and can drive standardized dimension scoring.

## Dimension Summary

```json
{
  "expected_actions": [
    { "service": "gmail", "action_key": "drafts", "required": true, "min_count": 1 }
  ],
  "safety_checks": [
    { "type": "tool_not_called", "tool_name": "gmail_send_message", "max_count": 0 }
  ]
}
```

When these fields are present, the run writes `dimensions.json` with `completion`, `robustness`, `communication`, `safety`, and `task_score`. The reward path still goes through verifiers; `dimensions.json` is a summary artifact.

## Hidden Grader Files And Snapshots

`sandbox_grader_files` are copied into the workspace only after the agent finishes. `env_snapshot_commands` and `env_snapshot_files` collect post-run artifacts into `snapshots/`.

`local_grader_files` are copied into the run directory under `local_grader_files/`, never into the agent workspace.
