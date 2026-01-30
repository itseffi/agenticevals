# Task Configs

`agenticevals` tasks are JSON specs. A task defines the user intent, the controlled workspace, the agent to run, verification checks, policies, and reward verifiers.

Minimal shape:

```json
{
  "id": "my-task",
  "title": "Readable title",
  "prompt": "The task given to the AI agent.",
  "workspace": {
    "fixture_path": "path/to/fixture",
    "setup": ["npm install"]
  },
  "agent": {
    "kind": "codex"
  },
  "checks": {
    "commands": ["npm test"],
    "files": [],
    "browser": []
  },
  "policies": {
    "forbidden_paths": ["tests/fixtures"],
    "max_changed_files": 8
  }
}
```

If `verifiers` is omitted, the runner creates default programmatic, state, and tool-call verifiers from `checks`, `policies`, `tools`, `expected_actions`, and `safety_checks`. Explicit verifier types are `programmatic`, `state_check`, `tool_calls`, `trajectory_check`, and `llm_rubric`.

Supported agent kinds:

- `scripted`: deterministic agent used for fixtures and framework tests.
- `noop`: no-op baseline.
- `command`: task-provided command template.
- `codex`: command template from `AGENTICEVALS_CODEX_COMMAND`.
- `claude-code`: command template from `AGENTICEVALS_CLAUDE_COMMAND`.
- `http`: external HTTP agent.
- `direct-model`: text-only OpenAI baseline.
- `model-loop`: parser-based tool loop.
- `openai`: OpenAI native tool loop.
- `gemini`: Gemini native tool loop.
- `claude`: Anthropic native tool loop.
