# Task Configs

`agenticevals` tasks are JSON specs. A task defines the user intent, the controlled workspace, the agent to run, verification checks, policies, and scoring weights.

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
