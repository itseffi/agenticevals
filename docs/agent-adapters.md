# Agent Adapters

Agent adapters are intentionally thin. `agenticevals` should not depend on one vendor or one coding agent.

List configured adapters:

```bash
python3 -m agenticevals adapters
```

## Built-In Test Adapters

`noop` makes no changes and is useful as a baseline.

`scripted` executes deterministic actions from the task or environment item and is useful as a reference agent for verifier development.

## Codex

Configure:

```bash
export AGENTICEVALS_DEFAULT_AGENT=codex
export AGENTICEVALS_CODEX_COMMAND="codex exec --skip-git-repo-check --sandbox workspace-write --cd {workspace} {prompt}"
```

Run:

```bash
python3 -m agenticevals run configs/tasks/patch-python-bug.json --agent codex
```

Placeholders available in command templates:

- `{prompt}`: shell-quoted prompt text
- `{prompt_path}`: shell-quoted path to a prompt file inside the workspace
- `{workspace}`: shell-quoted workspace path

## Claude Code

Configure:

```bash
export AGENTICEVALS_DEFAULT_AGENT=claude-code
export AGENTICEVALS_CLAUDE_COMMAND="claude -p --permission-mode acceptEdits --add-dir={workspace} {prompt}"
```

Run:

```bash
python3 -m agenticevals run configs/tasks/patch-python-bug.json --agent claude-code
```

## Custom Agents

Use `agent.kind = "command"`:

```json
{
  "agent": {
    "kind": "command",
    "command": "my-agent --cwd {workspace} --prompt-file {prompt_path}"
  }
}
```

The custom command must run inside the workspace and leave its work in the filesystem. `agenticevals` scores the resulting state.

## HTTP Agents

Use `agent.kind = "http"` when the agent is exposed as an HTTP service:

```json
{
  "agent": {
    "kind": "http",
    "command": "http://127.0.0.1:8000/run"
  }
}
```

If `agent.command` is omitted, set:

```bash
export AGENTICEVALS_HTTP_AGENT_URL="http://127.0.0.1:8000/run"
```

The adapter sends JSON with task metadata, the workspace path, optional sandbox URL, and step/time limits. The service should return:

```json
{
  "ok": true,
  "final_message": "Completed the task.",
  "events": []
}
```

`events` is optional. When present, events are appended to the run trace.

## Direct Model

Use `direct-model` when you want a text-only API baseline:

```bash
export OPENAI_API_KEY="..."
export AGENTICEVALS_DEFAULT_MODEL="<openai-model>"
python3 -m agenticevals run configs/tasks/patch-python-bug.json --agent direct-model
```

The direct adapter intentionally does not grant filesystem tools. For coding agents that modify a workspace, use a command adapter.

## Native Provider Tool Loops

Use `openai`, `gemini`, or `claude` when you want provider-native function/tool calls against the same built-in computer tools and declared task tools:

```bash
export OPENAI_API_KEY="..."
python3 -m agenticevals run configs/tasks/model-loop-write-file.json --agent openai

export GEMINI_API_KEY="..."
python3 -m agenticevals run configs/tasks/model-loop-write-file.json --agent gemini

export ANTHROPIC_API_KEY="..."
python3 -m agenticevals run configs/tasks/model-loop-write-file.json --agent claude
```

Model names can be set per task with `agent.model` or through `AGENTICEVALS_OPENAI_MODEL`, `AGENTICEVALS_GEMINI_MODEL`, and `AGENTICEVALS_ANTHROPIC_MODEL`. The docs avoid pinning current provider model names because those change over time.

All three record per-turn token counts, cost, latency, tool calls, and tool results in the typed trajectory. Fixture mode is available by setting `agent.model` to `fixture`.

## Live Adapter Verification

Verify installed command-line adapters against a real task:

```bash
python3 -m agenticevals verify-adapters --task configs/tasks/model-loop-write-file.json
```

The command reports unavailable CLIs and failed runs explicitly. It does not silently replace a missing external agent with a fixture.
