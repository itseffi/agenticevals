---
name: agenticevals-author-eval
description: Create or modify agenticevals task configs, fixtures, hidden graders, mock services, and quality checks. Use when asked to add a new eval, task, grader, benchmark item, or agentic evaluation scenario.
---

# agenticevals Eval Authoring

Use this skill when creating a new eval in this repo.

## Rules

- Keep the eval about agent behavior in an environment, not a single chatbot answer.
- Put agent-visible files under `examples/<task-id>/`.
- Put hidden grader files under `configs/tasks/graders/<task-id>/`.
- Do not place hidden grader files or expected hidden answers inside the fixture.
- Prefer final-state checks plus a git policy.
- Include a scripted success path when practical.
- Verify that a no-op agent fails.
- Keep public docs factual; do not add hype claims or borrowed benchmark framing.

## Fast Path

Scaffold a hidden-grader task:

```bash
python3 -m agenticevals new-task <task-id>
```

Then edit:

- `examples/<task-id>/`
- `configs/tasks/graders/<task-id>/`
- `configs/tasks/<task-id>.json`

Run quality checks:

```bash
python3 -m agenticevals validate configs/tasks/<task-id>.json
python3 -m agenticevals validate-task-quality configs/tasks/<task-id>.json
python3 -m agenticevals run configs/tasks/<task-id>.json --json
python3 -m agenticevals normalize runs/<run-dir>
python3 -m agenticevals view runs/<run-dir>
```

## Task Checklist

A good task has:

- A clear `prompt`.
- A deterministic fixture.
- A hidden grader or mock-service audit when correctness cannot be checked from visible files alone.
- `checks.commands`, `checks.files`, or `checks.browser`.
- `policies.require_changed_files`.
- `policies.max_changed_files` when the expected edit surface is narrow.
- A scripted agent path that passes.
- A no-op path that fails.

## Mock-Service Tasks

For tool/service evals:

- Declare `services`.
- Declare `tools`.
- Declare `tool_endpoints`.
- Add `expected_actions`.
- Add `safety_checks`.
- Ensure service state resets before each run.
- Ensure `audit.json` proves what happened.

## Completion

Before returning, run:

```bash
python3 -m unittest discover -s tests
python3 -m compileall -q agenticevals mock_services examples tests
scripts/check-public-language.sh
```

Return the task path, run directory, and quality-check status.
