from __future__ import annotations

from typing import Any


BUILTIN_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "terminal",
        "description": "Run a shell command in the workspace. Returns stdout, stderr, and the exit code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
                "timeout": {"type": "integer", "description": "Timeout in seconds.", "default": 60},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Workspace-relative path."}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write a UTF-8 text file in the workspace. Overwrites if it exists.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path."},
                "content": {"type": "string", "description": "File content."},
            },
            "required": ["path", "content"],
        },
    },
]


BUILTIN_TOOL_NAMES = {schema["name"] for schema in BUILTIN_TOOL_SCHEMAS}


def execute_builtin_tool(tool_name: str, arguments: dict[str, Any], computer: Any) -> dict[str, Any]:
    if computer is None:
        return {"ok": False, "error": "no computer interface"}
    if tool_name in {"terminal", "shell", "exec"}:
        result = computer.terminal(str(arguments.get("command", "")), timeout=int(arguments.get("timeout", 60)))
        return {"ok": result.ok, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if tool_name in {"read_file", "read"}:
        return {"ok": True, "content": computer.read_file(str(arguments["path"]))}
    if tool_name in {"write_file", "write"}:
        computer.write_file(str(arguments["path"]), str(arguments.get("content", "")))
        return {"ok": True}
    if hasattr(computer, "dispatch_tool"):
        result = computer.dispatch_tool(tool_name, arguments)
        return result.to_dict()
    return {"ok": False, "error": f"unknown tool: {tool_name}"}
