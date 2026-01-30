from __future__ import annotations

from pathlib import Path

from agenticevals.trace import Trajectory
from agenticevals.utils import CommandResult, run_command


class Shell:
    def __init__(self, cwd: Path, trace: Trajectory):
        self.cwd = cwd
        self.trace = trace

    def run(self, command: str, timeout: int, event_type: str = "shell.command") -> CommandResult:
        self.trace.add(event_type, command=command, cwd=str(self.cwd), timeout=timeout)
        result = run_command(command, self.cwd, timeout=timeout)
        self.trace.add(
            f"{event_type}.result",
            command=command,
            returncode=result.returncode,
            stdout=result.stdout[-4000:],
            stderr=result.stderr[-4000:],
            timed_out=result.timed_out,
        )
        return result

