from __future__ import annotations

import subprocess

from agenticevals.schema import VerifierSpec
from agenticevals.verifiers.base import BaseVerifier, VerifierContext, criterion
from agenticevals.verifiers.schema import CriterionResult


class ProgrammaticVerifier(BaseVerifier):
    verifier_type = "programmatic"

    def verify(self, context: VerifierContext, spec: VerifierSpec) -> list[CriterionResult]:
        if "results" in spec.config:
            return self.from_precomputed(spec, spec.config["results"])
        commands = list(spec.config.get("commands", []))
        if not commands:
            return [
                criterion(
                    name=spec.name or "programmatic",
                    verifier_type=self.verifier_type,
                    score=1.0,
                    weight=spec.weight,
                    passed=True,
                    detail="no programmatic checks configured",
                    required=spec.required,
                )
            ]
        results = []
        timeout = int(spec.config.get("timeout", 60))
        for command in commands:
            proc = subprocess.run(
                str(command),
                cwd=str(context.workspace),
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            detail = f"returncode={proc.returncode}"
            if proc.stderr:
                detail += f"; stderr={proc.stderr[-500:]}"
            results.append((str(command), proc.returncode == 0, detail))
        return self.from_precomputed(spec, results)

    def from_precomputed(self, spec: VerifierSpec, results: list[tuple[str, bool, str]]) -> list[CriterionResult]:
        if not results:
            return [
                criterion(
                    name=spec.name or "programmatic",
                    verifier_type=self.verifier_type,
                    score=1.0,
                    weight=spec.weight,
                    passed=True,
                    detail="no checks configured",
                    required=spec.required,
                )
            ]
        each = spec.weight / len(results)
        criteria = []
        for name, passed, detail in results:
            criteria.append(
                criterion(
                    name=f"{spec.name or 'programmatic'}:{name}",
                    verifier_type=self.verifier_type,
                    score=1.0 if passed else 0.0,
                    weight=each,
                    passed=passed,
                    detail=detail,
                    required=spec.required,
                    evidence={"check": name},
                )
            )
        return criteria
