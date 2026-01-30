#!/usr/bin/env bash
set -euo pipefail

pattern='Atropos|Nous|MLGym|strongest|world-class|research-grade'
targets=(README.md docs)
if [[ -d benchmarks ]]; then
  targets+=(benchmarks)
fi

if rg -n "$pattern" "${targets[@]}"; then
  echo "Public docs contain forbidden borrowed framing or inflated claims." >&2
  exit 1
fi
