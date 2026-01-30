#!/usr/bin/env bash
set -euo pipefail

python3 -m agenticevals validate configs/tasks/patch-python-bug.json
python3 -m agenticevals run configs/tasks/patch-python-bug.json

