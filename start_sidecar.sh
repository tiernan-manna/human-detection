#!/usr/bin/env bash
# Launch the human-detection sidecar for the pilot dashboard.
# Run this once at the start of your shift; leave it running in a spare terminal.
#
#   ./start_sidecar.sh                 # defaults (127.0.0.1:8765)
#   ./start_sidecar.sh --port 9000     # override port
#   HUMAN_DETECTION_CONF=0.25 ./start_sidecar.sh
#
# Stops cleanly on Ctrl+C.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Activate the venv if it exists; otherwise assume the user wants the system
# Python (for CI, containers, etc.).
if [[ -d ".venv" ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
elif [[ -d "venv" ]]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
fi

exec python scripts/run_sidecar.py "$@"
