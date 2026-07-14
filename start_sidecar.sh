#!/usr/bin/env bash
# Launch the human-detection sidecar for the pilot dashboard.
#
# This is the ONE command a pilot (or dev) needs. On first run it bootstraps a
# virtualenv and installs dependencies; subsequent runs skip straight to launch
# unless requirements.txt changed. The WALDO model auto-downloads on first
# inference, so there is nothing else to set up.
#
#   ./start_sidecar.sh                 # defaults (127.0.0.1:8765)
#   ./start_sidecar.sh --port 9000     # forwarded to the sidecar
#   ./start_sidecar.sh --reinstall     # force a dependency reinstall, then run
#   HUMAN_DETECTION_CONF=0.25 ./start_sidecar.sh
#
# Env knobs:
#   PYTHON_BIN   python interpreter used to create the venv (default: python3)
#
# Stops cleanly on Ctrl+C. Leave it running in a spare terminal for your shift,
# or install it as a login service so it's always up — see
# scripts/install_sidecar_service.sh (macOS) or docs/sidecar-autostart.md.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$REPO_DIR/.venv"
REQ_FILE="$REPO_DIR/requirements.txt"
# Records the hash of the requirements we last installed so we only reinstall
# when they actually change (keeps day-to-day startup near-instant).
STAMP_FILE="$VENV_DIR/.requirements.sha256"

# --- flag handling: peel off --reinstall before forwarding the rest ----------
FORCE_REINSTALL=0
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--reinstall" ]]; then
    FORCE_REINSTALL=1
  else
    ARGS+=("$arg")
  fi
done

hash_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

# --- 1. virtualenv -----------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[sidecar] creating virtualenv at .venv (first run)..."
  if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    echo "[sidecar] ERROR: failed to create venv with '$PYTHON_BIN'." >&2
    echo "[sidecar] Install Python 3.10+ or set PYTHON_BIN to a valid interpreter." >&2
    exit 1
  fi
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# --- 2. dependencies (only when requirements change) -------------------------
current_hash="$(hash_file "$REQ_FILE")"
need_install=0
if [[ "$FORCE_REINSTALL" == "1" ]]; then
  need_install=1
elif [[ ! -f "$STAMP_FILE" ]] || [[ "$(cat "$STAMP_FILE" 2>/dev/null)" != "$current_hash" ]]; then
  need_install=1
fi
if [[ "$need_install" == "1" ]]; then
  echo "[sidecar] installing dependencies (this only happens on first run or after a requirements change)..."
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r "$REQ_FILE"
  echo "$current_hash" >"$STAMP_FILE"
fi

# --- 3. launch ---------------------------------------------------------------
# `${ARGS[@]+...}` guard: on macOS's stock bash 3.2, expanding an empty array
# under `set -u` would abort with "unbound variable".
exec python scripts/run_sidecar.py ${ARGS[@]+"${ARGS[@]}"}
