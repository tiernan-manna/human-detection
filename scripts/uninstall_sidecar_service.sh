#!/usr/bin/env bash
# Remove the human-detection sidecar launchd login service installed by
# scripts/install_sidecar_service.sh. Safe to run even if it was never
# installed (it just reports "not installed").
#
#   scripts/uninstall_sidecar_service.sh

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[uninstall] macOS-only. See docs/sidecar-autostart.md for Windows." >&2
  exit 1
fi

LABEL="com.manna.human-detection-sidecar"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [[ ! -f "$PLIST" ]]; then
  echo "[uninstall] '$LABEL' is not installed (no plist at $PLIST). Nothing to do."
  exit 0
fi

launchctl unload "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
echo "[uninstall] Removed '$LABEL'. The sidecar will no longer start at login."
echo "[uninstall] (Logs left in ~/Library/Logs/human-detection — delete manually if desired.)"
