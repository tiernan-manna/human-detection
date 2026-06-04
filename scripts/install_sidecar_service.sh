#!/usr/bin/env bash
# Install the human-detection sidecar as a macOS login service (launchd agent)
# so it starts automatically and restarts if it ever crashes — pilots never
# have to remember to run ./start_sidecar.sh.
#
# ┌─ IMPORTANT ────────────────────────────────────────────────────────────┐
# │ This is OPT-IN. Nothing auto-installs: this script only does anything   │
# │ when a human runs it explicitly on a pilot machine. Merging it to main  │
# │ has ZERO effect on developers who don't invoke it — there is no build   │
# │ hook, no postinstall, no import side effect. Run the uninstaller below  │
# │ to remove it.                                                           │
# └────────────────────────────────────────────────────────────────────────┘
#
#   scripts/install_sidecar_service.sh       # install + start now
#   scripts/uninstall_sidecar_service.sh     # stop + remove
#
# After install:
#   launchctl list | grep human-detection    # confirm it's loaded
#   tail -f ~/Library/Logs/human-detection/sidecar.out.log
#
# macOS only. For Windows, see docs/sidecar-autostart.md.

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[install] This installer is macOS-only. See docs/sidecar-autostart.md for Windows." >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_SCRIPT="$REPO_DIR/start_sidecar.sh"
LABEL="com.manna.human-detection-sidecar"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST="$LAUNCH_AGENTS_DIR/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/human-detection"

if [[ ! -f "$START_SCRIPT" ]]; then
  echo "[install] ERROR: start_sidecar.sh not found at $START_SCRIPT" >&2
  exit 1
fi

mkdir -p "$LAUNCH_AGENTS_DIR" "$LOG_DIR"
chmod +x "$START_SCRIPT" 2>/dev/null || true

cat >"$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${START_SCRIPT}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${REPO_DIR}</string>
    <!-- Start at login and keep it alive: if the process exits for any
         reason, launchd brings it straight back. -->
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <!-- Don't hammer respawns if something is fundamentally broken. -->
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/sidecar.out.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/sidecar.err.log</string>
</dict>
</plist>
PLIST_EOF

# Reload idempotently: unload an existing instance (ignore errors) then load.
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo "[install] Installed and started '${LABEL}'."
echo "[install]   plist: $PLIST"
echo "[install]   logs:  $LOG_DIR/sidecar.out.log (and sidecar.err.log)"
echo "[install]   check: launchctl list | grep human-detection"
echo "[install]   remove: scripts/uninstall_sidecar_service.sh"
