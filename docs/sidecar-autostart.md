# Running the sidecar on a pilot machine

The pilot dashboard is served from EC2; the **only** thing that runs locally on
the pilot's computer is this detection sidecar. It listens on
`ws://127.0.0.1:8765` and the dashboard connects to it when the pilot toggles
**Human Detection** on. If the sidecar isn't running, the dashboard toggle just
shows "Detection service not running" — nothing else breaks.

This doc covers (1) starting it by hand and (2) making it start automatically.

> **Opt-in by design.** The auto-start installers below only do something when a
> human runs them on a pilot machine. They are not wired into any build,
> checkout, or `pip install` step, so having them in the repo has no effect on
> developers who don't touch the sidecar.

## 1. Start it by hand

First run bootstraps a virtualenv, installs dependencies, and downloads the
WALDO model automatically. Subsequent runs start immediately.

**macOS / Linux**

```bash
./start_sidecar.sh
```

**Windows (PowerShell)**

```powershell
.\start_sidecar.ps1
```

Leave it running in a terminal for the shift. `Ctrl+C` stops it cleanly.
Flags (e.g. `--port 9000`) are forwarded to the sidecar; `--reinstall` /
`-Reinstall` forces a dependency refresh.

## 2. Auto-start at login (recommended for pilots)

### macOS (launchd)

```bash
scripts/install_sidecar_service.sh      # installs + starts now, restarts on crash
scripts/uninstall_sidecar_service.sh    # removes it
```

Verify / inspect:

```bash
launchctl list | grep human-detection
tail -f ~/Library/Logs/human-detection/sidecar.out.log
```

The service runs `start_sidecar.sh` at login with `KeepAlive`, so it survives
crashes and reboots.

### Windows (Task Scheduler)

No tested installer script is shipped yet — register a logon task pointing at
the launcher (run once, from the repo root, in an elevated PowerShell). Validate
on the actual pilot hardware before relying on it:

```powershell
$repo = (Get-Location).Path
schtasks /Create /TN "MannaHumanDetectionSidecar" /SC ONLOGON /RL LIMITED `
  /TR "powershell -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$repo\start_sidecar.ps1`""
```

Remove it with:

```powershell
schtasks /Delete /TN "MannaHumanDetectionSidecar" /F
```

## Model & tuning defaults

Detection behaviour (model file, confidence thresholds, gates) is owned by the
sidecar config (`src/human_detection/config.py`) and the WALDO weights — that's
the model workstream, not this ops tooling. The launcher deliberately sets no
model/threshold env vars so it always runs whatever `config.py` defines as the
current best defaults.
