"""Headed-less smoke test for /webdemo using the system Chrome.

Boots the page, waits for the model to load (EP pill), lets the replay run
a few frames, and dumps the stats row + any console errors. Optionally runs
the in-page benchmark.

Usage:
    .venv/bin/python scripts/webdemo_smoke.py [--port 8766] [--bench N]
        [--ep auto|webnn|webgpu|wasm] [--timeout 180]

Requires: pip install playwright (uses channel="chrome", i.e. the installed
Google Chrome — no separate browser download needed).
"""

from __future__ import annotations

import argparse
import json
import sys

from playwright.sync_api import sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--ep", default="auto")
    ap.add_argument("--precision", default=None, choices=["fp32", "fp16"])
    ap.add_argument("--bench", type=int, default=0, help="run the benchmark over N frames")
    ap.add_argument("--pacing", default="paced", choices=["paced", "uncapped"])
    ap.add_argument("--frames", type=int, default=6, help="replay frames to observe")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}/webdemo"
    console: list[str] = []
    errors: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="chrome",
            headless=True,
            args=[
                # WebNN (Chrome flag) + WebGPU in headless.
                "--enable-features=WebMachineLearningNeuralNetwork",
                "--enable-unsafe-webgpu",
                "--use-angle=metal",
            ],
        )
        page = browser.new_page()
        page.on("console", lambda m: console.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url, wait_until="domcontentloaded")

        # Wait for the worker to finish loading the model (pill leaves
        # the pending state).
        page.wait_for_function(
            "() => !document.getElementById('ep-pill').className.includes('pending')",
            timeout=args.timeout * 1000,
        )
        ep = page.text_content("#ep-pill")
        iso = page.evaluate("crossOriginIsolated")
        print(f"ep pill:              {ep}")
        print(f"crossOriginIsolated:  {iso}")
        boot_err = page.evaluate(
            "document.getElementById('boot-error').hidden ? '' : document.getElementById('boot-error').textContent"
        )
        if boot_err:
            print(f"BOOT ERROR:\n{boot_err}")
            return 1

        if args.ep != "auto":
            page.select_option("#ctl-ep", args.ep)
            page.wait_for_function(
                "() => !document.getElementById('ep-pill').className.includes('pending')",
                timeout=args.timeout * 1000,
            )
            print(f"ep pill after switch: {page.text_content('#ep-pill')}")

        if args.precision:
            page.select_option("#ctl-precision", args.precision)
            page.wait_for_function(
                "() => !document.getElementById('ep-pill').className.includes('pending')",
                timeout=args.timeout * 1000,
            )
            print(f"model after switch:   {page.text_content('#stat-model')}")

        # Let the replay process a few frames.
        page.wait_for_function(
            f"() => parseFloat(document.getElementById('stat-avg-ms').textContent) > 0",
            timeout=args.timeout * 1000,
        )
        page.wait_for_timeout(args.frames * 600)
        stats = page.evaluate(
            """() => Object.fromEntries(
                 ['stat-ep','stat-model','stat-shape','stat-fps-in','stat-fps-out',
                  'stat-avg-ms','stat-p95-ms','stat-capacity','stat-stages',
                  'stat-dets-rate','stat-funnel']
                 .map(id => [id, document.getElementById(id).textContent]))"""
        )
        for k, v in stats.items():
            print(f"{k:18s} {v}")

        if args.bench > 0:
            page.fill("#bench-frames", str(args.bench))
            page.select_option("#bench-pacing", args.pacing)
            page.click("#bench-run")
            page.wait_for_function(
                "() => document.getElementById('bench-status').textContent.startsWith('done')"
                " || document.getElementById('bench-status').textContent.includes('Error')"
                " || document.getElementById('bench-status').textContent.includes('timeout')"
                " || document.getElementById('bench-status').textContent.includes('unavailable')",
                timeout=args.timeout * 1000 * 4,
            )
            print(f"bench status:        {page.text_content('#bench-status')}")
            results_visible = page.evaluate(
                "!document.getElementById('bench-results').hidden"
            )
            if results_visible:
                text = page.evaluate(
                    "document.getElementById('bench-results').innerText"
                )
                print("--- bench results ---")
                print(text)

        browser.close()

    if errors:
        print("--- page errors ---")
        print("\n".join(errors))
    bad_console = [c for c in console if c.startswith("[error]")]
    if bad_console:
        print("--- console errors ---")
        print("\n".join(bad_console[:20]))
    print(json.dumps({"pageErrors": len(errors), "consoleErrors": len(bad_console)}))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
