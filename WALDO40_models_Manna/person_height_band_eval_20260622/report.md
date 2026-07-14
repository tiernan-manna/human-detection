# WALDO4.0 Person 10-30 px Eval

Date: 2026-06-22

This eval scores only `person` ground-truth boxes that measure `10-30 px` tall after 320-letterbox scaling. Person boxes outside that band are ignored, not counted as false positives.

Dataset slice:

- Validation images: `1339`
- Target person boxes: `874`
- Ignored outside-band person boxes: `4828`
- Images with target-band person boxes: `136`

| Model | AP50 | AP50-95 | AR50-95 | F1 @0.10 | F1 @0.25 | Best F1 |
|---|---:|---:|---:|---:|---:|---:|
| `s-p2` | 0.66898 | 0.36045 | 0.56728 | 0.37462 | 0.64036 | 0.66667 @ 0.32 |
| `m-p2` | 0.74461 | 0.42701 | 0.60892 | 0.39692 | 0.65257 | 0.72366 @ 0.45 |
| `l-p2` | 0.75341 | 0.43643 | 0.60572 | 0.41679 | 0.65178 | 0.73632 @ 0.39 |

Verdict: `l-p2` is the best model on this custom 10-30 px person band by AP50-95 and swept F1. `m-p2` is effectively tied with `l-p2` at fixed `conf=0.25`.

Files:

- `summary.csv`: combined source metrics
- `summary.json`: combined source metrics
- `*_summary.csv/json`: per-model summaries
- `*_thresholds.json`: per-model IoU threshold details
- `eval_person_height_band.py`: evaluator used for this run
