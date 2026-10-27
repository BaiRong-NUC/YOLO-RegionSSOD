# YOLO-RegionSSOD Session Guide

## Active project

The active paper is **region-partial semi-supervised tiny-object detection on
high-resolution UAV imagery**, implemented in this repository. The detector is
YOLO11s on VisDrone. This is an image-detection project; video is out of scope.

`docs/RGB_IR_Reliability_Experiment_Design.md` is a separate, historical
RGB-IR reliability design. Do not mix it, its datasets, or its results with
the RegionSSOD paper.

## Read first

1. `docs/paper_completion_plan.md` — master status, gates, writing plan, and
   the latest progress log.
2. `docs/experiment_plan_8weeks.md` — scientific protocol and experiment design.
3. `docs/baseline_protocol.md` — reproducible B0/B1/B2 commands.
4. `README.md` — environment and quick-start commands.

Before editing, run `git status --short --branch` and inspect existing changes.
Never revert user or other-session changes without authorization.

## Current evidence rules

- `data/`, `runs/`, and `weights/` are intentionally ignored and may be local-only.
- The existing 20-image/1-epoch smoke run proves the toolchain, not the method.
- Smoke metrics must never be inserted into paper result tables.
- A paper result requires the fixed split, fixed config, full-image val, and a
  traceable weight/log.
- Do not mix Ultralytics mAP with VisDrone official AP without an explicit note.
- The current VisDrone test-dev has already been inspected in earlier work and
  is not a new blind test set.

## Experiment boundaries

- Annotation budgets 5%, 10%, and 20% mean labeled **area ratio**, not image count.
- Unlabeled tiles contain hidden GT and must not be treated as pure background.
- Main experiments use YOLO11s, 640 training, 100 epochs, SGD, batch 8, and full
  val at 960.
- B0 is the full-supervision upper reference; B1 is partial-supervision; B2 is
  global-threshold self-training; P1 is the proposed scale-aware method.
- P1's 30-epoch pilot and full-val evaluation were accepted on 2026-10-27:
  mAP50-95 +0.3349 percentage points and small-object recall +1.8776 points vs B2.
  The original Gate 3 threshold was not met. The user chose to retain the positive
  preliminary result and prioritize writing/analysis; see paper_completion_plan.md 3.8.
- Do not add backbone, P2, attention, or multi-module changes before the core
  P1 mechanism passes Gate 3.

## Working protocol

- Use the existing `yolo` Conda environment.
- Run `python -m pytest` after source changes.
- Store checkpoints, logs, generated data, and caches only in ignored directories.
- After a meaningful work session, append a dated entry to section 10 of
  `docs/paper_completion_plan.md`.
- Report clearly whether work is complete, partially implemented, or only a
  smoke test.
- Keep the paper's reproducible command path current in `README.md` and
  `docs/baseline_protocol.md`.
