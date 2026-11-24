# YOLO-RegionSSOD Session Guide

## Active project

The active paper is **region-partial semi-supervised tiny-object detection on
high-resolution UAV imagery**, implemented in this repository. The detector is
YOLO11s on VisDrone. This is an image-detection project; video is out of scope.

The historical RGB-IR design is in the local archive under `cache/` and in
commit `0965ca7`. Keep its datasets and results separate from RegionSSOD.

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
- Before making a novelty claim, read the closest related work S3OD in
  `docs/paper/{related_work,references}.md`. It already combines scale-aware
  pseudo-label thresholds and teacher-guided negative learning; do not claim
  those ideas are first introduced here.
- Do not add backbone, P2, attention, or multi-module changes before the core
  P1 mechanism passes Gate 3.

## Working protocol

- For each paper, put reproduction code in `reproduce/<paper>/` and its
  baseline and method diagrams (SVG and PNG) in `docs/reproduce/<paper>/`.
  Put project-authored external-paper launch and evaluation scripts in
  `scripts/reproduce/<paper>/` and their tests in `tests/reproduce/<paper>/`.
  Use the same paper slug across these directories.
  Register each new paper in `docs/reproduce/README.md`. Start its card with two separate
  diagrams of the paper's own baseline and proposed method. Use the same canvas,
  layer coordinates and numbering for the changed training or inference sites.
  Place each number at the actual site and on its detail panel. Trace both models'
  inputs, modules, supervision and outputs. For each change, explain the
  baseline failure it addresses and cite the paper's corresponding evidence.
  Show the detector stages and their connections, annotate what each module
  does, and expand the paper-specific modules at their actual points of use.
  Follow `docs/reproduce/s3od/s3od-architecture.svg` for the layout: a full-height
  backbone on the left, a compact head and loss overview above the module
  details, and layer-by-layer connections across the lower half.
  Record the paper version and distinguish reported results from local runs.
  Keep superseded figure versions in `tmp/`. Preserve paths already used by
  the editor for local previews and drawing scripts when organizing these files.

- Use the existing `yolo` Conda environment.
- As of 2026-10-30, the user requires at least 30 complete paired epochs before
  performance decisions. Short runs are engineering checks only. Autonomous
  candidate runs currently use exactly 30 epochs; larger training is handed off.
- Commit meaningful changes with Chinese messages and push, as newly authorized
  on 2026-10-30. Create a temporary monitor when training starts and remove it
  after terminal-state handling.
- During real training, the user allows either waiting with the persistent goal
  paused or using the wait to verify relevant papers. Keep the actual training
  process and its monitor running; avoid repetitive progress-only turns and
  parallel experiments. If a paused goal cannot be resumed by available
  controls, report that instead of claiming it was resumed.
- Run `python -m pytest` after source changes.
- Keep the active tree focused on the paper under development, accepted results,
  their controls, reproducibility dependencies and tests. When an experiment
  stops without an improvement to retain, record the outcome and comparison in
  section 10 of the completion plan. Archive its dedicated code, configs, tests
  and reports in `cache/<date>-<experiment>/` under their original relative
  paths. Record the source commit and file hashes. Check shared imports, frozen
  receipt paths, paper links and active tests before removing the active copies.
  Keep shared files and paper evidence at their required paths. Verify the
  archive and the retained tests before committing the cleanup.
- Store checkpoints, logs, generated data, and caches only in ignored directories.
- After a meaningful work session, append a dated entry to section 10 of
  `docs/paper_completion_plan.md`.
- Report clearly whether work is complete, partially implemented, or only a
  smoke test.
- Keep the paper's reproducible command path current in `README.md` and
  `docs/baseline_protocol.md`.
