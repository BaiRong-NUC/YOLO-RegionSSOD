# Project installation

- Installed: 2026-11-02 (Asia/Shanghai).
- Scope: `D:/Python/YOLO-RegionSSOD/.agents/skills/academic-research-suite`.
- Source: https://github.com/Imbad0202/academic-research-skills-codex
- Source directory: `skills/academic-research-suite`.
- Commit: `a1556557887f345aac99bdc6027e008e91fef236`.
- Adapter version: `3.24.0`.
- Author: Cheng-I Wu (Imbad0202).
- License: CC BY-NC 4.0; see `LICENSE.adapter` and the retained licenses
  and notices under `ars/`.

Installed with the bundled Codex skill-installer and an explicit project
destination. All 2,657 upstream files were verified against their Git blob
SHA-1 identities. Upstream content is unchanged. This installation note and
the copy of the adapter repository's root license are local additions.

Windows installation required extended-length absolute paths and backslashes
in the installer source path. The downloaded archive and verification records
were kept under the project's ignored `tmp/` directory.

## Usage

In this project, invoke `$academic-research-suite` followed by the requested
task, for example `ars-plan` or `ars-citation-check`. The router loads the
relevant workflow on demand.

The repository's `AGENTS.md`, existing paper locations, experiment budgets,
and evidence rules continue to govern project work. Installing this skill
does not start a research workflow or authorize training.

Optional runtime flags, hooks, provider credentials, and global Codex
configuration were not enabled or modified.

## Validation

- Upstream file verification: 2,657 checked, zero missing or modified.
- Existing project suite: `python -m pytest`, 179 passed.
- Research workflows themselves have not been exercised by this installation.

For updates, select and inspect a new upstream revision, preserve these
installation records, and repeat the integrity and project checks.
