# CLAUDE.md (development sessions only)

Project 4 (BIOS 774): an agent that decides how to analyze a complex biomedical
dataset given a user objective. Due **Mon Nov 30, 11:59 PM ET**, together with
Project 3. The plan is in `docs/PLAN.md`.

Pipeline runs use `--setting-sources local`, so they never load this file.
Change pipeline behavior in `AGENT.md`, `.claude/skills/`, `.claude/agents/` or
`scripts/`, not here.

## Working rules

- Plan dates are an ordering, not gates. Keep going between milestones.
- Stop and ask the user only when:
  - a decision changes the objectives or the scope,
  - an action cannot be undone,
  - the user is needed (Longleaf, credentials, logins), or
  - tests still fail after one bounded fix attempt.
- Commit at each milestone, with a message saying what now works. Never push;
  the user pushes.
- The user is mostly away from Wed Nov 25 to Sun Nov 29. Record decisions,
  open questions and results in `docs/PROGRESS.md` so they can catch up.

## Other projects are read-only

- `../dim-reduction-agent` (P1), `../gen-model-agent` (P2),
  `../representational-learning-agent` (P3) and `../reference_repos` are
  references. Never edit, commit, or run anything inside them.
- Copy what is needed into this repo and adapt it here. Never import across
  folders or add another project to a path.
- P3 is still changing. Do not copy P3 code by hand. `scripts/sync_p3.sh
  <commit>` copies a pinned P3 commit into `tools/p3/` near the end; until then
  P3 is reached only through `scripts/tools/p3_adapter.py` (JSON in, JSON out),
  which falls back to the cached frozen embeddings in `cache/`.

## How it runs

- Entry point: `./run.sh <dataset> "<objective>" [--report-number N] [--budget-min M]`.
  It makes `venv/`, verifies the data with `scripts/fetch_data.py`, then runs
  `claude -p` headless with `AGENT.md`, `.claude/pipeline-settings.json`
  (made absolute by `bin/runtime_settings.py`) and the critic in
  `.claude/agents/`.
- The pipeline must not read `docs/`, `resources/`, `README.md` or this file;
  `tests/test_settings.py` checks the deny rules.
- Every Python command goes through `venv/bin/python` (a bare `pip install`
  once changed a system conda env). Read downloaded data with `-I`.
- Skills and `AGENT.md` stay generic: no dataset names, gene names or file
  names. A test greps for them.
- One bounded fix per failure, then record it and continue. One failed tool
  never stops the report.
- "Default" or "standard choice" is never a reason; validators reject it.
- Read the figures and rendered PDF pages, not only exit codes.
- Long runs: laptop on AC power with the lid open (`caffeinate -s` only holds
  on AC).

## Datasets

- `rocov2`: generated_report_1 (radiology image + caption).
- `papalexi_eccite`: generated_report_2 (scPerturb ECCITE-seq, RNA + protein,
  CRISPR perturbations with non-targeting controls).
- `pbmc_multiome`: third test dataset and fallback, not submitted.

Data files are never committed; each dataset's `DATA_DESCRIPTION.md` is.

## Deliverables

`reports/<dataset>/generated_report_<N>/` for two datasets, `report.pdf` (at
most 4 pages, agent design focus), and slides for an 8-minute talk.
