# Project 4 plan: biomedical data analysis agent

## Context

P4 is due Mon Nov 30, the same day as P3. P4 must take a dataset and an objective, then on its own profile the data, name the analytical challenges, plan, run tools, evaluate, and write a report. The deliverables are generated_report_1 (ROCOv2), generated_report_2 (scPerturb), a report PDF of at most 4 pages, and slides for an 8-minute talk. The harness reuses the P2/P3 design: `run.sh` → `claude -p` with `AGENT.md`, generic skills, deterministic scripts, validators and a critic. P1 and P2 are finished and can be copied now. P3 is still changing (HEAD `5b71d7e`, with uncommitted work; AGENT.md, validators, report.py and `summary.json` are not built yet), so P3 is used only through a JSON adapter and copied in near the end.

Hard rules: P1, P2, P3 and reference_repos are read-only. Nothing is edited, committed or run inside them; we copy files out. The pipeline cannot read `docs/`, `README.md`, `CLAUDE.md` or dev files.

User time: about 2 h on Mon and Tue, then away until Sun Nov 29. **Target for Tue night: `./run.sh rocov2 "<objective>"` produces a rough report.**

---

## 1. scPerturb recommendation

Source: Zenodo record 13350497 (scPerturb v1.4, RNA and protein .h5ad files) and record 7058382 (ATAC .zip files). Sizes are from the Zenodo API. Cell and perturbation counts come from the papers and memory, so they are marked "verify". The fetch step will confirm them.

| Candidate | Files (size) | Cells | Perturbations | Controls | Second modality |
|---|---|---|---|---|---|
| **PapalexiSatija2021_eccite** | RNA 147 MB + protein 1.2 MB | ~20.7k (verify) | ~25 genes in the IFN-γ / PD-L1 pathway, ~4 guides each (verify) | non-targeting (NT) guides | **Yes**: surface protein (ADT, 4 markers incl. PD-L1) from ECCITE-seq |
| PapalexiSatija2021_eccite_arrayed | RNA 52 MB + protein 0.55 MB | a few thousand (verify) | few (verify) | yes (verify) | Yes: ADT |
| FrangiehIzar2021 | RNA 1.46 GB + protein 25 MB | ~218k | ~250 genes x 3 conditions (control, IFN-γ, T-cell co-culture) | NT | Yes: ~20 proteins (CITE-seq) |
| MimitouSmibert2021 | 834 MB zip (ATAC record) | ~? | few | yes | ATAC + protein (ASAP-seq), but zip, not h5ad |

**Recommendation: PapalexiSatija2021_eccite** (dataset name `papalexi_eccite`).
- Two modalities in the same cells (RNA + protein), small enough for the laptop (148 MB).
- It has the challenges P4's new tools are for: NT controls vs perturbed cells (experimental design), replicates (batch), unequal cells per target (imbalance), and, as noted in the original paper, strong cell-cycle and replicate variation that hides weak perturbation effects. That is the case contrastive PCA (foreground = perturbed, background = NT controls) is meant for.
- The protein panel is small (4 markers), so "integrate the second modality with a learned joint model" is a poor choice here. The agent should pick protein-level DE and effect sizes instead. That gives the report a clear example of the agent choosing differently by data.
- Backup if something is wrong with the file: FrangiehIzar2021 (heavier, 3 conditions, 20 proteins).

## 2. Dataset roles and default objectives

| Dataset | Role | Data source for dev | Default objective (user-editable) |
|---|---|---|---|
| `rocov2` | generated_report_1 | APFS clone (`cp -c`) of `../representational-learning-agent/data/rocov2` plus its `DATA_DESCRIPTION.md` | "Find what structure separates the radiology images and captions, check whether image and caption representations agree, and say which labels each modality can predict." |
| `papalexi_eccite` | generated_report_2 | Zenodo download via `fetch_data.py` | "Find which CRISPR perturbations change the transcriptome and surface proteins relative to non-targeting controls, separating perturbation effects from replicate and cell-cycle variation." |
| `pbmc_multiome` | third test and fallback, not submitted | `cp -c` from P3 `data/pbmc_multiome` | "Identify cell populations and check whether RNA and ATAC agree on them." |

`cp -c` makes copy-on-write clones: no extra disk and fully independent of P3. The ROCOv2 frozen-embedding cache (`cache/rocov2/{biomedclip_frozen,dinov2_vitb14_frozen}`, 74 MB) is data, not code, and is cloned the same way so ROCOv2 works before the P3 sync.

## 3. Reuse list

**Copied now from P2 (`../gen-model-agent`), adapted:**
- Harness: `run.sh`, `bin/agent_stream.py`, `bin/agents_json.py`, `bin/runtime_settings.py`, `.claude/settings.json` (dev) and `.claude/pipeline-settings.json` (env prefix renamed to `BIOAGENT_`).
- `tests/test_settings.py` (docs/README/CLAUDE.md read deny, no `cd`, no `git`, no writes to data/cache/reports).
- `scripts/fetch_data.py` (stdlib only; parses `## Files`, checks md5/sha256), `scripts/loaders.py` (h5ad/npy/csv driven by `## Loading`), `scripts/doctor.py`.
- `scripts/profiler.py` as the base (it already has `class_counts`, `imbalance_ratio`, `is_nonneg_integer`, `modality`).
- Validator pattern from `validate_objective.py` / `validate_plan.py` (reject "default"/"standard" reasons; check cited metric paths exist).
- `scripts/report.py` (refuses untraced numbers in `findings.md`; Contents page; glossary; wrap by measured width).
- Critic pattern (`.claude/agents/gen-critic.md`) and skill structure (`objective-interpretation`, `reporting`).

**Copied now from P1 (`../dim-reduction-agent`):**
- From `scripts/reduce_dim.py`: `_run_pca`, `_run_umap`, `apply_preprocessing` (pure functions).
- From `scripts/evaluate.py`: trustworthiness and label-silhouette functions.
- From `scripts/cluster.py`: k-means with silhouette scan and density cross-check.
- From `scripts/visualize.py`: scatter and scree plots.
- From `validate_plan.py` / `plan_schema.py`: method roles (visualization-only output cannot feed later steps) and the O(N²) size guard.

**Taken from the P3 design (no code copied now):**
- `setting-sources local` plus absolute permission rules plus deny `cd` (P2 lesson, P3 tests).
- `plan_validation.json` must pass and be newer than `plan.json` before `run_plan.py` runs.
- Every metric has a baseline; outputs keyed `<tool>_s<seed>`; an evidence step before choosing methods (P3's `baseline_check`).
- Embedding cache layout `cache/<d>/<encoder>/{<modality>.npy, ids.csv, manifest.json}`.
- `## Loading` syntax for parquet image/text modalities and `first_code_of` labels (P4 writes its own small parquet reader for this).
- Generic skills checked with dry-run datasets; `summary.json` as P3's output contract.

**Ideas from reference_repos (skimmed):** a `warnings` list every step appends to and the report prints (Sentinelle); a statistics table linking each test to its figure with effect size (ibl); "separate known results from planned checks" and a readiness rating in the critique (biostat-superpowers). None of them checks batch effects or imbalance, so those are new.

## 4. Architecture

```
run.sh <dataset> "<objective>" [--report-number N] [--budget-min M]
  → fetch_data.py, venv setup
  → claude -p  (AGENT.md, pipeline settings, bio-critic)
      1 profiler.py            → profile.json (+ profile_summary.md)
      2 objective.json         → validate_objective.py
      3 challenges.json        → validate_challenges.py      (new)
      4 plan.json              → validate_plan.py            (tool menu, new)
      5 run_plan.py            → tools/<tool>.py per step: request.json in, result.json out
      6 evaluate.py            → metrics/*.json, each with a baseline
      7 bio-critic             → critique.json; at most one plan revision
      8 findings.md            → report.py → reports/<d>/generated_report_<N>/
```

`reports/<d>/generated_report_<N>/` holds: `report.pdf`, `analysis_plan.md` (rendered from plan.json and challenges.json), `methods_summary.md`, `metrics/`, `figures/`. This matches the five required outputs.

### Profiler extensions (all generic, driven by `DATA_DESCRIPTION.md`)
- **Modality and structure:** per modality: kind (counts, continuous, image, text), structured vs unstructured, n samples, n features, sparsity, integer-valued, paired IDs across modalities.
- **Missingness:** NaN fraction per modality, missing labels (e.g. `nan` codes), samples missing a modality.
- **Imbalance:** class counts and imbalance ratio for every declared label and condition; groups below a minimum size.
- **Batch / replicate:** declared `batch.<name>` columns plus auto-detected candidates (low-cardinality obs columns named like replicate/batch/donor/plate/lane/sample/split). For each: fraction of PCA variance explained by batch (R² on top PCs), kNN same-batch fraction vs expected, and Cramér's V between batch and condition (confounding).
- **Experimental design:** declared `design.condition` (with `control=`), or detected control labels (control, NT, non-targeting, ctrl, DMSO). Reports n perturbations, cells per perturbation, control fraction, and a first-pass signal check: perturbation silhouette on control-fitted PCA.
- **QC tails for count data:** fraction of cells beyond 5 MAD in library size and genes detected.

### Challenge identification (new step with its own JSON and validator)
- `scripts/challenge_rules.py`: fixed taxonomy, each with a deterministic trigger on profile.json: `high_dimensional`, `sparse_counts`, `batch_effect`, `batch_condition_confounding`, `class_imbalance`, `small_groups`, `missing_values_or_labels`, `multimodal_paired`, `unstructured_modality`, `weak_condition_signal`, `label_input_leakage` (from DATA_DESCRIPTION caveats), `train_test_shift`, `qc_outliers`.
- `challenges.json`: each entry has id, evidence (profile path + value), severity, and the consequence for analysis. The agent may add or dismiss challenges only with cited evidence.
- `validate_challenges.py`: every triggered rule is listed or dismissed with a reason; cited paths and values match profile.json; no generic reasons.

### Analysis plan
- `scripts/tool_menu.py`: the registry (name, source, input kinds, challenges addressed, preconditions on profile, cost, outputs, metrics, baseline).
- `validate_plan.py`: each step cites challenge ids and evidence; tool preconditions hold; every high-severity challenge is addressed or deferred with a reason; visualization-only outputs never feed later steps; size and budget guards.

## 5. Tool menu

Only tools that address a challenge the profiler can detect in at least one of the three datasets are wired up.

| Tool | Source | Challenge addressed | Evidence the agent uses to choose it | Datasets |
|---|---|---|---|---|
| Preprocess (normalize, log1p, HVG, standardize, drop/impute missing) | P1 `apply_preprocessing` | `sparse_counts`, `high_dimensional`, `missing_values_or_labels` | `is_nonneg_integer`, sparsity, n_features, missing fractions | all |
| PCA | P1 | `high_dimensional`; baseline space for every later tool | n_features ≫ needed; explained variance curve | all |
| UMAP (visualization only) | P1 | show structure; visual check of batch and condition | n samples > ~500; any batch/label/condition to color by | all |
| Batch diagnostic (variance explained by batch, kNN mixing, confounding) | new | `batch_effect`, `batch_condition_confounding`, `train_test_shift` | declared or detected batch/split columns | papalexi (replicate), rocov2 (train/test split) |
| Batch-aware centering (per-batch mean centering in PC space) | new | `batch_effect` when not confounded | diagnostic shows batch R² above threshold and Cramér's V low | papalexi (if triggered) |
| Contrastive PCA (foreground vs background, α sweep) | new (L20; Abid et al. 2018) | `weak_condition_signal`: condition effect hidden by variation shared with controls | design has controls; perturbation silhouette on PCA low; top PCs explain similar variance in controls and perturbed | papalexi |
| Differential expression / abundance (Welch t and Wilcoxon, BH FDR) per condition vs control, or cluster vs rest | new (scipy) | `weak_condition_signal`, design questions; second modality (protein) | design.condition with control, or clusters; one call per modality | papalexi (RNA + protein), pbmc (markers) |
| Perturbation effect sizes (E-distance to controls with permutation test; n DE genes; protein shift) | new | ranking perturbations; `small_groups` (permutation p-values, min group size) | design.condition with control; cells per group | papalexi |
| Clustering (k-means, silhouette scan, density cross-check) + ARI vs labels | P1 | unlabeled structure; checks whether labels match data structure | no labels, or labels to compare against | pbmc, rocov2 |
| Embedding probe (kNN / logistic with stratified split, macro-F1 vs majority baseline) | new (small) | `class_imbalance`, `unstructured_modality`, `label_input_leakage` (image-only vs text-only probe) | declared labels; imbalance ratio sets macro-F1 | rocov2, pbmc |
| Representation learning (P3: frozen encoders, CCA/contrastive alignment, retrieval recall@k) | P3 via `tools/p3_adapter.py` | `multimodal_paired`, `unstructured_modality` | ≥2 paired modalities; image/text kinds need pretrained encoders | rocov2, pbmc |
| VAE outlier score (NB-VAE −ELBO vs kNN-distance baseline) | P2 `models/vae.py` + training loop | `qc_outliers` (possible doublets / low-quality cells) | QC-tail fraction above threshold in count data | papalexi, pbmc (first item on cut list) |

**Not wired up (listed in the report as extensions):** t-SNE, kernel/sparse PCA, MDS, Isomap, LLE, Laplacian eigenmaps, diffusion maps (P1: no challenge in our data that PCA/UMAP doesn't already cover; diffusion maps would fit trajectories, which none of the datasets has); P2 flow, GAN, diffusion model, imputation, TSTR augmentation (no challenge needs synthetic data; scRNA zeros are not missing values); Mixscape-style escaper classification (perturbation heterogeneity is not detectable from the profile alone); learned batch integration (Harmony/scVI).

### P3 as a tool
- `scripts/tools/p3_adapter.py --request req.json --out result.json`. Request: dataset, modalities, labels, task (`retrieval` / `label_transfer`), budget, seeds.
- If `tools/p3/` exists: run P3's own scripts there (extract_embeddings `--if-missing`, its profiler, a P3 `plan.json` built from a fixed template, its validate_plan and run_plan), then read P3's `metrics/summary.json` / `summary.json`. No nested `claude -p` call.
- If not: return `status: "fallback_frozen_cache"` using `cache/<d>/<encoder>/` with simple alignment metrics (retrieval recall@k, probe), or `status: "unavailable"`. The plan and report handle both.
- `scripts/sync_p3.sh <commit>`: requires a commit hash; uses `git -C ../representational-learning-agent archive <commit> scripts bin requirements.txt data/*/DATA_DESCRIPTION.md` (read-only on P3) piped into `tools/p3/`; writes `tools/p3/P3_COMMIT`; excludes docs, CLAUDE.md, outputs, cache, venv. Tests then run against `tools/p3/`, never inside P3. Default pin: P3's HEAD on Fri Nov 27, if its copied tests pass here.

## 6. New skills, agents and scripts

Skills (`.claude/skills/`, generic; a test greps them and AGENT.md for dataset, gene and file names):
- `objective-interpretation` (adapted from P2)
- `challenge-identification` (new: taxonomy, evidence rules, severity)
- `analysis-planning` (new: tool menu, plan.json schema, how to cite evidence)
- `evaluation` (new: baseline per metric, how to judge each tool's output)
- `reporting` (adapted from P2: findings.md rules, traced numbers, strategy → findings → recommendations)

Agent: `.claude/agents/bio-critic.md` (fresh context; checks challenge-plan coverage, baselines, confounding, over-reading UMAP, generic reasons, readiness rating).

New scripts: `challenge_rules.py`, `validate_challenges.py`, `tool_menu.py`, `tools/{batch_diag,batch_center,cpca,de,effect_size,probe,p3_adapter,vae_outlier}.py`, parquet support in `loaders.py`, `evaluate.py` (aggregator), `scripts/sync_p3.sh`.

## 7. Milestones (an order, not gates; commit at each, user pushes)

| # | When (approx.) | Contents |
|---|---|---|
| M0 | Sun Nov 22 | **scPerturb `DATA_DESCRIPTION.md` + `fetch_data.py` first** (md5s from the Zenodo API); start the 148 MB download; scaffold; CLAUDE.md with working rules; copy P2 harness; clone rocov2/pbmc data and the rocov2 embedding cache |
| M1 | Mon Nov 23 | loaders (parquet), extended profiler on all 3 datasets, challenge rules + validator, unit tests |
| M2 | Mon–Tue | tool menu core (preprocess, PCA, UMAP, batch diagnostic, clustering, probe, frozen-cache alignment), plan schema + validator, run_plan, minimal report.py, AGENT.md + skills → **Tue night: rough ROCOv2 report from run.sh** (user runs it in their Tue session) |
| M3 | Wed Nov 25 | CPCA, DE, effect sizes, protein modality; papalexi end to end |
| M4 | Thu Nov 26 | bio-critic + one revision; figure and report quality pass (read rendered pages); pbmc robustness run; VAE outlier if time |
| M5 | Fri Nov 27 | `sync_p3.sh` at pinned P3 commit; P3 adapter live; re-run all three; package generated_report_1/2 |
| M6 | Sat Nov 28 | 4-page report.pdf (LaTeX, agent design focus); 8-minute slides; fresh-clone check |
| — | Sun Nov 29 | user review; final runs |

Working rules for CLAUDE.md: dates are an ordering, not gates; keep going between milestones; stop only when a decision changes objectives or scope, something cannot be undone, the user is needed (Longleaf, credentials), or tests still fail after one bounded fix; commit at milestones, never push; never edit or run anything inside P1/P2/P3/reference_repos; all Python through `venv/bin/python`; laptop on AC power with the lid open during long runs.

## 8. Cut list (cut from the top)

1. VAE outlier scoring (keep MAD QC flags in the profile)
2. Batch-aware centering (report the diagnostic only)
3. Critic-driven plan revision (keep the critique, drop the second pass)
4. P3 trained models (use P3 frozen embeddings only)
5. Full pbmc run (keep profile + challenges as a robustness check)
6. Slide polish (plain 8-slide deck)

Never cut: profiler challenge detection, challenges.json + validator, plan validator linking challenges to tools, a baseline for every metric, both generated reports, report.pdf, slides, the docs/ read deny.

## 9. Verification

- `venv/bin/python -m pytest tests/`: settings isolation, generic-skill grep, validators reject bad challenges/plans (missing evidence, uncovered high-severity challenge, "default" reasons), profiler on small synthetic fixtures (planted batch, control group, missing labels, imbalance).
- Profiler on all three real datasets: check expected challenges fire (papalexi: batch, design, weak signal; rocov2: unstructured, multimodal, imbalance, missing labels, leakage; pbmc: multimodal, imbalance, no batch).
- `./run.sh rocov2 --smoke` (zero permission denials), then full runs; read figures and rendered PDF pages, not only exit codes.
- Fresh clone of the repo + `run.sh` on one dataset before submission.

## 10. Open items (not blocking)
- L20 slides are not in `BIOS774/lectures/`. CPCA will follow Abid et al. 2018 unless the slides are added to `resources/`.
- P3 needs to produce `summary.json` (planned in P3) for the adapter's live path; until then the adapter uses the frozen cache.
