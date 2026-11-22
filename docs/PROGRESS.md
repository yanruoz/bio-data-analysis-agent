# Progress log

## Sun Nov 22 (M0)
- scPerturb dataset chosen: PapalexiSatija2021_eccite (RNA + 4 surface proteins).
  Downloaded from Zenodo 13350497 and md5-verified with `scripts/fetch_data.py`.
  Facts checked on the real files: 20,729 cells; 18,649 genes; 25 targeted genes
  (3-4 guides each) + 2,386 non-targeting controls; replicates rep1/rep3/rep4
  (rep2 has 77 cells); replicate not confounded with perturbation.
- rocov2 and pbmc_multiome data and the rocov2 frozen-embedding cache cloned
  (`cp -c`) from P3; their DATA_DESCRIPTION.md files copied unchanged.
- P2 harness copied: run.sh, bin/, settings, tests/test_settings.py (+ P4 denies
  for docs/, resources/, README.md, CLAUDE.md, scripts/, cache/, tools/).
- fetch_data.py rewritten (stdlib only, multi-file `## Files`, md5 or sha256)
  because P2's handled a single file; P3's is not copied before the sync.

## Sun Nov 22 (M1)
- `scripts/loaders.py`: new multi-modality loader (h5ad per modality, parquet
  image/text), `## Loading` keys modality/label/design/batch/qc/id_column/split.
  `first_code_of` = first code in the list that the map names (P3 semantics).
- `scripts/profiler.py`: modality summaries, groupings (labels, condition,
  batches, split, auto-detected batch-like columns), structure per modality
  (R² by group, kNN enrichment, silhouette), condition shift vs a control null,
  batch-condition Cramér's V, modality kNN agreement, caveat flags. ~5 s each.
- `scripts/challenge_rules.py` (17-id taxonomy, fixed thresholds) and
  `scripts/validate_challenges.py`; 9 synthetic-data tests.
- Triggers on real data:
  - papalexi_eccite: high_dimensional, sparse_counts, batch_effect (high:
    replicate R² 0.05 vs condition 0.07), weak_condition_signal (median shift
    0.12 of control spread), targeted_panel (4 proteins), class_imbalance,
    multimodal_paired, low_modality_agreement. Not confounded (V = 0.07).
  - rocov2: unstructured_modality, heterogeneous_inputs, label_input_leakage
    (45% of captions name their own modality label), class_imbalance,
    missing_values_or_labels (high: body region missing for non-X-ray),
    small_groups, multimodal_paired, pretraining_overlap. No train/test shift.
  - pbmc_multiome: high_dimensional, sparse_counts, class_imbalance,
    small_groups, multimodal_paired, low_modality_agreement, label_input_leakage.
- Decision (applies the user's menu rule): VAE outlier tool removed from the
  menu, listed as an extension, because qc_outliers fires on no dataset.
