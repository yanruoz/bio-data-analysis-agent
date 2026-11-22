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
