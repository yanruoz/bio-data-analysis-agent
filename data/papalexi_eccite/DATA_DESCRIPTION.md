# DATA_DESCRIPTION: papalexi_eccite

## What this is
A pooled CRISPR knockout screen in THP-1 cells (a human monocytic leukemia cell
line) stimulated with IFN-gamma, profiled with ECCITE-seq (Papalexi et al.,
Nature Genetics 2021). Every cell has two modalities measured in the same cell:
gene expression (RNA) and four surface proteins (antibody-derived tags, ADT).
Each cell also carries one guide RNA that names the gene knocked out in that
cell, or a non-targeting guide (control). The targeted genes include
regulators of the IFN-gamma response and of the immune checkpoint protein
PD-L1. The files are the harmonized copies distributed by scPerturb (Peidli et
al., Nature Methods 2024), Zenodo record 13350497 (scPerturb v1.4), one
AnnData file per modality.

## Files
- file: PapalexiSatija2021_eccite_RNA.h5ad | url: https://zenodo.org/records/13350497/files/PapalexiSatija2021_eccite_RNA.h5ad?download=1 | md5: d9e8bfe20b0e7b0919be042e8fcf6d03
- file: PapalexiSatija2021_eccite_protein.h5ad | url: https://zenodo.org/records/13350497/files/PapalexiSatija2021_eccite_protein.h5ad?download=1 | md5: 07290242b0e835c9474bb816de9cda45

## Loading
- modality.rna: file=PapalexiSatija2021_eccite_RNA.h5ad | kind=counts
- modality.protein: file=PapalexiSatija2021_eccite_protein.h5ad | kind=counts
- design.condition: column=perturbation | extract=^(.+?)(?:g[0-9]+)?$ | control=control
- batch.replicate: column=hto
- qc.rna: columns=ncounts, ngenes, percent_mito

## Structure
- 20,729 cells, the same cells in the same order in both files (`obs_names`
  match across files).
- RNA: 20,729 cells x 18,649 genes, raw UMI counts (integer-valued, stored as
  float32), about 82% zeros. Median 11,815 UMIs and 3,292 detected genes per
  cell.
- Protein: 20,729 cells x 4 surface proteins (CD86, PDL1, PDL2, CD366), raw
  ADT counts, almost no zeros.
- About 150 MB in total.

## Experimental design
- `obs['perturbation']` names the guide (for example `STAT1g2`) or `control`
  for non-targeting guides. Removing the trailing guide number (`g1` to `g4`)
  gives the targeted gene (see `## Loading`). `obs['guide_id']` keeps the
  individual non-targeting guide names (`NTg1` ...).
- 25 targeted genes, 3 or 4 guides each, plus 2,386 control cells (11.5%).
  `obs['nperts']` is 0 for control cells and 1 for perturbed cells.
- Cells per targeted gene range from 1,341 (IFNGR1) down to 109 (MYC) and 48
  (SPI1).

## Replicates
`obs['hto']` (cell hashtag) records the replicate: rep1-tx 9,351 cells,
rep3-tx 5,992, rep4-tx 5,309, rep2-tx 73 and rep2-ctrl 4. The targeted genes
are spread over rep1, rep3 and rep4 in similar proportions, so replicate is not
confounded with perturbation. rep2 is too small to analyze on its own.

## Known caveats
- Cells were already filtered by the data providers: every cell has at least
  about 3,000 UMIs and under 10% mitochondrial reads, so a low-quality-cell
  filter will remove little.
- In a pooled CRISPR screen, a cell carrying a guide is not always a cell with
  a functional knockout. Many cells with a targeting guide may look like
  controls, which dilutes the average effect of a perturbation.
- Expression differences between replicates and across the cell cycle can be
  larger than the effect of many perturbations.
- `obs` columns `ncounts`, `ngenes`, `percent_mito` and `percent_ribo` are
  computed per file: in the protein file they describe ADT counts, not RNA.
- Metadata columns `tissue_type`, `cell_line`, `cancer`, `disease`,
  `perturbation_type`, `celltype` and `organism` are constant and carry no
  information.
- Only four proteins were measured, so the protein modality is a targeted
  readout, not a genome-wide profile.
- License: scPerturb data are distributed under CC BY 4.0.
