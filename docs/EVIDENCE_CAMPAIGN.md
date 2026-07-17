# Evidence campaign for the AMICA manuscript

This repository separates validation code and result records from the installable
`amica` package. Real-data fitting must run through Slurm; the scripts must not be
executed as heavy jobs on a login node.

## Fortran parity

Generate the prespecified 18-cell manifest:

```bash
python scripts/parity/parity_manifest.py --output manifests/parity_cells.csv
```

Each array task runs one cell and writes one JSON record. The aggregate command
requires all 18 records and exits non-zero when any prespecified check fails:

```bash
python scripts/parity/aggregate_manifest.py \
  --input-dir results/parity/cells --output-dir results/parity
```

The campaign covers twelve single-model combinations of density-term count,
optimisation path, and sample rejection, plus six multi-model fixtures. Passing
cells support claims only for those explicit configurations.

## Held-out MIR

Each subject uses five contiguous test blocks. Five-second guard regions around
the test block are removed from training. Filtering and resampling are fixed
recording-level operations; centring, whitening, PCA, ICA, and density parameters
are learned from training samples only. All methods use the same explicit test
samples at 50, 100, and 200 histogram bins.

After all 96 subject records complete:

```bash
python scripts/heldout/aggregate_heldout.py \
  --input-dir results/heldout/subjects --output-dir results/heldout
```

The command refuses incomplete cohorts and verifies that method-level sample
hashes agree within every fold.

## Provenance and archive

Every job embeds a provenance record. A standalone node record can be generated
with `scripts/validation/provenance.py`. Build a preview archive without
`--require-complete`; use that flag for the deposit candidate after all evidence
files exist. The archive deliberately excludes raw recordings, fitted FIF files,
credentials, and large full posterior arrays.
