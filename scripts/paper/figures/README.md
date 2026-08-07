# Manuscript figure and table producers

Every figure and table in the preprint is generated here. These scripts used to
live in the Overleaf project, which meant the manuscript's own figure code was
not in any public repository; they were moved here so one clone answers both
"how was this benchmarked" and "how was this figure made".

## Running them

```bash
python make_tab_correctness.py --write     # writes results/paper_assets/tab_correctness.tex
python make_main_figures.py                # writes the eight main/supplementary figures
```

`make_tab_correctness`, `make_tab_datasets_configs` and
`make_tab_multimodel_summary` print to stdout unless given `--write`. The others
always write.

Two environment variables control the paths, and both have working defaults:

| Variable | Default | Use |
|---|---|---|
| `AMICA_BENCH_DATA` | this repository | where the benchmark result trees are read from |
| `AMICA_BENCH_TEX_OUT` | `results/paper_assets/` | where `.tex` and figures are written |

Output defaults inside the repository rather than to the Overleaf project, so
running a producer never writes outside the clone by surprise. To regenerate the
manuscript assets in place, point `AMICA_BENCH_TEX_OUT` at the Overleaf clone.

## What runs from a clone, and what does not

Five of the eight tables regenerate byte-identically from a bare clone. Three
need result trees too large for git and will **refuse to run**, naming the
missing input, rather than emit a short table:

| Producer | Needs | Size |
|---|---|---|
| `make_tab_datasets_configs` | `results/backend_parity_v3/` | 81 MB |
| `make_tab_multimodel_summary` | `figdata/mmbench_*/` fits | 485 MB |
| `make_tab_synthetic_recovery` | `figdata/synth/amica_python_synthetic_v1*/` | — |

`make_main_figures.py` likewise reads the full result set. For these, set
`AMICA_BENCH_DATA` to the validation workspace or an extracted data archive.

The refusals are deliberate. `make_tab_multimodel_seeds` previously exited 0
while printing "8 of 9 seeds" instead of "9 of 10" when one input was absent,
which reads as a different result rather than a broken run. Every producer that
can silently drop a row now checks its inputs first.

## `main_figure_stats.json` is an input, not an output

`make_tab_correctness` calls it "the frozen-numbers artifact the manuscript was
written from", and three producers read it. `make_main_figures.py` recomputes
the same statistics but does **not** reproduce the file key-for-key — the
regenerated version drops `backend_worst_row_agreement`, among others. It
therefore writes its version alongside the figures (`AMICA_BENCH_TEX_OUT`),
leaving the frozen copy here untouched. Compare the two deliberately; do not
overwrite one with the other.
