"""Where the figure and table producers read data from and write assets to.

These scripts were written inside the Overleaf clone, where the validation
workspace happened to sit three or four levels up, so each one anchored itself
with ``Path(__file__).parents[3]``. That anchor is meaningless here, and it was
never checked -- a wrong parent count silently resolves to some other directory
and the script fails later with a confusing "no such file".

Two roots, both overridable, so the same script runs from a bare clone, from
the validation workspace, or against an extracted data archive:

``DATA_ROOT``
    Where the benchmark result trees live. Defaults to this repository, which
    carries the small aggregated inputs. Point ``AMICA_BENCH_DATA`` at the
    validation workspace or an extracted data archive to reach the bulk trees
    (the multi-model ``.npz`` fits are ~485 MB and are not in git).

``OUT_ROOT``
    Where emitted ``.tex`` and figure files land. Defaults to
    ``results/paper_assets/`` here rather than the Overleaf project root, so
    running a producer never writes outside the repository by surprise. Set
    ``AMICA_BENCH_TEX_OUT`` to the Overleaf clone to regenerate the manuscript
    assets in place.
"""

from __future__ import annotations

import os
from pathlib import Path

# scripts/paper/figures/_paths.py -> figures -> paper -> scripts -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = Path(os.environ.get("AMICA_BENCH_DATA") or REPO_ROOT).resolve()

OUT_ROOT = Path(
    os.environ.get("AMICA_BENCH_TEX_OUT") or REPO_ROOT / "results" / "paper_assets"
).resolve()


def data(*parts: str) -> Path:
    """Resolve a workspace-relative data path against ``DATA_ROOT``."""
    return DATA_ROOT.joinpath(*parts)


def out(name: str) -> Path:
    """Resolve an emitted asset name against ``OUT_ROOT``, creating the dir."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUT_ROOT / name
