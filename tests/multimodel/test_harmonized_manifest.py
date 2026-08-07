from __future__ import annotations

import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "multimodel"
sys.path.insert(0, str(MODULE_DIR))

from make_harmonized_manifest import SUBJECTS, rows  # noqa: E402


def test_dataset_subject_identifiers_match_loader_conventions():
    assert SUBJECTS["ds004505"] == tuple(range(1, 26))
    assert SUBJECTS["ds004504"] == tuple(range(37, 66))
    assert SUBJECTS["ds004621"] == tuple(range(1, 43))


def test_smoke_manifest_has_four_rows_per_dataset():
    smoke = list(rows(smoke=True, pilot=False, extra_surrogates=False))
    assert len(smoke) == 12
    ds4504 = [row for row in smoke if row["dataset"] == "ds004504"]
    assert len(ds4504) == 4
    assert {row["subject"] for row in ds4504} == {37}
