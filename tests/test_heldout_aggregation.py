import json

import numpy as np

from scripts.heldout.aggregate_heldout import _holm, aggregate, load_records


def _write_record(path, *, subject, hashes=None):
    methods = ("amica", "picard", "infomax", "fastica")
    offsets = {"amica": 0.5, "picard": 0.2, "infomax": 0.1, "fastica": 0.0}
    folds = []
    for fold, n_test in enumerate((100, 300)):
        entries = []
        for method in methods:
            sha = (hashes or {}).get((fold, method), f"fold-{fold}")
            entries.append(
                {
                    "method": method,
                    "heldout_mir": {
                        "100": {
                            "kbits_per_sec": offsets[method] + fold,
                            "sample_index_sha256": sha,
                        }
                    },
                }
            )
        folds.append({"fold": fold, "n_test": n_test, "methods": entries})
    payload = {
        "analysis": "post-hoc guard-banded held-out complete MIR",
        "status": "complete",
        "dataset": "ds004505",
        "subject": subject,
        "folds": folds,
    }
    path.write_text(json.dumps(payload))


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    raw = [0.03, 0.001, 0.02, 0.5]
    adjusted = np.asarray(_holm(raw))
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all(adjusted >= raw)
    assert np.all(adjusted <= 1)


def test_fold_size_weighting_and_subject_contrasts(tmp_path):
    _write_record(tmp_path / "sub-1.json", subject=1)
    frame = load_records(tmp_path, allow_incomplete=True)
    subjects, contrasts = aggregate(frame, n_bootstrap=100, seed=7)
    amica = subjects.query("method == 'amica'").iloc[0]
    assert np.isclose(amica.heldout_mir_kbits_s, (0.5 * 100 + 1.5 * 300) / 400)
    picard = contrasts.query("comparator == 'picard'").iloc[0]
    assert np.isclose(picard.mean_delta_mir_kbits_s, 0.3)
    assert picard.positive_n == 1


def test_method_sample_hash_mismatch_is_rejected(tmp_path):
    _write_record(
        tmp_path / "sub-1.json",
        subject=1,
        hashes={(0, "fastica"): "different"},
    )
    try:
        load_records(tmp_path, allow_incomplete=True)
    except RuntimeError as exc:
        assert "identical held-out sample indices" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("sample mismatch was not rejected")
