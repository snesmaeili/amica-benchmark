from pathlib import Path

import numpy as np

from scripts.parity.adapters.fortran_adapter import FortranAdapter
from scripts.parity.metrics import match_models, rejection_metrics
from scripts.parity.parity_manifest import manifest_rows


def test_manifest_has_prespecified_18_unique_cells():
    rows = list(manifest_rows())
    assert len(rows) == 18
    assert len({row["cell_id"] for row in rows}) == 18
    assert sum(row["num_models"] == 1 for row in rows) == 12
    assert sum(row["num_models"] > 1 for row in rows) == 6


def test_model_matching_handles_model_permutation_and_row_signs():
    rng = np.random.default_rng(4)
    reference = np.stack([rng.normal(size=(4, 4)) for _ in range(3)])
    candidate = reference[[2, 0, 1]].copy()
    candidate[:, [1, 3]] *= -1
    assignment, correlations = match_models(reference, candidate)
    np.testing.assert_array_equal(assignment, [1, 2, 0])
    np.testing.assert_allclose(correlations, 1.0)


def test_rejection_metrics_compare_rejected_samples():
    reference = np.array([True, True, False, False, True])
    candidate = np.array([True, True, False, True, True])
    metrics = rejection_metrics(reference, candidate)
    assert metrics["reference_rejected"] == 2
    assert metrics["candidate_rejected"] == 1
    assert metrics["rejected_count_difference"] == 1
    assert metrics["rejection_jaccard"] == 0.5


def test_fortran_param_file_exposes_cell_configuration(tmp_path):
    params = {
        "num_models": 3,
        "num_mix": 5,
        "lrate": 0.01,
        "min_dll": 1e-9,
        "do_reject": True,
        "rejsig": 3.0,
        "rejstart": 2,
        "rejint": 3,
        "numrej": 5,
        "do_newton": False,
        "newt_start": 50,
        "rholrate": 0.05,
        "rho0": 1.5,
        "minrho": 1.0,
        "maxrho": 2.0,
        "newt_ramp": 10,
        "newtrate": 1.0,
        "max_decs": 3,
        "invsigmax": 100.0,
        "invsigmin": 1e-8,
        "doscaling": True,
    }
    output = tmp_path / "run.param"
    FortranAdapter._write_param_file(
        output,
        Path("data.fdt"),
        Path("out"),
        6,
        1000,
        6,
        params,
        200,
    )
    text = output.read_text()
    assert "num_models 3" in text
    assert "num_mix_comps 5" in text
    assert "do_reject 1" in text
    assert "do_newton 0" in text
    assert "write_LLt 1" in text
