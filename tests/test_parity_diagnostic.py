import numpy as np
import pytest

from scripts.parity.diagnose_manifest_cell import (
    parse_checkpoints,
    trajectory_metrics,
)


def test_trajectory_metrics_preserves_signed_differences_and_lengths():
    metrics = trajectory_metrics(
        np.array([-2.0, -1.5, -1.0]),
        np.array([-2.0, -1.4]),
    )

    assert metrics["reference_n_values"] == 3
    assert metrics["candidate_n_values"] == 2
    assert metrics["n_common"] == 2
    assert metrics["candidate_minus_reference"] == pytest.approx([0.0, 0.1])
    assert metrics["initial_absolute_difference"] == 0.0
    assert metrics["final_absolute_difference"] == pytest.approx(0.1)
    assert metrics["maximum_absolute_difference"] == pytest.approx(0.1)
    assert metrics["first_nonfinite_index"] is None


def test_trajectory_metrics_reports_nonfinite_values():
    metrics = trajectory_metrics(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    assert metrics["first_nonfinite_index"] == 1
    assert metrics["maximum_absolute_difference"] == 0.0


def test_parse_checkpoints_deduplicates_and_sorts():
    assert parse_checkpoints("50,1,3,3") == (1, 3, 50)


def test_parse_checkpoints_rejects_nonpositive_values():
    with pytest.raises(Exception):
        parse_checkpoints("0,1")
