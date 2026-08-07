from argparse import Namespace
from pathlib import Path

from scripts.archive.build_validation_bundle import build


def test_preview_bundle_records_missing_evidence(tmp_path):
    benchmark = tmp_path / "benchmark"
    manuscript = tmp_path / "manuscript"
    evidence = tmp_path / "evidence"
    output = tmp_path / "output"
    for path in (benchmark, manuscript, evidence, output):
        path.mkdir()
    (benchmark / "pyproject.toml").write_text("[project]\nname='test'\n")
    (manuscript / "zenodo.tex").write_text("test")
    build(
        Namespace(
            version="test",
            benchmark_root=benchmark,
            manuscript_root=manuscript,
            evidence_root=evidence,
            output_dir=output,
            require_complete=False,
        )
    )
    bundle = output / "amica-validation-test"
    assert (bundle / "ARTIFACT_MANIFEST.json").exists()
    assert (output / "amica-validation-test.zip").exists()

