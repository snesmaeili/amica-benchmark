from scripts.validation.provenance import validate_provenance


def test_provenance_schema_reports_missing_fields():
    assert "command" in validate_provenance({"schema_version": 1})


def test_minimal_provenance_schema_is_valid():
    payload = {
        "schema_version": 1,
        "captured_utc": "2026-07-16T00:00:00+00:00",
        "command": ["python", "job.py"],
        "python": {},
        "platform": {},
        "environment": {},
        "slurm": {},
        "packages": {},
        "repositories": [],
    }
    assert validate_provenance(payload) == []
