import json

from scripts.validation.audit_historical_environment import audit


def test_historical_audit_flags_unrecorded_fields(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"hardware": {"hostname": "node1"}}))
    result = audit(tmp_path)
    assert result["fields"]["hostname"]["available"]
    assert "cpu_model" in result["unresolved_required_fields"]

