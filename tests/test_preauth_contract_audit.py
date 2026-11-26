"""Tests for the pre-authorisation static contract audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_preauth_contract.py"
spec = importlib.util.spec_from_file_location("audit_preauth_contract", MODULE_PATH)
assert spec is not None and spec.loader is not None
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def test_finds_literal_subscript_and_get() -> None:
    source = """def score(txn):
    return txn["newbalanceOrig"] + txn.get("isFraud", 0)
"""
    findings = audit_module.inspect_source(source, "example.py")
    assert [(f["line"], f["field"]) for f in findings] == [
        (2, "isFraud"), (2, "newbalanceOrig")
    ]


def test_ignores_safe_fields_and_comments() -> None:
    source = """# txn["newbalanceDest"] is forbidden
def score(txn):
    return txn["oldbalanceOrg"] + txn.get("amount", 0)
"""
    assert audit_module.inspect_source(source, "safe.py") == []


def test_audit_reports_existing_v1_leakage() -> None:
    report = audit_module.audit()
    assert report["contract"] == "pre-authorisation"
    assert report["status"] == "fail"
    assert any(f["field"] == "newbalanceOrig" for f in report["findings"])
