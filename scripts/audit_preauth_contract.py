"""Static guardrail for post-authorisation fields in the live scoring path.

Run: python scripts/audit_preauth_contract.py
Exit 0 produces a JSON audit report; --strict exits 1 when violations exist.
The audit intentionally reports known v1 defects until migration is complete.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

FORBIDDEN = frozenset({"newbalanceOrig", "newbalanceDest", "isFraud", "isFlaggedFraud"})
TARGETS = ("features.py", "rules.py", "scoring.py")
ROOT = Path(__file__).resolve().parents[1]


def inspect_source(source: str, filename: str) -> list[dict[str, object]]:
    """Locate literal references to unavailable fields; no imports or execution."""
    tree = ast.parse(source, filename=filename)
    findings: list[dict[str, object]] = []
    for node in ast.walk(tree):
        # Subscript catches txn["newbalanceOrig"]; get() catches txn.get("isFraud").
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            key = node.slice.value
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            key = node.args[0].value
        else:
            continue
        if isinstance(key, str) and key in FORBIDDEN:
            findings.append({"file": filename, "line": node.lineno, "field": key})
    return sorted(findings, key=lambda item: (item["line"], item["field"]))


def audit(root: Path = ROOT) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    for filename in TARGETS:
        path = root / "src" / "fraud_engine" / filename
        findings.extend(inspect_source(path.read_text(encoding="utf-8"), str(path.relative_to(root))))
    return {
        "contract": "pre-authorisation",
        "status": "fail" if findings else "pass",
        "findings": findings,
        "note": "Static literal-key audit only; not proof of point-in-time correctness.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail if forbidden fields are used")
    args = parser.parse_args()
    report = audit()
    print(json.dumps(report, indent=2))
    return int(args.strict and report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
