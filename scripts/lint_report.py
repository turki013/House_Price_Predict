#!/usr/bin/env python
"""Lint a README/report for hype wording and numbers that no real run produced.

Usage:
    python scripts/lint_report.py README.md --results reports/results.json [--ignore 5 42] [--strict]

Exit code 1 when hype words or unverified metric-like numbers are found (warnings only with --strict).
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dstk.report_lint import lint_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--results", help="JSON file written by the evaluation script")
    ap.add_argument("--ignore", nargs="*", type=float, default=[], help="numbers to skip (folds, seeds)")
    ap.add_argument("--strict", action="store_true", help="treat claim-word warnings as failures")
    a = ap.parse_args()
    r = lint_file(a.report, a.results, a.ignore)
    for h in r["hype"]:
        print(f"ERROR   line {h['line']}: hype word '{h['word']}' -> {h['text']}")
    for h in r["claims"]:
        print(f"WARNING line {h['line']}: '{h['word']}' needs a test or interval behind it -> {h['text']}")
    for h in r["unverified_numbers"]:
        print(f"ERROR   line {h['line']}: number {h['number']} not found in results -> {h['text']}")
    failed = bool(r["hype"] or r["unverified_numbers"] or (a.strict and r["claims"]))
    print("FAILED" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
