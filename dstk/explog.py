"""Append-only experiment log (JSON Lines) with a Markdown renderer."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

FIELDS = ["id", "date", "hypothesis", "change", "rows", "metric", "value", "ci_lo", "ci_hi",
          "baseline_value", "paired_verdict", "decision", "notes"]


def log_experiment(path: str | Path, **fields: Any) -> Dict[str, Any]:
    """Append one record. Required: id, hypothesis, change, rows, metric, value, decision.
    `rows` must say which rows were evaluated (e.g. 'test n=1448 after dedup') so comparisons stay honest."""
    required = ["id", "hypothesis", "change", "rows", "metric", "value", "decision"]
    missing = [k for k in required if k not in fields]
    if missing:
        raise ValueError(f"log_experiment: missing fields {missing}")
    record = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), **fields}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_log(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def render_markdown(records: List[Dict[str, Any]]) -> str:
    head = "| id | hypothesis | change | rows | metric | value [95% CI] | baseline | paired verdict | decision |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in records:
        ci = f" [{_fmt(r.get('ci_lo'))}, {_fmt(r.get('ci_hi'))}]" if r.get("ci_lo") is not None else " [no CI]"
        lines.append("| " + " | ".join([_fmt(r.get("id")), _fmt(r.get("hypothesis")), _fmt(r.get("change")),
                                        _fmt(r.get("rows")), _fmt(r.get("metric")),
                                        _fmt(r.get("value")) + ci, _fmt(r.get("baseline_value")),
                                        _fmt(r.get("paired_verdict")), _fmt(r.get("decision"))]) + " |")
    return "\n".join(lines)
