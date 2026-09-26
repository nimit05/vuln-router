"""Benchmark loader. PrimeVul by default; the interface is the commitment, the
benchmark is not (docs/01-problem.md §1.2).

A benchmark must yield units with: unit_id, code, gold_label, commit (the split
group), and optionally cwe and pair_id. `pair_id` is what makes PrimeVul worth
preferring -- it links a vulnerable function to its patch, and pair accuracy is
the metric that separates models reasoning about the defect from models
reacting to surface form.
"""
from __future__ import annotations
import json
from pathlib import Path


def load_primevul(path: str | Path, paired: bool = False,
                  limit: int | None = None) -> list[dict]:
    """Read a primevul_*.jsonl into the unit schema.

    Splits are PrimeVul's own, which are group-aware by construction -- do NOT
    re-split randomly (docs/03-protocol.md §4). Field names differ slightly
    between the original release and v0.1, so both spellings are accepted.
    """
    units = []
    for i, line in enumerate(open(path)):
        if not line.strip():
            continue
        r = json.loads(line)
        code = r.get("func") or r.get("function") or r.get("code")
        if code is None:
            raise ValueError(f"no code field in record {i}: {sorted(r)[:12]}")
        uid = str(r.get("idx", r.get("id", i)))
        units.append(dict(
            unit_id=uid,
            code=code,
            gold_label=int(r.get("target", r.get("label", 0))),
            commit=str(r.get("commit_id", r.get("commit", uid))),
            cwe=(r.get("cwe") or [None])[0] if isinstance(r.get("cwe"), list)
                else r.get("cwe"),
            project=r.get("project"),
            pair_id=r.get("hash") or r.get("func_hash") if paired else None,
        ))
        if limit and len(units) >= limit:
            break
    return units


def units_to_sources(units: list[dict]) -> dict[str, str]:
    """{unit_id -> code}, the input to features.frame()."""
    return {u["unit_id"]: u["code"] for u in units}
