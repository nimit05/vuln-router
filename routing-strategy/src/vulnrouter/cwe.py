"""The CWE taxonomy: the graph S11 propagates over.

GraphRouter (Feng, Shen & You, ICLR 2025) names taxonomy as future work -- their
example is a model family tree, LLaMA2 -> LLaMA3 -> LLaMA3.1. CWE ships a
human-curated one, so instantiating it is the cheapest novel piece of S11
(docs/04-graph-router.md).

Why it earns its place, concretely: PrimeVul's CWE distribution has a long tail
of classes with one or two functions. A per-CWE estimate at n=1 is noise. An
estimate that falls back on CWE-119 when CWE-121 is nearly empty is a shrinkage
estimator over a hierarchy someone else already validated.

The built-in map is a SEED, not the full hierarchy. It holds ChildOf edges from
the CWE-1000 research view for classes common in C/C++ corpora, each checkable
by hand at cwe.mitre.org. For a real run load the official export
(`load_parents_csv`): 900+ classes, no hand-transcription risk.
"""
from __future__ import annotations
import csv
import re

# ChildOf edges, CWE-1000 (research) view. Deliberately short: a wrong edge is
# worse than a missing one, because it pools two unrelated classes and the
# pooling is invisible in the final number.
PARENTS: dict[str, str] = {
    "CWE-125": "CWE-119",   # out-of-bounds read
    "CWE-787": "CWE-119",   # out-of-bounds write
    "CWE-786": "CWE-119",   # access before start of buffer
    "CWE-788": "CWE-119",   # access after end of buffer
    "CWE-121": "CWE-787",   # stack-based buffer overflow
    "CWE-122": "CWE-787",   # heap-based buffer overflow
    "CWE-119": "CWE-118",   # improper restriction of ops within a buffer
    "CWE-416": "CWE-672",   # use after free
    "CWE-401": "CWE-772",   # missing release of memory
    "CWE-772": "CWE-404",   # missing release of resource
    "CWE-404": "CWE-664",   # improper resource shutdown or release
    "CWE-672": "CWE-664",   # operation on a resource after expiry or release
    "CWE-190": "CWE-682",   # integer overflow or wraparound
    "CWE-191": "CWE-682",   # integer underflow
    "CWE-78":  "CWE-74",    # OS command injection
    "CWE-79":  "CWE-74",    # cross-site scripting
    "CWE-89":  "CWE-74",    # SQL injection
    "CWE-74":  "CWE-707",   # improper neutralisation
    "CWE-20":  "CWE-707",   # improper input validation
    "CWE-400": "CWE-664",   # uncontrolled resource consumption
}

ROOT = "CWE-ROOT"           # single sink, so every class is connected
UNKNOWN = "CWE-UNKNOWN"     # the static tier could not classify this unit


def normalise(cwe: str | None) -> str:
    """'119', 'cwe_119', 'CWE-119: ...' -> 'CWE-119'. None/junk -> CWE-UNKNOWN."""
    if cwe is None:
        return UNKNOWN
    m = re.search(r"(\d+)", str(cwe))
    return f"CWE-{m.group(1)}" if m else UNKNOWN


def load_parents_csv(path: str, id_col: str = "CWE-ID",
                     rel_col: str = "Related Weaknesses") -> dict[str, str]:
    """Parse MITRE's CSV export.

    Related Weaknesses cells look like
    '::NATURE:ChildOf:CWE ID:119:VIEW ID:1000:ORDINAL:Primary::'.
    Only ChildOf is kept: PeerOf and CanPrecede are associations, not
    generalisations, and pooling across them is not shrinkage, it is a bug.
    """
    parents: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            child = normalise(row.get(id_col))
            for chunk in (row.get(rel_col) or "").split("::"):
                if "ChildOf" in chunk and "CWE ID:" in chunk:
                    f = chunk.split(":")
                    parents.setdefault(child, normalise(f[f.index("CWE ID") + 1]))
                    break
    return parents


def ancestors(cwe: str, parents: dict[str, str] | None = None,
              max_depth: int = 12) -> list[str]:
    """[cwe, parent, ..., ROOT]. Cycle-safe and depth-capped."""
    parents = PARENTS if parents is None else parents
    chain, seen, node = [], set(), normalise(cwe)
    while node not in seen and len(chain) < max_depth:
        chain.append(node)
        seen.add(node)
        nxt = parents.get(node)
        if nxt is None:
            break
        node = nxt
    chain.append(ROOT)
    return chain
