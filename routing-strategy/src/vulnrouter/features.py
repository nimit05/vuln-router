"""Pre-call features on a code unit.

Constraint (docs/02-strategies.md, Family C): the routing decision must cost
microseconds against seconds of inference, so nothing here calls a model. These
are read straight from the source text.

Returns a unit_id-indexed DataFrame, which is the only form evaluate.align()
accepts -- positional arrays silently misalign against the measurement table.
"""
from __future__ import annotations
import re
import pandas as pd

_IF     = re.compile(r"\bif\s*\(")
_LOOP   = re.compile(r"\b(for|while)\s*\(")
_ELSE   = re.compile(r"\belse\b")
_CASE   = re.compile(r"\bcase\b")
_GOTO   = re.compile(r"\bgoto\b")
_TERN   = re.compile(r"\?[^;:]{1,80}:")
_LOGIC  = re.compile(r"&&|\|\|")
_DEREF  = re.compile(r"(?<![\w)])\*\s*\w")
_ARROW  = re.compile(r"->")
_INDEX  = re.compile(r"\w\s*\[")
_ALLOC  = re.compile(r"\b(malloc|calloc|realloc|alloca|new)\s*\(")
_FREE   = re.compile(r"\b(free|delete)\s*[\(\[]")
_UNSAFE = re.compile(r"\b(strcpy|strcat|sprintf|gets|memcpy|memmove|scanf|"
                     r"system|exec[lv]p?|popen)\s*\(")
_CAST   = re.compile(r"\(\s*(unsigned|signed|char|short|int|long|size_t|void)\b[^)]*\)")
_CMP    = re.compile(r"[<>]=?|==|!=")
_CALL   = re.compile(r"\b[A-Za-z_]\w*\s*\(")


def extract(code: str) -> dict[str, float]:
    lines = [l for l in code.splitlines() if l.strip()]
    depth, max_depth = 0, 0
    for ch in code:
        if ch == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == "}":
            depth = max(0, depth - 1)
    n_if, n_loop = len(_IF.findall(code)), len(_LOOP.findall(code))
    n_alloc, n_free = len(_ALLOC.findall(code)), len(_FREE.findall(code))
    return {
        "loc": len(lines),
        "chars": len(code),
        "max_nest": max_depth,
        "n_if": n_if,
        "n_loop": n_loop,
        "n_else": len(_ELSE.findall(code)),
        "n_case": len(_CASE.findall(code)),
        "n_goto": len(_GOTO.findall(code)),
        "n_ternary": len(_TERN.findall(code)),
        "n_logic_ops": len(_LOGIC.findall(code)),
        # McCabe-style proxy: decision points + 1
        "cyclomatic": n_if + n_loop + len(_CASE.findall(code))
                      + len(_LOGIC.findall(code)) + 1,
        "n_deref": len(_DEREF.findall(code)),
        "n_arrow": len(_ARROW.findall(code)),
        "n_index": len(_INDEX.findall(code)),
        "n_alloc": n_alloc,
        "n_free": n_free,
        # unbalanced alloc/free is the classic CWE-401/415 shape
        "alloc_free_gap": n_alloc - n_free,
        "n_unsafe_api": len(_UNSAFE.findall(code)),
        "n_cast": len(_CAST.findall(code)),
        "n_cmp": len(_CMP.findall(code)),
        "n_call": len(_CALL.findall(code)),
        "call_density": len(_CALL.findall(code)) / max(len(lines), 1),
    }


def frame(units: dict[str, str]) -> pd.DataFrame:
    """units: {unit_id -> source}. Returns a unit_id-indexed feature frame."""
    return pd.DataFrame({u: extract(c) for u, c in units.items()}).T.astype(float)
