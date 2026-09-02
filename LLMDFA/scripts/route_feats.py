"""Static, pre-LLM features per benchmark program (no model involved)."""
import os, re, sys, json, collections

ROOT = os.environ.get("LLMDFA_BENCH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark"))

IF   = re.compile(r"\bif\s*\(")
LOOP = re.compile(r"\b(for|while)\s*\(")
COND = re.compile(r"\bif\s*\(([^\{]*)\)")          # condition text
CALL = re.compile(r"\b[A-Za-z_]\w*\s*\(")
MATH = re.compile(r"\bMath\s*\.")
RANDish = re.compile(r"Random|staticReturnsTrue|staticReturnsFalse|nextInt|currentTimeMillis")
STATICF = re.compile(r"\b(IO\.static|static\s+\w+\s+\w+\s*=)")
METHOD = re.compile(r"(public|private|protected)\s+[\w<>\[\],\s]+\s+\w+\s*\(")

def index():
    """stem -> [paths]. Juliet splits variants like _51 into _51a.java/_51b.java;
    the LLMDFA log names the case by the base stem, so register both."""
    m = {}
    for dp, _, fns in os.walk(ROOT):
        for fn in fns:
            if not fn.endswith(".java"):
                continue
            stem, path = fn[:-5], os.path.join(dp, fn)
            m.setdefault(stem, []).append(path)
            base = re.sub(r"[a-g]$", "", stem)          # _51a -> _51
            if base != stem:
                m.setdefault(base, []).append(path)
    return m

def extract(paths):
    """Features over the whole program: multi-file variants are concatenated."""
    if isinstance(paths, str):
        paths = [paths]
    s = "\n".join(open(p, errors="replace").read() for p in sorted(paths))
    conds = COND.findall(s)
    cond_txt = " ".join(conds)
    lines = [l for l in s.splitlines() if l.strip()]
    return {
        "loc": len(lines),
        "chars": len(s),
        "n_if": len(IF.findall(s)),
        "n_loop": len(LOOP.findall(s)),
        "n_methods": len(METHOD.findall(s)),
        "n_cond": len(conds),
        "cond_calls": len(CALL.findall(cond_txt)),
        "cond_math": len(MATH.findall(cond_txt)),
        "cond_nondet": len(RANDish.findall(cond_txt)),
        "has_static": 1 if STATICF.search(s) else 0,
        "n_try": s.count("try {"),
    }

if __name__ == "__main__":
    idx = index()
    print(f"indexed {len(idx)} unique java stems")
    sample = list(idx)[:3]
    for st in sample:
        print(" ", st, extract(idx[st]))
