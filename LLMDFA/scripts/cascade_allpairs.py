"""Verifier-gated cascade over EVERY ordered (small, large) model pair.

ROUTER_V2_PERCASE.md 2.1 rejected the cascade on a few hand-picked variants.
This closes it at full coverage: 12 ordered pairs x 3 bug types, replayed from
existing logs. No GPU.

A cascade sends every case to the small model, runs Z3, and re-runs only the
cases Z3 rejected on the large model. It therefore always pays the small model
first, so it can only win if the rejection rate is low AND z3_fail actually
marks the small model's errors.

    python3 cascade_allpairs.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_logsig as R                      # noqa: E402
from score_split import metrics               # noqa: E402

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reproduction")
MIN_CASES = 50


def main():
    merged = R.load([f"{BASE}/logs", f"{BASE}/logs_router"])
    models = sorted({m for (m, _b) in merged})
    short = {m: m.split("-")[0] for m in models}

    print("=== Z3 rejection rate per (model, bug) ===")
    print(f"{'model':<34}{'bug':<6}{'n':>6}{'reject%':>9}{'F1':>7}{'s/case':>9}")
    for (m, b), files in sorted(merged.items()):
        rows = list(files.values())
        rej = sum(1 for r in rows if r["z3_fail"] > 0) / len(rows)
        mm = metrics(rows)
        print(f"{m:<34}{b:<6}{len(rows):>6}{100*rej:>8.1f}%"
              f"{mm['f1']:>7.3f}{mm['sec_per_case']:>9.1f}")

    print("\n=== cascade: small -> (z3_fail) -> large, all ordered pairs ===")
    print(f"{'bug':<6}{'small':<9}{'large':<9}{'rej%':>7}{'F1 sm':>7}{'F1 lg':>7}"
          f"{'F1 casc':>9}{'cost/lg':>9}{'cost/sm':>9}  verdict")
    wins = 0
    for bug in ("xss", "osci", "dbz"):
        for s in models:
            for l in models:
                if s == l or (s, bug) not in merged or (l, bug) not in merged:
                    continue
                fs, fl = merged[(s, bug)], merged[(l, bug)]
                common = sorted(set(fs) & set(fl))
                if len(common) < MIN_CASES:
                    continue
                srows = [fs[f] for f in common]
                lrows = [fl[f] for f in common]
                crows = [fl[f] if fs[f]["z3_fail"] > 0 else fs[f] for f in common]
                cs = sum(r["single time cost"] for r in srows)
                cl = sum(r["single time cost"] for r in lrows)
                cc = cs + sum(fl[f]["single time cost"]
                              for f in common if fs[f]["z3_fail"] > 0)
                f1s = metrics(srows)["f1"]
                f1l = metrics(lrows)["f1"]
                f1c = metrics(crows)["f1"]
                rej = sum(1 for f in common if fs[f]["z3_fail"] > 0) / len(common)
                # A real win must (a) actually escalate something, (b) beat the
                # large model on quality AND cost, and (c) beat the small model
                # on quality -- otherwise you would simply use the small model.
                win = ""
                if rej > 0.0 and cc < cl and f1c >= f1l and f1c > f1s:
                    win = f"WIN (+{f1c-f1s:.3f} vs small, {cc/cs:.2f}x its cost)"
                wins += bool(win)
                print(f"{bug:<6}{short[s]:<9}{short[l]:<9}{100*rej:>6.1f}%"
                      f"{f1s:>7.3f}{f1l:>7.3f}{f1c:>9.3f}{cc/cl:>8.2f}x"
                      f"{cc/cs:>8.2f}x  {win}")
    print(f"\nnon-degenerate wins: {wins}")


if __name__ == "__main__":
    main()
