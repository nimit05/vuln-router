"""Scoring, the cost-quality frontier, and the three pre-registered gates.

Protocol: docs/03-protocol.md. Nothing here fits anything -- fitting lives in the
caller, on the fit split only, so that this module can never leak held-out data
into a selection decision.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd
from .table import Table
from .metrics import prf1
from .strategies import Strategy, Fixed, Oracle


def align(t: Table, feats: pd.DataFrame) -> np.ndarray:
    """Reindex a per-unit feature frame onto the table's unit order.

    NOT optional. `load()` pivots by unit_id, so the table's row order is the
    SORTED unit order, which is lexicographic ("u0, u1, u10, u100"), not the
    order features were generated in. Passing a positional array instead of
    going through here silently misaligns every feature with every cost and
    every loss, and the result is a gate that quietly reports "no signal".
    """
    if not isinstance(feats, pd.DataFrame):
        raise TypeError("features must be a DataFrame indexed by unit_id, "
                        "so they can be aligned to the table (see align()).")
    out = feats.reindex(t.units)
    if out.isna().any().any():
        missing = int(out.isna().any(axis=1).sum())
        raise ValueError(f"{missing} units have no features after alignment")
    return out.to_numpy(float)


def score(t: Table, choice: np.ndarray) -> dict:
    """Confusion, F1, total security loss and total cost for one routing."""
    u = np.arange(len(t.units))
    fn, fp = t.fn[u, choice], t.fp[u, choice]
    tp = int(((t.gold == 1) & (fn == 0)).sum())
    fn_n, fp_n = int(fn.sum()), int(fp.sum())
    p, r, f1 = prf1(tp, fp_n, fn_n)
    return dict(tp=tp, fp=fp_n, fn=fn_n, precision=p, recall=r, f1=f1,
                loss=float(t.loss[u, choice].sum()),
                cost=float(t.cost[u, choice].sum()))


def frontier(t: Table, strategy: Strategy, lams) -> pd.DataFrame:
    """Sweep lam and return the cost-quality curve. The reporting object."""
    out = []
    for lam in lams:
        row = score(t, strategy.route(t, lam))
        out.append(dict(strategy=strategy.name, lam=lam, **row))
    return pd.DataFrame(out)


def aiq(curve: pd.DataFrame, cost_lo: float, cost_hi: float,
        quality: str = "f1") -> float:
    """Area under the cost-quality frontier, normalised to [cost_lo, cost_hi].

    A scalar summary of a curve. Comparing two routers at one operating point is
    not a comparison; this is (docs/03-protocol.md §2).
    """
    c = curve.sort_values("cost")
    x = np.clip(c.cost.to_numpy(float), cost_lo, cost_hi)
    y = c[quality].to_numpy(float)
    keep = np.concatenate(([True], np.diff(x) > 0))
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return float("nan")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def references(t: Table, lam: float = 0.0) -> pd.DataFrame:
    """Every fixed model plus the per-unit oracle. The three lines every plot
    must carry -- including ALL fixed models, not just the best one, since
    naming 'the best fixed model' is already a mild oracle."""
    rows = [dict(strategy=f"always-{m}", **score(t, Fixed(m).route(t, lam)))
            for m in t.models]
    rows.append(dict(strategy="ORACLE", **score(t, Oracle().route(t, lam))))
    return pd.DataFrame(rows).sort_values("f1", ascending=False)


# --------------------------------------------------------------------------
# Pre-registered gates (docs/03-protocol.md §3)
# --------------------------------------------------------------------------

def gate_g1_pool_complementarity(t: Table, lam: float = 0.0) -> pd.DataFrame:
    """G1 -- do any two models have complementary errors?

    For every pair, the 2-model oracle F1 against the better member's F1. If no
    pair shows a material lift, the pool is effectively ordered and NO router
    can win: change the pool before writing router code, and report the negative.
    """
    solo = {m: score(t, Fixed(m).route(t, lam))["f1"] for m in t.models}
    rows = []
    for a, b in combinations(t.models, 2):
        ia, ib = t.models.index(a), t.models.index(b)
        sub = t.loss[:, [ia, ib]] + lam * t.cost[:, [ia, ib]]
        pick = np.array([ia, ib])[np.argmin(sub, axis=1)]
        orc = score(t, pick)["f1"]
        best = max(solo[a], solo[b])
        rows.append(dict(pair=f"{a}+{b}", oracle_f1=orc, best_member_f1=best,
                         lift=orc - best))
    return pd.DataFrame(rows).sort_values("lift", ascending=False)


def gate_g2_cost_endogeneity(t: Table, feats: pd.DataFrame,
                             seed: int = 0) -> pd.DataFrame:
    """G2 -- is cost predictable from the unit, or is it a per-model constant?

    Every routing paper assumes c(m,x) = c(m). Here we test it: held-out R^2 of
    a linear fit of per-unit cost on unit features, per model. R^2 materially
    above 0 means cost is endogenous and the cost regressor belongs in the
    argmin; R^2 ~ 0 means adopt the standard constant-cost formulation and say so.
    """
    n = len(t.units)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    A = np.hstack([align(t, feats), np.ones((n, 1))])
    rows = []
    for j, m in enumerate(t.models):
        y = t.cost[:, j]
        beta, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
        pred = A[te] @ beta
        ss_res = float(((y[te] - pred) ** 2).sum())
        ss_tot = float(((y[te] - y[tr].mean()) ** 2).sum())   # vs constant c(m)
        rows.append(dict(model=m, mean_cost=float(y.mean()),
                         cv=float(y.std() / max(y.mean(), 1e-9)),
                         r2_vs_constant=1 - ss_res / max(ss_tot, 1e-12)))
    return pd.DataFrame(rows)


def gate_g3_cascade_fire_rate(fires: dict[str, np.ndarray],
                              t: Table | None = None) -> pd.DataFrame:
    """G3 -- how often does each escalation trigger fire, and is it informative?

    A cascade always pays for the base model, so it saves money only when
    escalation is rare. `error_lift` is P(error | fired) / P(error) for the base
    model: a trigger that fires on most units selects everything and
    discriminates nothing, whatever its lift.
    """
    rows = []
    for name, fired in fires.items():
        f = np.asarray(fired, bool)
        row = dict(trigger=name, fire_rate=float(f.mean()))
        if t is not None:
            base_err = (t.fn[:, 0] + t.fp[:, 0]) > 0
            row["error_lift"] = (float(base_err[f].mean() / base_err.mean())
                                 if f.any() and base_err.mean() > 0 else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_margin(t: Table, a: np.ndarray, b: np.ndarray,
                     n_boot: int = 2000, seed: int = 0,
                     quality: str = "f1") -> tuple[float, float, float]:
    """CI on (strategy a - strategy b) quality, resampling units.

    Report this beside every headline delta. If the interval already excludes
    zero, more data will not change the conclusion -- do not spend GPU hours
    narrowing it (protocol §7).
    """
    n = len(t.units)
    rng = np.random.default_rng(seed)
    point = score(t, a)[quality] - score(t, b)[quality]
    draws = np.empty(n_boot)
    for i in range(n_boot):
        rows = rng.integers(0, n, n)
        draws[i] = _score_idx(t, a, rows)[quality] - _score_idx(t, b, rows)[quality]
    return point, float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))


def _score_idx(t: Table, choice: np.ndarray, rows: np.ndarray) -> dict:
    fn, fp = t.fn[rows, choice[rows]], t.fp[rows, choice[rows]]
    gold = t.gold[rows]
    tp = int(((gold == 1) & (fn == 0)).sum())
    p, r, f1 = prf1(tp, int(fp.sum()), int(fn.sum()))
    return dict(f1=f1, precision=p, recall=r)
