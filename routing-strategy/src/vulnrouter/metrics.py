"""Detection metrics. F1 for comparability, VD-Score and pair accuracy for
everything else -- see docs/01-problem.md §1.2."""
from __future__ import annotations
import numpy as np


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def security_loss(fn: np.ndarray, fp: np.ndarray,
                  c_fn: float = 10.0, c_fp: float = 1.0) -> np.ndarray:
    """Asymmetric per-unit loss (problem doc §1.5). c_fn >> c_fp."""
    return c_fn * np.asarray(fn, float) + c_fp * np.asarray(fp, float)


def vd_score(scores: np.ndarray, labels: np.ndarray,
             fpr_budget: float = 0.005) -> float:
    """False-negative rate at the strictest threshold whose FPR <= budget.

    PrimeVul's deployment metric: how many real vulnerabilities are missed while
    the alert queue stays tolerable. Lower is better. Returns 1.0 when no
    threshold meets the budget (the model cannot be operated at that FPR).
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores)                       # descending
    s, y = scores[order], labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    ok = (fp / n_neg) <= fpr_budget
    if not ok.any():
        return 1.0
    k = int(np.flatnonzero(ok)[-1])                   # most permissive within budget
    return float(1.0 - tp[k] / n_pos)


def pair_accuracy(pred_vuln: np.ndarray, pred_patched: np.ndarray) -> float:
    """Fraction of (vulnerable, patched) pairs called correctly on BOTH sides.

    The metric that separates models reasoning about the defect from models
    reacting to surface form -- and therefore the one worth routing on.
    """
    v, p = np.asarray(pred_vuln, int), np.asarray(pred_patched, int)
    return float(((v == 1) & (p == 0)).mean()) if len(v) else float("nan")
