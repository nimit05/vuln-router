"""The strategy zoo. Every strategy maps a Table to one model index per unit.

Docs: docs/02-strategies.md. Strategy ids (S0..S10) are quoted in each docstring
so the code and the anchors stay in sync.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .table import Table


class Strategy:
    name: str = "strategy"

    def route(self, t: Table, lam: float) -> np.ndarray:
        """Return an array of model indices, one per unit."""
        raise NotImplementedError


@dataclass
class Fixed(Strategy):
    """S0 -- always the same model. Reported as a point, never a curve."""
    model: str

    def __post_init__(self):
        self.name = f"always-{self.model}"

    def route(self, t: Table, lam: float) -> np.ndarray:
        return np.full(len(t.units), t.models.index(self.model), int)


@dataclass
class Random(Strategy):
    """S0 -- uniform random. The floor a router must clear to be a router."""
    seed: int = 0
    name: str = "random"

    def route(self, t: Table, lam: float) -> np.ndarray:
        return np.random.default_rng(self.seed).integers(
            0, len(t.models), len(t.units))


@dataclass
class Oracle(Strategy):
    """S1 -- per-unit upper bound. Not achievable: it reads the outcome.

    Its COST is as informative as its quality. If the oracle is both better and
    cheaper than the best fixed model, quality and cost are not in tension for
    this pool (docs/02-strategies.md, Family A).
    """
    name: str = "oracle"

    def route(self, t: Table, lam: float) -> np.ndarray:
        return np.argmin(t.loss + lam * t.cost, axis=1)


@dataclass
class PredictiveDefer(Strategy):
    """S7 -- M-way cost-sensitive learn-to-defer (Mozannar & Sontag, ICML 2020).

    Routes argmin_m [ loss_hat[u,m] + lam * cost_hat[u,m] ]. Both predictors are
    fitted on the FIT split only and passed in already frozen; this class does
    no fitting, so it cannot accidentally see held-out data.

    cost_hat is a matrix, not a per-model constant, because cost is endogenous
    (docs/01-problem.md §1.4, gate G2).
    """
    loss_hat: np.ndarray
    cost_hat: np.ndarray
    name: str = "defer-Mway"

    def route(self, t: Table, lam: float) -> np.ndarray:
        return np.argmin(self.loss_hat + lam * self.cost_hat, axis=1)


@dataclass
class BinaryThreshold(Strategy):
    """S5/S6 -- Hybrid LLM (ICLR 2024) / RouteLLM (ICLR 2025) shape.

    One difficulty (or win-probability) score per unit; above the threshold goes
    to `strong`, below to `weak`. The threshold IS the cost knob, so lam is
    mapped to a quantile of the score rather than used directly -- that keeps the
    sweep well-defined whatever scale the scorer emits, and makes the cut point
    relative to the corpus instead of an absolute constant that drifts between
    projects.
    """
    score: np.ndarray
    weak: str
    strong: str
    name: str = "binary-threshold"

    def route(self, t: Table, lam: float) -> np.ndarray:
        q = float(np.clip(lam, 0.0, 1.0))
        thr = np.quantile(self.score, q)
        iw, istr = t.models.index(self.weak), t.models.index(self.strong)
        return np.where(self.score >= thr, istr, iw)


@dataclass
class DualAscent(Strategy):
    """S8 -- workload budget via online primal-dual (Bandits with Knapsacks,
    FOCS 2013 / J.ACM 2018).

    Wraps any strategy that accepts a lam. One budget for the whole scan: lam
    rises when spend runs ahead of schedule and falls when it lags, so the cut
    point calibrates itself to the stream instead of being a fixed constant.

    Caveat, stated in the docs and repeated here because it bounds the claim:
    the regret guarantee is asymptotic. On a short scan the dual may not
    converge, and this degrades to a fixed lam.
    """
    inner: Strategy
    budget: float
    eta: float = 0.05
    lam0: float = 1e-3
    name: str = "dual-ascent"

    def route(self, t: Table, lam: float = 0.0) -> np.ndarray:
        n = len(t.units)
        pace = self.budget / max(n, 1)          # target spend per unit
        lam_t, choice = self.lam0, np.zeros(n, int)
        spent = 0.0
        for u in range(n):
            m = int(np.argmin(self.inner_scores(t, u, lam_t)))
            choice[u] = m
            spent += t.cost[u, m]
            # dual ascent on the budget violation so far
            lam_t = max(0.0, lam_t + self.eta * (spent - pace * (u + 1)) / max(pace, 1e-9))
        self.final_lam = lam_t
        self.realised_cost = spent
        return choice

    def inner_scores(self, t: Table, u: int, lam_t: float) -> np.ndarray:
        if isinstance(self.inner, PredictiveDefer):
            return self.inner.loss_hat[u] + lam_t * self.inner.cost_hat[u]
        if isinstance(self.inner, Oracle):
            return t.loss[u] + lam_t * t.cost[u]
        raise TypeError(f"DualAscent cannot wrap {type(self.inner).__name__}")
