"""The measurement table: one row per (unit, model), produced by probe.py.

Everything downstream -- every strategy, every gate, the whole frontier -- reads
this object and nothing else. Schema is docs/03-protocol.md §5.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd

REQUIRED = ["unit_id", "model", "seconds", "pred_label", "gold_label"]


@dataclass
class Table:
    units: pd.Index          # unit_id, ordered
    models: list[str]        # short names, ordered
    loss: np.ndarray         # (n_units, n_models) asymmetric security loss
    cost: np.ndarray         # (n_units, n_models) GPU-seconds
    fn: np.ndarray           # (n_units, n_models) 0/1
    fp: np.ndarray           # (n_units, n_models) 0/1
    gold: np.ndarray         # (n_units,) 0/1
    group: np.ndarray        # (n_units,) split group (commit or project)
    raw: pd.DataFrame        # the long-form rows, for anything ad hoc

    @property
    def shape(self) -> tuple[int, int]:
        return self.loss.shape

    def subset(self, mask: np.ndarray) -> "Table":
        m = np.asarray(mask, bool)
        return Table(self.units[m], self.models, self.loss[m], self.cost[m],
                     self.fn[m], self.fp[m], self.gold[m], self.group[m],
                     self.raw[self.raw.unit_id.isin(self.units[m])])


def load(path: str, c_fn: float = 10.0, c_fp: float = 1.0,
         group_col: str = "commit") -> Table:
    """Load a probe JSONL and pivot it into dense (unit x model) matrices.

    Only units measured by EVERY model are kept. A partially-measured unit
    cannot be routed -- the oracle and every counterfactual would be undefined
    on it -- and silently dropping the requirement is how cross-model
    comparisons stop being matched.
    """
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"probe table missing required columns: {missing}")

    models = sorted(df.model.unique())
    counts = df.groupby("unit_id").model.nunique()
    complete = counts[counts == len(models)].index
    dropped = len(counts) - len(complete)
    if dropped:
        print(f"[table] dropped {dropped} partially-measured units "
              f"({len(complete)} kept, {len(models)} models)")
    df = df[df.unit_id.isin(complete)].copy()

    df["fn"] = ((df.gold_label == 1) & (df.pred_label == 0)).astype(int)
    df["fp"] = ((df.gold_label == 0) & (df.pred_label == 1)).astype(int)

    piv = lambda col: (df.pivot(index="unit_id", columns="model", values=col)
                         .reindex(columns=models).to_numpy())
    fn, fp, cost = piv("fn"), piv("fp"), piv("seconds")
    units = (df.pivot(index="unit_id", columns="model", values="fn")).index

    first = df.drop_duplicates("unit_id").set_index("unit_id").reindex(units)
    gold = first.gold_label.to_numpy().astype(int)
    group = (first[group_col].to_numpy() if group_col in first
             else np.asarray(units, dtype=object))

    return Table(units, models, c_fn * fn + c_fp * fp, cost,
                 fn, fp, gold, group, df)
