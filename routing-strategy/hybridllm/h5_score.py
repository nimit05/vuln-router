#!/usr/bin/env python3
"""H5 -- pool the per-fold DeBERTa predictions into one LOPO AUC.

Each fold directory holds upstream's `predictions.pt` (n, 2) -- a logit per
candidate, in `--candidate_models` order, so column 0 is the small model --
and `labels.pt`. Row order follows the fold's test jsonl, which is how a
prediction gets its path id back.

The AUC target is built with `hl_labels`, the same functions the linear rows
used, so the number lands in the same column as `07-hybrid-llm.md` 3 and
`RESULTS_HYBRIDLLM.md`. Anything else would be comparing two different
questions.

There is no leaky control here. That row needs five more fine-tunes on random
splits, which is a separate run, not a rescoring.

    python3 hybridllm/h5_score.py --runs runs/srclen512 \
        --folds data/hl/pairs__qwen-7b__granite-8b__rho10__folds \
        --pairs data/hl/pairs__qwen-7b__granite-8b__rho10.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hl_labels import det_label  # noqa: E402


def softmax_small(logits: np.ndarray) -> np.ndarray:
    """P(route small) from the two candidate logits."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="dir of per-project fold outputs")
    ap.add_argument("--folds", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--match-t", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch

    rows = [json.loads(l) for l in open(a.pairs) if l.strip()]
    by_id = {r["id"]: r for r in rows}
    small = rows[0]["candidates"][0]["model"]
    large = rows[0]["candidates"][1]["model"]

    scores: dict[str, float] = {}
    missing_folds = []
    for project in sorted(os.listdir(a.runs)):
        d = os.path.join(a.runs, project)
        pt = os.path.join(d, "predictions.pt")
        if not os.path.isdir(d) or not os.path.exists(pt):
            if os.path.isdir(d):
                missing_folds.append(project)
            continue
        preds = np.asarray(torch.load(pt, map_location="cpu", weights_only=False),
                           dtype=float)
        test = [json.loads(l) for l in
                open(os.path.join(a.folds, f"{project}.test.jsonl")) if l.strip()]
        if len(test) != len(preds):
            raise SystemExit(
                f"{project}: {len(preds)} predictions vs {len(test)} test rows -- "
                f"row order cannot be trusted, refusing to score")
        for r, p in zip(test, softmax_small(preds)):
            scores[r["id"]] = float(p)

    if missing_folds:
        print(f"  {len(missing_folds)} fold(s) with no predictions: "
              f"{', '.join(f[:40] for f in missing_folds)}")

    ids = [i for i in scores if i in by_id]
    p_small = np.array([scores[i] for i in ids])
    y = np.array([1.0 - det_label(by_id[i]["candidates"][0]["scores"]["q"],
                                  by_id[i]["candidates"][1]["scores"]["q"],
                                  a.match_t) for i in ids])
    yb = (y >= 0.5).astype(int)

    print(f"{len(ids)} scored paths of {len(rows)}   "
          f"P(small suffices) target mean = {yb.mean():.3f}")
    if yb.min() == yb.max():
        raise SystemExit("target is constant -- AUC undefined")

    auc = roc_auc_score(yb, p_small)
    print(f"\n  LOPO AUC = {auc:.3f}")
    print(f"  (linear backbone on the same pair and target: 0.471)")
    print(f"  p_small: mean {p_small.mean():.3f}  sd {p_small.std():.3f}  "
          f"min {p_small.min():.3f}  max {p_small.max():.3f}")

    out = a.out or os.path.join(a.runs, "h5.scores.json")
    json.dump({"loss_type": "det_2cls", "match_t": a.match_t,
               "backbone": f"h5:deberta-v3-large:{os.path.basename(a.runs)}",
               "pairs": a.pairs, "small": small, "large": large,
               "lopo_auc": auc, "n_scored": len(ids),
               "p_small_suffices": scores}, open(out, "w"), indent=0)
    print(f"\nwrote {out}  (feed to h3_frontier.py --scores)")


if __name__ == "__main__":
    main()
