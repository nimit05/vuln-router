#!/usr/bin/env python3
"""H2 -- train the Hybrid LLM router, leave-one-project-out.

Upstream fine-tunes `microsoft/deberta-v3-large` on the query text with a
2-class head (train_router.py:331). Two things make that the wrong FIRST run
here, and neither changes the method:

1. **14 projects is the whole universe.** A 400M-parameter encoder on 2,257
   examples with a 63%-of-the-data project (ff4j) is fitting the project, and
   leave-one-project-out would report that honestly as noise. The paper's
   contribution is the LABEL and the THRESHOLD, not the backbone.
2. **A DeBERTa fine-tune per fold needs a GPU**, and the cost table this feeds
   is a GPU-seconds measurement, so the box has to be booked rather than
   borrowed.

So the backbone is pluggable and the CPU one is the default. `--backbone
deberta` writes the fold files upstream's `train_router.py` expects and stops,
because the run itself belongs on gpu0.

The split is leave-one-project-out for the reason in
`graphrouter/b7_split_patch.py`: paths from one repo share sources, sinks and
idioms, so any split inside a project puts near-duplicates on both sides and
every router scores well for the wrong reason (protocol section 4).

Labels come from `hl_labels`, which is upstream's arithmetic unmodified.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hl_labels import det_label, match_prob     # noqa: E402

FEATS = ("n_hops", "cross_file", "n_methods", "slice_loc", "slice_chars",
         "max_nest", "n_if", "n_loop", "n_call", "n_catch")


def build_targets(rows: list[dict], loss_type: str, t: float):
    """Upstream's two targets. `det_2cls` -> hard 0/1, `prob_2cls` -> soft p.

    p is P(route small). Note what `t` does and does not reach: quality takes
    only three values (1 correct, 1-1/rho false positive, 0 false negative), so
    at t = 0 both rules are pure comparisons and the false-negative price rho
    CANNOT move them. rho enters solely through `t`: a tolerance of 1/rho
    forgives a false alarm from the small model, and nothing below t = 1
    forgives a miss. That entanglement is the finding, not a bug here.
    """
    qs = [r["candidates"][0]["scores"]["q"] for r in rows]
    ql = [r["candidates"][1]["scores"]["q"] for r in rows]
    if loss_type == "det_2cls":
        return np.array([1.0 - det_label(a, b, t) for a, b in zip(qs, ql)])
    if loss_type == "prob_2cls":
        return np.array([match_prob(a, b, t) for a, b in zip(qs, ql)])
    raise SystemExit(f"unknown --loss-type {loss_type}")


def featurize(train: list[dict], test: list[dict], max_features: int):
    """Char n-grams on the sliced code plus the structural counts B3 recorded.

    char_wb rather than word: the text is Java, and identifier morphology
    (`getParameter`, `escapeHtml`) carries the signal that word tokens split
    apart. Fitted on TRAIN only -- fitting the vectoriser on all rows leaks the
    held-out project's vocabulary, which is the same leak the split exists to
    prevent.
    """
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=max_features, min_df=3, sublinear_tf=True)
    Xtr_t = vec.fit_transform([r["input"] for r in train])
    Xte_t = vec.transform([r["input"] for r in test])

    def num(rows):
        return np.array([[float(r["feat"].get(k, 0)) for k in FEATS]
                         for r in rows])
    sc = StandardScaler().fit(num(train))
    Xtr_n, Xte_n = sc.transform(num(train)), sc.transform(num(test))

    cwes = sorted({r["cwe"] for r in train})
    def onehot(rows):
        M = np.zeros((len(rows), len(cwes)))
        for i, r in enumerate(rows):
            if r["cwe"] in cwes:
                M[i, cwes.index(r["cwe"])] = 1.0
        return M

    return (sparse.hstack([Xtr_t, sparse.csr_matrix(Xtr_n),
                           sparse.csr_matrix(onehot(train))]).tocsr(),
            sparse.hstack([Xte_t, sparse.csr_matrix(Xte_n),
                           sparse.csr_matrix(onehot(test))]).tocsr())


def fit_predict(Xtr, ytr_soft, Xte, C: float, seed: int) -> np.ndarray:
    """Logistic regression against a SOFT target.

    sklearn takes hard labels only, so each example is entered twice -- once as
    class 1 with weight p, once as class 0 with weight 1-p. Minimising weighted
    log-loss over that pair is identical to cross-entropy against [p, 1-p],
    which is what upstream's `F.cross_entropy(pred_labels, match_prob)` does
    (ranker.py:134). For det_2cls p is already 0 or 1 and one of the two copies
    carries zero weight, so the same code path covers both losses.
    """
    X2 = sparse.vstack([Xtr, Xtr]).tocsr()
    y2 = np.concatenate([np.ones(Xtr.shape[0]), np.zeros(Xtr.shape[0])])
    w2 = np.concatenate([ytr_soft, 1.0 - ytr_soft])
    keep = w2 > 1e-9
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(X2[keep], y2[keep], sample_weight=w2[keep])
    return clf.predict_proba(Xte)[:, list(clf.classes_).index(1.0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--loss-type", default="det_2cls",
                    choices=["det_2cls", "prob_2cls"])
    ap.add_argument("--match-t", type=float, default=0.0,
                    help="upstream's --match_t: the quality-gap tolerance")
    ap.add_argument("--backbone", default="linear", choices=["linear", "deberta"])
    ap.add_argument("--max-features", type=int, default=20000)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.pairs) if l.strip()]
    projects = sorted({r["project"] for r in rows})
    y = build_targets(rows, a.loss_type, a.match_t)
    print(f"{len(rows)} paths, {len(projects)} projects, "
          f"{a.loss_type} t={a.match_t:g}")
    print(f"  target mean P(route small) = {y.mean():.3f}")

    if a.backbone == "deberta":
        d = os.path.splitext(a.pairs)[0] + "__folds"
        os.makedirs(d, exist_ok=True)
        for held in projects:
            for name, sel in (("train", lambda r: r["project"] != held),
                              ("test", lambda r: r["project"] == held)):
                with open(os.path.join(d, f"{held}.{name}.jsonl"), "w") as fh:
                    for r in rows:
                        if sel(r):
                            fh.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(projects)} folds to {d}/")
        print("run upstream on a GPU box:\n"
              f"  python third_party/HybridLLM/train_router.py \\\n"
              f"    --loss_type {a.loss_type} --match_t {a.match_t} \\\n"
              f"    --train_data_path {d}/<project>.train.jsonl --n_candidates 2")
        return

    pred = {}
    for held in projects:
        train = [r for r in rows if r["project"] != held]
        test = [r for r in rows if r["project"] == held]
        ytr = np.array([y[i] for i, r in enumerate(rows) if r["project"] != held])
        if len(set(np.round(ytr).tolist())) < 2:
            # every training path agrees; the head has nothing to separate, so
            # fall back to that constant rather than fitting a degenerate model
            p = np.full(len(test), float(ytr.mean()))
        else:
            Xtr, Xte = featurize(train, test, a.max_features)
            p = fit_predict(Xtr, ytr, Xte, a.C, a.seed)
        for r, v in zip(test, p):
            pred[r["id"]] = float(v)
        print(f"  held out {held[:44]:46} n={len(test):5}  "
              f"mean P(small)={p.mean():.3f}")

    out = a.out or (os.path.splitext(a.pairs)[0]
                    + f"__{a.loss_type}_t{a.match_t:g}.scores.json")
    json.dump({"loss_type": a.loss_type, "match_t": a.match_t,
               "backbone": a.backbone, "pairs": a.pairs,
               "small": rows[0]["candidates"][0]["model"],
               "large": rows[0]["candidates"][1]["model"],
               "p_small_suffices": pred}, open(out, "w"), indent=0)
    print(f"\nwrote {out}  ({len(pred)} paths)")


if __name__ == "__main__":
    main()
