#!/usr/bin/env python3
"""H5 -- can ANYTHING about a path predict the Hybrid LLM label across projects?

H3/H4 established the two halves of the Hybrid LLM result on this benchmark:
the objective is excellent (label-following 0.405 AvgF1 against 0.207 for the
best single model) and the router cannot reach it (median LOPO AUC 0.494 over
ten pairs, against 0.865 when the split is allowed to leak inside a project).
docs/07-hybrid-llm.md 3 is the argument that this is an INFORMATION ceiling,
not a capacity one: two unrelated feature families land within 0.01 of each
other on both sides of the split.

That argument has a hole. Both families describe the CODE, and code text is
exactly what carries a repository's fingerprint. This script tests the two
candidate fixes that do not describe the code:

  1. **Small-model confidence** (`--feats struct+small`, `small`). Route AFTER
     the cheap call rather than before it, and let the router read the small
     model's own verdict token logprob. A confidence signal describes the
     MODEL's state on this input, not the repository's idioms, so it has a
     reason to transfer where character n-grams do not. This turns Hybrid LLM
     into a cascade: the small call is always paid for, which is affordable
     here only because c_large / c_small is 3.4x (granite-8b vs qwen-7b).

  2. **Per-project rank normalisation** (`--rank`). Replace each feature by its
     percentile WITHIN its own project, so the router sees "long for this repo"
     instead of "1,200 characters, therefore ff4j". Legitimate at inference: a
     scan has the whole repository's candidate set before it routes any of it.
     Ranks for a held-out project are computed from that project's rows only,
     so nothing crosses the fold boundary.

Everything else is held fixed against H2/H4 so the AUC column is comparable
row for row: same pairs file, same `hl_labels` targets, same soft-target
logistic fit, same leave-one-project-out split, same leaky random-5-fold
control. `--feats text` reproduces H2's backbone as the reference row.

Leakage discipline: the router may read only small-side signals. `pred_large`,
`p_large`, `sec_large`, `q_small`, `q_large` and `gold` are the answer or half
of it, and none of them reach the feature matrix.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hl_labels import det_label, match_prob            # noqa: E402
from h1_build_pairs import load_probe, p_vuln          # noqa: E402
from h2_train_router import featurize, fit_predict     # noqa: E402

STRUCT = ("n_hops", "cross_file", "n_methods", "slice_loc", "slice_chars",
          "max_nest", "n_if", "n_loop", "n_call", "n_catch")

SMALL = ("s_p_vuln", "s_margin", "s_pred", "s_dlp", "s_out_tokens",
         "s_retries", "s_parsefail", "s_tlp_mean", "s_tlp_min", "s_tlp_std",
         "s_tlp_frac_lt1", "s_seconds")


def small_signals(r: dict) -> dict[str, float]:
    """What the cheap call reveals about itself, and nothing about the answer.

    `p_vuln` is h1's, so the verdict probability here is the same number the
    label was built from -- but the label compares it against the LARGE model's
    quality, which is what the router has to guess. A row with no parsed verdict
    arrives at p = 0.5 (defect 1 of docs/05-probe-defects.md), i.e. maximally
    unconfident, which is the right thing for a router to see.
    """
    tlp = [float(x) for x in (r.get("token_logprobs") or []) if x is not None]
    arr = np.array(tlp, dtype=float) if tlp else np.array([0.0])
    pv = p_vuln(r)
    return {
        "s_p_vuln": pv,
        "s_margin": abs(pv - 0.5),          # 0 = coin flip, .5 = certain
        "s_pred": float(r.get("pred_label") or 0),
        "s_dlp": float(r.get("decision_logprob") or 0.0),
        "s_out_tokens": float(r.get("out_tokens") or 0),
        "s_retries": float(r.get("n_retries") or 0),
        "s_parsefail": float(r.get("parse_failures") or 0),
        "s_tlp_mean": float(arr.mean()),
        "s_tlp_min": float(arr.min()),
        "s_tlp_std": float(arr.std()),
        "s_tlp_frac_lt1": float((arr < -1.0).mean()),
        "s_seconds": float(r.get("seconds") or 0.0),
    }


def rankify(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Percentile of each feature within its own project, in [0, 1].

    Average ranks, because these features tie heavily (cross_file is 0/1) and
    ordinal ranks would spread tied rows apart arbitrarily. A project with one
    row gets 0.5 everywhere, which is the only honest value there.
    """
    out = np.zeros_like(X, dtype=float)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        n = len(idx)
        if n == 1:
            out[idx] = 0.5
            continue
        for j in range(X.shape[1]):
            out[idx, j] = (rankdata(X[idx, j], method="average") - 1.0) / (n - 1.0)
    return out


def build_matrix(rows, probe, feats: str, rank: bool):
    """Dense feature matrix + names. `text` is handled by the caller (sparse)."""
    names, cols = [], []
    if feats in ("struct", "struct+small"):
        names += list(STRUCT)
    if feats in ("small", "struct+small"):
        names += list(SMALL)
    X = np.zeros((len(rows), len(names)))
    for i, r in enumerate(rows):
        d = dict(r["feat"])
        if feats in ("small", "struct+small"):
            d.update(small_signals(probe[r["id"]]))
        for j, k in enumerate(names):
            X[i, j] = float(d.get(k, 0.0))
    if rank:
        X = rankify(X, np.array([r["project"] for r in rows]))
    return X, names


def fit_dense(Xtr, ytr, Xte, backbone: str, seed: int) -> np.ndarray:
    """Soft-target fit, same duplication trick as h2.fit_predict.

    Each row is entered twice -- class 1 with weight p, class 0 with weight
    1-p -- so weighted log-loss equals cross-entropy against [p, 1-p], which is
    upstream's objective. Works unchanged for det_2cls, where p is 0 or 1 and
    one copy carries zero weight.
    """
    X2 = np.vstack([Xtr, Xtr])
    y2 = np.concatenate([np.ones(len(Xtr)), np.zeros(len(Xtr))])
    w2 = np.concatenate([ytr, 1.0 - ytr])
    keep = w2 > 1e-9
    if backbone == "gbm":
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                             learning_rate=0.06,
                                             random_state=seed)
    else:
        sc = StandardScaler().fit(Xtr)
        X2, Xte = sc.transform(X2), sc.transform(Xte)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    clf.fit(X2[keep], y2[keep], sample_weight=w2[keep])
    return clf.predict_proba(Xte)[:, list(clf.classes_).index(1.0)]


def cv_auc(rows, probe, y, yb, feats, rank, backbone, seed):
    """(LOPO AUC, leaky random-5-fold AUC, LOPO scores per id).

    The random split is not an alternative protocol, it is the control: high
    there and chance under LOPO means the router identified the repository.
    """
    groups = np.array([r["project"] for r in rows])
    if feats == "text":
        splits = [("lopo", [(np.where(groups != g)[0], np.where(groups == g)[0])
                            for g in np.unique(groups)])]
    else:
        splits = [("lopo", [(np.where(groups != g)[0], np.where(groups == g)[0])
                            for g in np.unique(groups)])]
    splits.append(("rand", list(StratifiedKFold(
        5, shuffle=True, random_state=seed).split(np.zeros(len(rows)), yb))))

    res, lopo_scores = {}, {}
    X = names = None
    if feats != "text":
        X, names = build_matrix(rows, probe, feats, rank)
    for name, folds in splits:
        pr = np.zeros(len(rows))
        for tr, te in folds:
            if len(set(np.round(y[tr]).tolist())) < 2:
                pr[te] = float(y[tr].mean())      # nothing to separate
                continue
            if feats == "text":
                Xtr, Xte = featurize([rows[i] for i in tr],
                                     [rows[i] for i in te], 20000)
                pr[te] = fit_predict(Xtr, y[tr], Xte, 1.0, seed)
            else:
                pr[te] = fit_dense(X[tr], y[tr], X[te], backbone, seed)
        res[name] = roc_auc_score(yb, pr)
        if name == "lopo":
            for r, v in zip(rows, pr):
                lopo_scores[r["id"]] = float(v)
    return res["lopo"], res["rand"], lopo_scores


def univariate(rows, probe, yb):
    """AUC of each small-model signal used raw, with no training at all.

    The LLMDFA router (+0.060 F1 held out) was ONE threshold on ONE feature.
    A single column that transfers is worth more than a fitted model that does
    not, so these are reported before any fit. AUC < 0.5 means the feature
    predicts with its sign flipped, which is still signal.
    """
    X, names = build_matrix(rows, probe, "struct+small", rank=False)
    out = []
    for j, k in enumerate(names):
        if len(set(X[:, j])) < 2:
            continue
        out.append((k, roc_auc_score(yb, X[:, j])))
    return sorted(out, key=lambda kv: -abs(kv[1] - 0.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--probe-dir", default="data/gr/probe5")
    ap.add_argument("--suffix", default="units_paths")
    ap.add_argument("--loss-type", default="det_2cls",
                    choices=["det_2cls", "prob_2cls"])
    ap.add_argument("--match-t", type=float, default=0.0)
    ap.add_argument("--backbone", default="linear", choices=["linear", "gbm"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default="text,struct,struct+rank,small,"
                                         "small+rank,struct+small,"
                                         "struct+small+rank")
    ap.add_argument("--emit", default=None,
                    help="config whose LOPO scores are written in h3's schema")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.pairs) if l.strip()]
    small = rows[0]["candidates"][0]["model"]
    large = rows[0]["candidates"][1]["model"]
    probe = load_probe(a.probe_dir, small, a.suffix)
    missing = [r["id"] for r in rows if r["id"] not in probe]
    if missing:
        raise SystemExit(f"{len(missing)} pair rows have no {small} probe row")

    qs = [r["candidates"][0]["scores"]["q"] for r in rows]
    ql = [r["candidates"][1]["scores"]["q"] for r in rows]
    if a.loss_type == "det_2cls":
        y = np.array([1.0 - det_label(s, l, a.match_t) for s, l in zip(qs, ql)])
    else:
        y = np.array([match_prob(s, l, a.match_t) for s, l in zip(qs, ql)])
    yb = (y >= 0.5).astype(int)

    print(f"{small} (small) -> {large} (large)   {len(rows)} paths, "
          f"{len(set(r['project'] for r in rows))} projects")
    print(f"target: {a.loss_type} t={a.match_t:g}, "
          f"P(small suffices) mean={y.mean():.3f}, positives={yb.sum()}")

    print("\nsingle features, no fitting (AUC vs the label):")
    for k, v in univariate(rows, probe, yb)[:8]:
        print(f"  {k:16} {v:.3f}")

    print(f"\nbackbone={a.backbone}")
    print(f"{'features':22} {'AUC LOPO':>9} {'AUC random (leaky)':>20}")
    print("-" * 53)
    best, emitted = None, None
    for cfg in a.configs.split(","):
        cfg = cfg.strip()
        rank = cfg.endswith("+rank")
        feats = cfg[:-5] if rank else cfg
        if feats not in ("text", "struct", "small", "struct+small"):
            raise SystemExit(f"unknown config {cfg}")
        lopo, rand, scores = cv_auc(rows, probe, y, yb, feats, rank,
                                    a.backbone, a.seed)
        print(f"{cfg:22} {lopo:9.3f} {rand:20.3f}")
        if best is None or lopo > best[1]:
            best = (cfg, lopo)
        if a.emit and cfg == a.emit:
            emitted = scores
    print("-" * 53)
    print(f"best LOPO: {best[0]} at AUC {best[1]:.3f}   "
          f"(H4 reference: text backbone, median 0.494 over ten pairs)")

    if a.emit:
        if emitted is None:
            raise SystemExit(f"--emit {a.emit} was not in --configs")
        out = a.out or (os.path.splitext(a.pairs)[0]
                        + f"__{a.loss_type}_t{a.match_t:g}__{a.emit}"
                          f"__{a.backbone}.scores.json")
        json.dump({"loss_type": a.loss_type, "match_t": a.match_t,
                   "backbone": f"h5:{a.emit}:{a.backbone}", "pairs": a.pairs,
                   "small": small, "large": large,
                   "p_small_suffices": emitted}, open(out, "w"), indent=0)
        print(f"wrote {out}  (feed to h3_frontier.py --scores)")


if __name__ == "__main__":
    main()
