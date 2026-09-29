#!/usr/bin/env python3
"""D2 -- GraphRouter v2: upstream's GNN, a security objective, IRIS's metric.

B7 measured why GraphRouter loses to the best single model (06-objective-
misalignment.md): its label `eye(n)[argmax(effect)]` asks "which model won THIS
path", and on an 85%-negative population the winner is whichever model says
"not vulnerable". granite-8b keeps 0.759 of the true bugs and wins 11.9% of
labels, so the router never picks it.

v2 keeps upstream's network, masking and optimiser byte-for-byte
(third_party/GraphRouter/model/graph_nn.py: EncoderDecoderNet, form_data) and
changes the three things that measurement blamed:

1. **Edge target = security utility, not a one-hot winner.** With p = the
   model's P(vulnerable) on the path and rho = cost of a miss / cost of a false
   alarm (rho = 10, the value docs/01-problem.md fixed for Hybrid LLM before any
   result was seen):

       true bug    : utility = p            (miss costs 1, scaled by rho/rho)
       false alarm : utility = 1 - p / rho  (false alarm costs 1/rho)

   Every model gets its own soft target on every path, so a model that keeps
   real bugs is credited on the 15% of paths where that matters instead of
   being out-voted on the 85% where it does not.

2. **Decision = argmax(predicted utility - lambda * cost)**, cost being the
   model's mean measured GPU-seconds per path (known before any call), scaled
   so the priciest model is 1. lambda = 0 is upstream's "Performance First".

3. **Nulls alongside every routed row** (PROGRESS.md reporting rules): a
   random router that sends the SAME number of paths per project to each model,
   and the upstream label run through this same trainer (v1), so the only
   difference between v1 and v2 rows is the label.

Leave-one-project-out, as B7: no path is routed by a network that saw its repo.
"""
from __future__ import annotations
import argparse, glob, importlib.util, json, math, os, random, sys
from collections import Counter, defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "third_party/GraphRouter/model"))
from graph_nn import EncoderDecoderNet, form_data                     # noqa: E402

CWE_TEXT = {
    "CWE-94": "Improper Control of Generation of Code ('Code Injection').",
    "CWE-22": "Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal').",
    "CWE-79": "Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting').",
    "CWE-78": "Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection').",
}
LLM_TEXT = {
    "qwen-1.5b": "Qwen2.5-Coder 1.5B instruct, small code model, cheapest.",
    "phi-3.8b": "Phi-4-mini 3.8B instruct, small general model.",
    "qwen-7b": "Qwen2.5-Coder 7B instruct, mid-size code model.",
    "granite-8b": "Granite 8B code instruct, mid-size code model.",
    "deepseek-6.7b": "DeepSeek-Coder 6.7B instruct, mid-size code model.",
    "qwen3-8b-nothink": "Qwen3 8B, mid-size general model, direct answer without reasoning.",
    "nemo-12b-nothink": "Mistral-Nemo 12B instruct, mid-size general model, direct answer.",
    "qwen3-32b-nothink": "Qwen3 32B, large general model, direct answer without reasoning.",
    "qwen3-32b-think": "Qwen3 32B, large general model, long chain-of-thought reasoning before answering.",
}


# --------------------------------------------------------------------------- data
def p_vuln(r: dict) -> float:
    """P(vulnerable) from a probe row; an unparsed verdict is 0.5, never benign."""
    if r.get("p_vuln") is not None:
        return float(r["p_vuln"])
    pred = r.get("pred_label")
    if pred is None:
        return 0.5
    lp = r.get("decision_logprob")
    if lp is None:
        return float(pred)
    pt = math.exp(min(float(lp), 0.0))
    return pt if pred == 1 else 1.0 - pt


def keeps(r: dict) -> bool:
    """Deployed verdict: keep the alert unless the model said 'not vulnerable'."""
    return r.get("pred_label") != 0


def load_probe_file(path: str) -> dict[str, dict]:
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") or ("in_tokens" in r and not r.get("in_tokens")):
            continue
        out[r["unit_id"]] = r
    return out


def load_metrics(path: str):
    spec = importlib.util.spec_from_file_location("score_subset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.metrics


def embed(texts: list[str], cache: str) -> np.ndarray:
    if os.path.exists(cache):
        z = np.load(cache)
        if z.shape[0] == len(texts):
            return z
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")       # upstream's encoder
    z = enc.encode(texts, batch_size=32, show_progress_bar=False)
    np.save(cache, z)
    return z


# ------------------------------------------------------------------------ scoring
class Scorer:
    """IRIS Sec 3.6 via their own metrics(); every project stays in the mean."""

    def __init__(self, paths, projects, metrics):
        self.paths, self.projects, self.metrics = paths, projects, metrics

    def __call__(self, keep: dict[str, bool]) -> dict:
        per = defaultdict(lambda: [0, 0])
        for p in self.paths:
            if keep.get(p["id"], True):
                per[p["project"]][0] += 1
                per[p["project"]][1] += p["label"]
        recs = [{"paths": per[j][0], "tp_paths": per[j][1], "recall": per[j][1] > 0}
                for j in self.projects]
        m = self.metrics(recs)
        kept = [p for p in self.paths if keep.get(p["id"], True)]
        m["tp_kept"] = sum(p["label"] for p in kept)
        m["kept"] = len(kept)
        return m


# ----------------------------------------------------------------------- training
def train_fold(D, train_q, test_q, target, cost_edge, cfg, seed):
    """Upstream's train loop (graph_nn.GNN_prediction.train_validate) with the
    label swapped for `target`. Returns predicted edge scores for test_q."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    Q, M = D["qemb"].shape[0], len(D["models"])
    edge_q = np.repeat(np.arange(Q), M)
    edge_m = np.tile(np.arange(M), Q)
    tr = np.zeros(Q * M, bool); te = np.zeros(Q * M, bool)
    for q in train_q:
        tr[q * M:(q + 1) * M] = True
    for q in test_q:
        te[q * M:(q + 1) * M] = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fd = form_data(dev)
    eff = target.reshape(-1)
    data = fd.formulation(task_id=D["temb"], query_feature=D["qemb"],
                          llm_feature=D["lemb"], org_node=edge_q.tolist(),
                          des_node=edge_m.tolist(), edge_feature=eff,
                          label=eff.reshape(-1, 1),
                          edge_mask=torch.tensor(tr, device=dev),
                          combined_edge=np.stack([cost_edge.reshape(-1), eff], 1),
                          train_mask=tr, valide_mask=np.zeros_like(tr), test_mask=te)
    net = EncoderDecoderNet(query_feature_dim=D["qemb"].shape[1],
                            llm_feature_dim=D["lemb"].shape[1],
                            hidden_features=cfg["embedding_dim"], in_edges=3).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    bce = torch.nn.BCELoss()
    train_mask = torch.tensor(tr, device=dev)
    for _ in range(cfg["epochs"]):
        net.train()
        loss = 0
        for _ in range(cfg["batch_size"]):
            drop = torch.rand(train_mask.size(), device=dev) < cfg["mask_rate"]
            predict = train_mask & drop                 # edges to predict
            see = train_mask & ~drop                    # edges used for messages
            out = net(task_id=data.task_id, query_features=data.query_features,
                      llm_features=data.llm_features, edge_index=data.edge_index,
                      edge_mask=predict, edge_can_see=see, edge_weight=data.combined_edge)
            loss = loss + bce(out.reshape(-1), data.label[predict].reshape(-1))
        (loss / cfg["batch_size"]).backward()
        opt.step(); opt.zero_grad()
    net.eval()
    with torch.no_grad():
        out = net(task_id=data.task_id, query_features=data.query_features,
                  llm_features=data.llm_features, edge_index=data.edge_index,
                  edge_mask=torch.tensor(te, device=dev), edge_can_see=train_mask,
                  edge_weight=data.combined_edge)
    return out.reshape(-1, M).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default=os.path.join(ROOT, "data/gr/probe5"))
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra probe files as name=path (e.g. qwen3-32b-nothink=...)")
    ap.add_argument("--drop", nargs="*", default=["deepseek-6.7b"],
                    help="excluded: deepseek is irreproducible (HANDOFF.md, kappa -0.03)")
    ap.add_argument("--only-ids", default=None,
                    help="restrict to these unit_ids (e.g. the D1 gate sample)")
    ap.add_argument("--units", default=os.path.join(ROOT, "data/gr/units_paths.jsonl"))
    ap.add_argument("--slices", default=os.path.join(ROOT, "data/gr/slices.jsonl"))
    ap.add_argument("--truth", default=os.path.join(ROOT, "data/gr/iris_truth.json"))
    ap.add_argument("--score-subset", default=os.path.join(ROOT, "../IRIS/reproduction/score_subset.py"))
    ap.add_argument("--cost-per-token", default=None,
                    help="JSON {model: [sec_per_in_tok, sec_per_out_tok]} for rows "
                         "whose seconds were not measured at batch 1")
    ap.add_argument("--rho", type=float, default=10.0)
    ap.add_argument("--probe-feature", default=None,
                    help="v3: a model already run on EVERY path (a cascade's first stage);\n"
                         "its P(vulnerable) is appended to the query features, so the\n"
                         "router decides escalation AFTER the cheap call, not before any call")
    ap.add_argument("--lambdas", default="0,0.05,0.1,0.2,0.4,0.8")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1881)
    ap.add_argument("--null-draws", type=int, default=500)
    ap.add_argument("--cache", default=os.path.join(ROOT, "data/gr/d2_qemb.npy"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/gr/d2_results.json"))
    a = ap.parse_args()
    cfg = dict(embedding_dim=8, lr=3e-4, wd=1e-4, batch_size=32, mask_rate=0.5,
               epochs=a.epochs)                       # upstream config.yaml values

    metrics = load_metrics(a.score_subset)
    projects = sorted(json.load(open(a.truth)))
    code = {}
    for l in open(a.units):
        if l.strip():
            u = json.loads(l); code[u["unit_id"]] = u["code"]
    paths = []
    for l in open(a.slices):
        if l.strip():
            r = json.loads(l)
            paths.append(dict(id=r["path_id"], project=r["project"], label=int(r["label"]),
                              cwe=r.get("query_cwe") or "CWE-UNKNOWN",
                              sliceable=bool(r.get("sliceable", 1))))

    probes = {}
    for f in sorted(glob.glob(os.path.join(a.probe_dir, "*__units_paths.jsonl"))):
        name = os.path.basename(f).split("__")[0]
        if name not in a.drop:
            probes[name] = load_probe_file(f)
    for spec in a.extra:
        name, f = spec.split("=", 1)
        probes[name] = load_probe_file(f)
    models = sorted(probes, key=lambda m: list(LLM_TEXT).index(m) if m in LLM_TEXT else 99)

    # routable = sliceable and measured by every model; the rest are kept by every
    # configuration (a filter that never ran keeps the alert) and scored as such
    only = set(json.load(open(a.only_ids))) if a.only_ids else None
    if only is not None:
        paths = [p for p in paths if p["id"] in only]
    Rq = [p for p in paths if p["sliceable"] and all(p["id"] in probes[m] for m in models)]
    print(f"{len(paths)} paths, {len(Rq)} routable, {sum(p['label'] for p in Rq)} TP; "
          f"models: {models}")
    score = Scorer(paths, projects, metrics)

    # per-row cost in batch-1 GPU-seconds
    cpt = json.load(open(a.cost_per_token)) if a.cost_per_token else {}
    def seconds(m, r):
        if m in cpt:
            si, so = cpt[m]
            return r["in_tokens"] * si + r["out_tokens"] * so
        return float(r["seconds"])
    P = np.array([[p_vuln(probes[m][p["id"]]) for m in models] for p in Rq])
    K = np.array([[keeps(probes[m][p["id"]]) for m in models] for p in Rq])
    S = np.array([[seconds(m, probes[m][p["id"]]) for m in models] for p in Rq])
    y = np.array([p["label"] for p in Rq])
    mean_sec = S.mean(0)
    cost_m = mean_sec / mean_sec.max()
    cost_edge = S / S.max()

    util = np.where(y[:, None] == 1, P, 1.0 - P / a.rho)                # v2 target
    onehot = np.eye(len(models))[np.argmax(np.where(y[:, None] == 1, P, 1 - P), 1)]  # v1

    qemb = embed([code[p["id"]] for p in Rq], a.cache if only is None else a.cache + ".gate.npy")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    temb_of = dict(zip(CWE_TEXT, enc.encode(list(CWE_TEXT.values()))))
    temb = np.stack([temb_of.get(p["cwe"], np.zeros(qemb.shape[1])) for p in Rq])
    lemb = enc.encode([LLM_TEXT.get(m, m) for m in models])
    if a.probe_feature:
        # MiniLM dims are ~0.05 in size; p and logit(p)/5 are O(1), so the linear
        # query projection cannot drown the one feature that carries skill
        pf = P[:, models.index(a.probe_feature)].clip(1e-4, 1 - 1e-4)
        qemb = np.concatenate([qemb, pf[:, None], (np.log(pf / (1 - pf)) / 5)[:, None]], 1)
        print(f"v3: {a.probe_feature} P(vulnerable) appended to query features")
    D = dict(qemb=qemb, temb=temb, lemb=lemb, models=models)

    proj_of = np.array([p["project"] for p in Rq])
    folds = [j for j in projects if (proj_of == j).any()]

    def lopo(target):
        pred = np.zeros_like(util)
        for i, held in enumerate(folds, 1):
            te = np.where(proj_of == held)[0]; tr = np.where(proj_of != held)[0]
            pred[te] = train_fold(D, tr, te, target, cost_edge, cfg, a.seed)
            print(f"    fold {i}/{len(folds)} {held[:40]} ({len(te)})", flush=True)
        return pred

    def keep_of(choice):
        keep = {p["id"]: True for p in paths}
        for i, p in enumerate(Rq):
            keep[p["id"]] = bool(K[i, choice[i]])
        return keep

    def matched_null(choice, rng):
        """Same number of paths per model per project, assigned at random."""
        c = choice.copy()
        for j in folds:
            idx = np.where(proj_of == j)[0]
            c[idx] = rng.permutation(choice[idx])
        return c

    rows = []
    def add(name, choice=None, keep=None, null=False):
        m = score(keep if keep is not None else keep_of(choice))
        row = dict(name=name, **{k: m[k] for k in ("detected", "avg_fdr", "avg_f1", "tp_kept", "kept")})
        if choice is not None:
            row["gpu_s"] = float(S[np.arange(len(Rq)), choice].sum())
            row["share"] = {models[k]: int(v) for k, v in Counter(choice.tolist()).items()}
            if null:
                rng = np.random.default_rng(0)
                f1 = [score(keep_of(matched_null(choice, rng)))["avg_f1"]
                      for _ in range(a.null_draws)]
                row["null_f1"] = [float(np.mean(f1)), float(np.std(f1))]
                row["z"] = (m["avg_f1"] - np.mean(f1)) / max(np.std(f1), 1e-9)
        rows.append(row)
        z = f"  null {row['null_f1'][0]:.3f}+-{row['null_f1'][1]:.3f} z={row['z']:+.1f}" if "z" in row else ""
        print(f"  {name:42} det {m['detected']:2}/16 FDR {m['avg_fdr']:6.2f} "
              f"F1 {m['avg_f1']:.3f} TPkept {m['tp_kept']:3}/{int(y.sum())} "
              f"GPU-s {row.get('gpu_s', 0):8.0f}{z}", flush=True)

    add("no filter", keep={p["id"]: True for p in paths})
    for k, m in enumerate(models):
        add(f"always-{m}", choice=np.full(len(Rq), k))
    or_choice = np.array([next((k for k in range(len(models)) if K[i, k] == bool(y[i])), 0)
                          for i in range(len(Rq))])
    add("per-path oracle", choice=or_choice)
    rng = np.random.default_rng(1)
    sh = []
    for _ in range(50):
        Ks = K.copy()
        for j in folds:
            idx = np.where(proj_of == j)[0]
            for k in range(len(models)):
                Ks[idx, k] = rng.permutation(K[idx, k])
        c = np.array([next((k for k in range(len(models)) if Ks[i, k] == bool(y[i])), 0)
                      for i in range(len(Rq))])
        keep = {p["id"]: True for p in paths}
        for i, p in enumerate(Rq):
            keep[p["id"]] = bool(Ks[i, c[i]])
        sh.append(score(keep)["avg_f1"])
    print(f"  shuffled oracle (skill destroyed)          F1 {np.mean(sh):.3f}+-{np.std(sh):.3f}")
    rows.append(dict(name="shuffled oracle", avg_f1=float(np.mean(sh)), sd=float(np.std(sh))))

    print("\n  -- training v1 (upstream one-hot argmax label), LOPO", flush=True)
    pred1 = lopo(onehot)
    add("GraphRouter v1 (upstream label)", choice=pred1.argmax(1), null=True)

    print("\n  -- training v2 (security-utility label, rho=%g), LOPO" % a.rho, flush=True)
    pred2 = lopo(util)
    for lam in [float(x) for x in a.lambdas.split(",")]:
        add(f"GraphRouter v2 lambda={lam:g}", choice=(pred2 - lam * cost_m).argmax(1), null=True)

    np.savez(os.path.splitext(a.out)[0] + ".npz", pred_v1=pred1, pred_v2=pred2,
             P=P, K=K, S=S, y=y, ids=np.array([p["id"] for p in Rq]),
             models=np.array(models))
    json.dump(dict(models=models, rho=a.rho, epochs=a.epochs, rows=rows,
                   mean_sec=mean_sec.tolist(),
                   mean_util=util.mean(0).tolist(),
                   util_on_tp=util[y == 1].mean(0).tolist()),
              open(a.out, "w"), indent=1, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
