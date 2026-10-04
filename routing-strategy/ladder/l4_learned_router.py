#!/usr/bin/env python3
"""L4 -- a learned, Jev-style router: read the unit, predict P(model m is right) for
every model in the pool, then pick argmax  P(right) - lam * gpu_s.

Jev (TypeSafe AI) is a zero-shot classifier: it guesses how demanding a query is from
the text and hand-written model descriptions. Here the same interface -- code in,
one calibrated probability per model out, no text generated -- is TRAINED on what
actually happened: for every OWASP unit we know which of the 12 models was right.

Two encoders:
  emb-lr    the saved MiniLM embedding -> one logistic regression per model (CPU)
  finetune  UniXcoder (125M code encoder) fine-tuned with one output per model (GPU)

Calibration ("0.9 means right 9 times in 10"): a per-model Platt scaling
(p = sigmoid(a * logit + b)) fitted on held-out train predictions -- out-of-fold for
emb-lr, a 15% validation slice for finetune. Never on test.

Writes predictions for (1) the test split, from a router fitted on train, and
(2) leave-one-category-out: every unit predicted by a router that never saw its
category. Scored by l5_eval_learned.py.
"""
import argparse, glob, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F

SAVED = {}          # parameters of the last router fitted, for --save


def load(a):
    labels = {r["unit_id"]: r for r in map(json.loads, open(a.labels))}
    split = json.load(open(a.split))["split"]
    res = {}
    for f in sorted(glob.glob(os.path.join(a.probes, "*__owasp.jsonl"))):
        name = os.path.basename(f).split("__")[0]
        res[name] = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f)}
    models = sorted(res)
    ids = sorted(labels)
    Y = np.array([[res[m][u]["pred_label"] == labels[u]["gold_label"] for m in models]
                  for u in ids], np.float32)
    return labels, split, models, ids, Y


# ---------------------------------------------------------------- calibration
def platt(z, y):
    """per-column a, b so that sigmoid(a*z + b) fits y (z, y: n x M)"""
    z, y = torch.tensor(z), torch.tensor(y)
    a = torch.ones(z.shape[1], requires_grad=True)
    b = torch.zeros(z.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(a * z + b, y)
        loss.backward()
        return loss
    opt.step(closure)
    return a.detach().numpy(), b.detach().numpy()


def apply_platt(z, ab):
    a, b = ab
    return 1 / (1 + np.exp(-(a * z + b)))


# ---------------------------------------------------------------- emb-lr
def fit_lr(X, Y, wd):
    X, Y = torch.tensor(X), torch.tensor(Y)
    W = torch.zeros(X.shape[1], Y.shape[1], requires_grad=True)
    b = torch.zeros(Y.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(X @ W + b, Y) + wd * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach().numpy(), b.detach().numpy()


def lr_router(Xtr, Ytr, Xte, seed=0):
    """choose weight decay by 5-fold CV, Platt on out-of-fold logits, predict Xte"""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(Xtr)), 5)
    best = None
    for wd in (0.0, 1e-6, 1e-5, 3e-5, 1e-4, 1e-3, 1e-2):
        oof = np.zeros_like(Ytr)
        for f in folds:
            tr = np.setdiff1d(np.arange(len(Xtr)), f)
            W, b = fit_lr(Xtr[tr], Ytr[tr], wd)
            oof[f] = Xtr[f] @ W + b
        loss = F.binary_cross_entropy_with_logits(torch.tensor(oof), torch.tensor(Ytr)).item()
        if best is None or loss < best[0]:
            best = (loss, wd, oof)
    _, wd, oof = best
    ab = platt(oof, Ytr)
    W, b = fit_lr(Xtr, Ytr, wd)
    SAVED.update(W=W, b=b, platt_a=ab[0], platt_b=ab[1])
    return apply_platt(Xte @ W + b, ab), dict(wd=wd, cv_bce=round(best[0], 4))


# ---------------------------------------------------------------- finetune
class Head(torch.nn.Module):
    def __init__(self, enc, n_out):
        super().__init__()
        self.enc = enc
        self.out = torch.nn.Linear(enc.config.hidden_size, n_out)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.out(((h * m).sum(1) / m.sum(1)).float())


def ft_router(codes_tr, Ytr, codes_te, a, seed=0):
    """fine-tune the encoder on 85% of train, keep the best epoch by validation BCE,
    Platt on the validation slice, predict codes_te"""
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(a.encoder_path)
    dev = "cuda"

    def encode(codes):
        t = tok(codes, truncation=True, max_length=a.max_len, padding="max_length", return_tensors="pt")
        return t["input_ids"], t["attention_mask"]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(codes_tr))
    nv = int(0.15 * len(perm))
    va, tr = perm[:nv], perm[nv:]
    Itr, Mtr = encode([codes_tr[i] for i in tr])
    Iva, Mva = encode([codes_tr[i] for i in va])
    Ite, Mte = encode(codes_te)
    Ytr_t = torch.tensor(Ytr[tr])
    net = Head(AutoModel.from_pretrained(a.encoder_path), Ytr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(tr) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

    def logits(I, M):
        net.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(I), 64):
                out.append(net(I[i:i + 64].to(dev), M[i:i + 64].to(dev)).float().cpu())
        return torch.cat(out).numpy()

    best = None
    for ep in range(a.epochs):
        net.train()
        order = torch.randperm(len(tr))
        for i in range(0, len(tr), a.bs):
            j = order[i:i + a.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = net(Itr[j].to(dev), Mtr[j].to(dev))
            loss = F.binary_cross_entropy_with_logits(z.float(), Ytr_t[j].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        zva = logits(Iva, Mva)
        vl = F.binary_cross_entropy_with_logits(torch.tensor(zva), torch.tensor(Ytr[va])).item()
        print(f"    epoch {ep + 1} val_bce {vl:.4f}", flush=True)
        if best is None or vl < best[0]:
            best = (vl, ep + 1, zva, logits(Ite, Mte))
            if a.save:
                state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    vl, ep, zva, zte = best
    ab = platt(zva, Ytr[va])
    if a.save:
        SAVED.update(state=state, platt_a=ab[0], platt_b=ab[1])
    del net; torch.cuda.empty_cache()
    return apply_platt(zte, ab), dict(best_epoch=ep, val_bce=round(vl, 4))


def main():
    ap = argparse.ArgumentParser()
    R = "results/2026-09-owasp-ladder"
    ap.add_argument("--probes", default=f"{R}/probes")
    ap.add_argument("--labels", default=f"{R}/labels.jsonl")
    ap.add_argument("--split", default=f"{R}/split.json")
    ap.add_argument("--emb", default=f"{R}/emb_minilm")
    ap.add_argument("--emb-model", default="all-MiniLM-L6-v2", help="embedder used for --emb (saved in the router)")
    ap.add_argument("--units", default="data/owasp_ladder/units_owasp.jsonl", help="code, for finetune")
    ap.add_argument("--encoder", choices=["emb-lr", "finetune"], required=True)
    ap.add_argument("--encoder-path", default="microsoft/unixcoder-base")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--no-loco", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="name in the output (default: the encoder)")
    ap.add_argument("--target", choices=["models", "label"], default="models",
                    help="models: P(each model right) = the router; label: P(vulnerable) directly, "
                         "no LLM -- the control for a router that solves the task itself")
    ap.add_argument("--apply", default="", help="cross-benchmark: fit on ALL units of the source "
                    "benchmark, predict every unit of this jsonl (unit_id, code); no split, no LOCO")
    ap.add_argument("--apply-emb", default="", help="MiniLM embedding prefix for --apply units (emb-lr)")
    ap.add_argument("--save", default="", help="save the router fitted on train (for jevlike_route.py)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    labels, split, models, ids, Y = load(a)
    if a.target == "label":
        models = ["direct"]
        Y = np.array([[labels[u]["gold_label"]] for u in ids], np.float32)
    cat = np.array([labels[u]["project"] for u in ids])
    itr = np.array([i for i, u in enumerate(ids) if split[u] == "train"])
    ite = np.array([i for i, u in enumerate(ids) if split[u] == "test"])
    if a.encoder == "emb-lr":
        E = np.load(a.emb + ".npy").astype(np.float32)
        eids = json.load(open(a.emb + ".ids.json"))
        X = E[[eids.index(u) for u in ids]]
        run = lambda tr, te: lr_router(X[tr], Y[tr], X[te], a.seed)
    else:
        code = {u["unit_id"]: u["code"] for u in map(json.loads, open(a.units))}
        C = [code[u] for u in ids]
        run = lambda tr, te: ft_router([C[i] for i in tr], Y[tr], [C[i] for i in te], a, a.seed)

    out = dict(models=models, target=a.target, encoder=a.tag or a.encoder, main={}, loco={}, info={})
    if a.apply:
        src = np.arange(len(ids))
        tgt = [json.loads(l) for l in open(a.apply)]
        t = time.time()
        if a.encoder == "emb-lr":
            Ea = np.load(a.apply_emb + ".npy").astype(np.float32)
            aid = json.load(open(a.apply_emb + ".ids.json"))
            pos = {u: i for i, u in enumerate(aid)}
            P, info = lr_router(X[src], Y[src], Ea[[pos[u["unit_id"]] for u in tgt]], a.seed)
        else:
            P, info = ft_router([C[i] for i in src], Y[src], [u["code"] for u in tgt], a, a.seed)
        out["apply"] = {u["unit_id"]: [round(float(x), 5) for x in p] for u, p in zip(tgt, P)}
        out["info"]["apply"] = dict(info, source_units=len(src), target=os.path.basename(a.apply))
        print(f"applied to {len(tgt)} units {time.time() - t:.0f}s {info}", flush=True)
        json.dump(out, open(a.out, "w"))
        return
    t = time.time()
    P, info = run(itr, ite)
    out["main"] = {ids[i]: [round(float(x), 5) for x in p] for i, p in zip(ite, P)}
    out["info"]["main"] = info
    print(f"main split done {time.time() - t:.0f}s {info}", flush=True)
    if a.save:
        cost = {}
        for l in open(os.path.join(a.probes, "timing.jsonl")):
            r = json.loads(l)
            c = r["wall_s"] / r["units"]
            cost[r["model"]] = min(c, cost.get(r["model"], c))
        meta = dict(models=models, gpu_s=[cost[m] for m in models], encoder=a.encoder,
                    encoder_path=a.encoder_path if a.encoder == "finetune" else a.emb_model,
                    max_len=a.max_len)
        torch.save(dict(meta=meta, **SAVED), a.save)
        print(f"saved router -> {a.save}", flush=True)
    if not a.no_loco:
        for c in sorted(set(cat)):
            ho, ot = np.where(cat == c)[0], np.where(cat != c)[0]
            P, info = run(ot, ho)
            out["loco"].update({ids[i]: [round(float(x), 5) for x in p] for i, p in zip(ho, P)})
            out["info"][c] = info
            print(f"loco {c} done {time.time() - t:.0f}s {info}", flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
