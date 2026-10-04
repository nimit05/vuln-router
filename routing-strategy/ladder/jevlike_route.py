#!/usr/bin/env python3
"""Jev-style router: code in, one calibrated P(right) per model out, plus the pick.

    python ladder/jevlike_route.py --router router_emb-lr.pt --code Unit.java --lam 1
    python ladder/jevlike_route.py --router router_ft.pt --units units.jsonl --lam 0.25 --out picks.jsonl

Like Jev it generates no text and answers a typed question ("which model?") with
probabilities; unlike Jev it was trained on which of our 12 models actually got each
OWASP unit right (ladder/l4_learned_router.py --save). The pick maximises
P(right) - lam * gpu_s: lam = 0 is accuracy only, lam = 1 means 0.1 GPU-s must buy at
least 0.1 accuracy.

Output per unit (JSON): {"pick": model, "p_right": {model: p}, "score": {model: p - lam*gpu_s}}
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(path, device):
    r = torch.load(path, map_location="cpu", weights_only=False)
    meta = r["meta"]
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(meta["encoder_path"])
    if meta["encoder"] == "emb-lr":
        from l2_embed import embed
        net = AutoModel.from_pretrained(meta["encoder_path"]).eval()

        def logits(codes):
            return embed(codes, tok, net) @ r["W"] + r["b"]
    else:
        from l4_learned_router import Head
        net = Head(AutoModel.from_pretrained(meta["encoder_path"]), len(meta["models"]))
        net.load_state_dict(r["state"])
        net = net.to(device).eval()

        def logits(codes):
            out = []
            with torch.no_grad():
                for i in range(0, len(codes), 32):
                    t = tok(codes[i:i + 32], truncation=True, max_length=meta["max_len"],
                            padding=True, return_tensors="pt")
                    out.append(net(t["input_ids"].to(device), t["attention_mask"].to(device)).float().cpu())
            return torch.cat(out).numpy()
    return meta, r, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True)
    ap.add_argument("--code", help="one source file")
    ap.add_argument("--units", help="jsonl with unit_id and code")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--out", default="-")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta, r, logits = load(a.router, dev)
    if a.code:
        items = [dict(unit_id=os.path.basename(a.code), code=open(a.code).read())]
    else:
        items = [json.loads(l) for l in open(a.units)]
    z = logits([u["code"] for u in items])
    p = 1 / (1 + np.exp(-(r["platt_a"] * z + r["platt_b"])))
    cost = np.array(meta["gpu_s"])
    f = sys.stdout if a.out == "-" else open(a.out, "w")
    for u, pu in zip(items, p):
        score = pu - a.lam * cost
        rec = dict(unit_id=u["unit_id"], lam=a.lam, pick=meta["models"][int(score.argmax())],
                   p_right={m: round(float(x), 3) for m, x in zip(meta["models"], pu)},
                   score={m: round(float(x), 3) for m, x in zip(meta["models"], score)})
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
