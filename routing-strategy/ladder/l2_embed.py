#!/usr/bin/env python3
"""L2a -- MiniLM embedding of every OWASP unit, for the kNN router (CPU is enough).

all-MiniLM-L6-v2 reads at most 256 tokens, and a unit is ~800, so each unit is
cut into 256-token windows; each window is mean-pooled over its tokens (what
sentence-transformers does for this model), the windows are averaged, and the
result is L2-normalised. Output: an .npy matrix in the units file's order and
a .json list of the matching unit_ids.
"""
import argparse, json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def embed(codes, tok, net, win=254, progress=False):
    """windowed mean-pooled, L2-normalised MiniLM embedding of each code string"""
    embs = []
    with torch.no_grad():
        for i, code in enumerate(codes):
            ids = tok(code, add_special_tokens=False)["input_ids"]
            wins = [ids[j:j + win] for j in range(0, max(len(ids), 1), win)]
            batch = tok.pad({"input_ids": [[tok.cls_token_id] + w + [tok.sep_token_id] for w in wins]},
                            return_tensors="pt")
            h = net(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1).float()
            v = ((h * m).sum(1) / m.sum(1)).mean(0)
            embs.append(torch.nn.functional.normalize(v, dim=0).numpy())
            if progress and (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(codes)}", flush=True)
    return np.stack(embs).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="prefix; writes .npy and .ids.json")
    ap.add_argument("--win", type=int, default=254)      # + [CLS] [SEP] = 256
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    net = AutoModel.from_pretrained(a.model).eval()
    units = [json.loads(l) for l in open(a.units)]
    embs = embed([u["code"] for u in units], tok, net, a.win, progress=True)
    np.save(a.out + ".npy", embs)
    json.dump([u["unit_id"] for u in units], open(a.out + ".ids.json", "w"))
    print("done", len(embs))


if __name__ == "__main__":
    main()
