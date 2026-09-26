#!/usr/bin/env python3
"""A3 -- build the LLM nodes in GraphRouter's own format.

Phase A step 2: "Create LLM nodes: write a short description of each model with
its price, encode with the same encoder."

Two things are matched to the reference implementation deliberately:

* the file is `configs/LLM_Descriptions.json` with their exact schema --
  {name: {feature, input_price, output_price, model}} -- so their
  `data_processing/construct_router_data.py` and loader read it unmodified;
* the encoder is `all-MiniLM-L6-v2` via sentence-transformers, which is what
  their `utils.get_embedding` uses. (The paper says a mid-size PLM; the code
  says MiniLM. The code wins -- it is what produced their numbers.)

**Price is the one honest deviation.** These are open-weight models served
locally, so there is no per-token price sheet. `input_price`/`output_price` are
filled with the MEASURED mean GPU-seconds per candidate from the probe table,
which is this project's cost axis (problem doc §1.4). Keeping the field names
means their code is untouched; the units are documented here so nobody reads
them as dollars.
"""
from __future__ import annotations
import argparse, json, os, glob
from collections import defaultdict

# Descriptions follow the shape of theirs: size, what it is for, and the cost
# signal. The text is the node feature, so it should say what actually
# distinguishes the model -- family and scale, not marketing.
POOL = {
    "Qwen2.5-Coder (1.5b)": dict(
        short="qwen-1.5b",
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        feature=("A very small code-specialised model (1.5 billion parameters) from the "
                 "Qwen2.5-Coder family, pretrained heavily on source code. It is the "
                 "cheapest option in the pool and the fastest to serve, suited to "
                 "high-volume triage where most candidates are expected to be benign, "
                 "but it has limited capacity for multi-step reasoning over a dataflow "
                 "path.")),
    "Phi-4-mini (3.8b)": dict(
        short="phi-3.8b",
        model="microsoft/Phi-4-mini-instruct",
        feature=("A small general-purpose instruction model (3.8 billion parameters) "
                 "from Microsoft's Phi family, trained on heavily filtered synthetic and "
                 "textbook-style data. It reasons better than its size suggests but is "
                 "not code-specialised, so it relies on general inference rather than "
                 "learned code idioms.")),
    "DeepSeek-Coder (6.7b)": dict(
        short="deepseek-6.7b",
        model="deepseek-ai/deepseek-coder-6.7b-instruct",
        feature=("A mid-sized code model (6.7 billion parameters) trained on a large "
                 "corpus of permissively licensed repositories with fill-in-the-middle "
                 "objectives. It is the same scale as Qwen2.5-Coder-7B but a different "
                 "family, which makes the two separable when attributing behaviour to "
                 "scale rather than to a particular checkpoint.")),
    "Qwen2.5-Coder (7b)": dict(
        short="qwen-7b",
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        feature=("A mid-sized code-specialised model (7 billion parameters) from the "
                 "Qwen2.5-Coder family. It shares training recipe and tokenizer with the "
                 "1.5b member of this pool, so the pair isolates the effect of scale "
                 "within one family.")),
    "Granite-3.1 (8b)": dict(
        short="granite-8b",
        model="ibm-granite/granite-3.1-8b-instruct",
        feature=("The largest model in this pool (8 billion parameters), IBM's "
                 "general-purpose Granite instruction model with enterprise code and "
                 "security tuning. It is the most expensive to serve and is the natural "
                 "escalation target for candidates a smaller model cannot resolve.")),
}


def measured_cost(probe_dir: str, unit_set: str) -> dict[str, float]:
    """Mean GPU-seconds per candidate, per model, from the probe table."""
    out: dict[str, float] = {}
    for path in glob.glob(os.path.join(probe_dir, f"*__{unit_set}.jsonl")):
        short = os.path.basename(path).split("__")[0]
        tot = n = 0.0
        for line in open(path):
            if not line.strip():
                continue
            try:
                tot += json.loads(line)["seconds"]; n += 1
            except Exception:
                continue
        if n:
            out[short] = tot / n
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default=os.path.expanduser("~/probe_out"))
    ap.add_argument("--unit-set", default="units_slices")
    ap.add_argument("--out-json", default=os.path.expanduser(
        "~/routing/graphrouter/configs/LLM_Descriptions.json"))
    ap.add_argument("--out-pkl", default=os.path.expanduser(
        "~/routing/graphrouter/configs/llm_description_embedding.pkl"))
    a = ap.parse_args()

    costs = measured_cost(a.probe_dir, a.unit_set)
    if costs:
        print("measured mean GPU-seconds per candidate:")
        for k, v in sorted(costs.items(), key=lambda kv: kv[1]):
            print(f"  {k:15s} {v:.3f}s")
    else:
        print("no probe output yet -- writing descriptions with placeholder cost 0.0; "
              "re-run after B6 to fill in measured cost")

    desc = {}
    for name, spec in POOL.items():
        c = float(costs.get(spec["short"], 0.0))
        desc[name] = {
            "feature": spec["feature"],
            # UNITS: mean GPU-seconds per candidate on A100-PCIE-40GB, not dollars.
            "input_price": round(c, 6),
            "output_price": round(c, 6),
            "model": spec["model"],
            "short": spec["short"],
        }

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, "w") as fh:
        json.dump(desc, fh, indent=4)
    print(f"wrote {a.out_json} ({len(desc)} LLM nodes)")

    try:
        import pickle
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode([desc[k]["feature"] for k in desc])
        with open(a.out_pkl, "wb") as fh:
            pickle.dump(emb, fh)
        print(f"wrote {a.out_pkl} shape={emb.shape}")
    except Exception as e:
        print(f"embedding skipped ({e}); re-run once sentence-transformers is installed")
