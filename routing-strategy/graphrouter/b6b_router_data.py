#!/usr/bin/env python3
"""B6b -- convert our probe table into GraphRouter's router_data.csv.

Their loader expects one row per (query, LLM), grouped so that each query's
`num_llms` rows are contiguous, with columns:

    task_id, task_description, task_description_embedding,
    query, query_embedding, ground_truth, metric, llm, effect, cost

Three adaptations, each forced by the domain and each recorded:

1. **task_id is the CWE**, and task_description is the CWE's own text. That is
   the closest analogue of their "task" (GSM8K, SQUAD, ...), and it is what the
   CWE-node extension will later attach a hierarchy to.

2. **effect is graded, not binary.** Their effect is a continuous QA metric, so
   `label = eye(n)[argmax(effect)]` picks a clear winner. A raw 0/1 correctness
   would tie across models on most rows and `argmax` would break the tie by
   column index -- the router would learn "always pick the first-listed model"
   and score well doing it. So effect is the probability the model put on the
   CORRECT answer, recovered from the verdict-token logprob we logged:

       p_vuln = exp(lp) if predicted vulnerable else 1 - exp(lp)
       effect = p_vuln if gold == 1 else 1 - p_vuln

   Correct-and-confident scores near 1, wrong-and-confident near 0, and a
   coin-flip near 0.5. Rows with no logprob (parse failures) fall back to the
   hard 0/1, which is the honest value for an unparseable answer.

3. **cost is measured GPU-seconds**, min-max normalised within each task -- the
   same normalisation their construct_router_data.py applies.

Also writes `split_groups.json`: the project each query belongs to. Their
`split_data()` slices by ROW POSITION assuming equal-sized tasks, which would
put alerts from one repo on both sides of the split. Alerts within a repo share
sources and sinks, so that is a near-duplicate leak (protocol §4). The patched
splitter consumes this file instead.
"""
from __future__ import annotations
import argparse, json, math, os
from collections import defaultdict

# Canonical CWE text, used as the task description. cwe.load_parents_csv() reads
# MITRE's full export when the whole taxonomy is needed; these four are the
# classes IRIS's query pack actually fires on.
CWE_TEXT = {
    "CWE-94": ("Improper Control of Generation of Code ('Code Injection'). The product "
               "constructs all or part of a code segment using externally-influenced "
               "input, but does not neutralise special elements that could modify the "
               "syntax or behaviour of the intended code segment."),
    "CWE-22": ("Improper Limitation of a Pathname to a Restricted Directory ('Path "
               "Traversal'). The product uses external input to construct a pathname "
               "intended to identify a file beneath a restricted parent directory, but "
               "does not neutralise sequences such as '..' that can resolve outside it."),
    "CWE-79": ("Improper Neutralization of Input During Web Page Generation "
               "('Cross-site Scripting'). The product does not neutralise or incorrectly "
               "neutralises user-controllable input before it is placed in output used "
               "as a web page served to other users."),
    "CWE-78": ("Improper Neutralization of Special Elements used in an OS Command "
               "('OS Command Injection'). The product constructs an OS command using "
               "externally-influenced input without neutralising special elements that "
               "could modify the intended command."),
}
FALLBACK = "A weakness class from the CWE taxonomy."


def graded_effect(row: dict) -> float:
    """Probability the model placed on the correct verdict. See module docstring."""
    lp, pred, gold = row.get("decision_logprob"), row["pred_label"], row["gold_label"]
    if lp is None:
        return 1.0 if pred == gold else 0.0
    p_tok = math.exp(min(float(lp), 0.0))          # P(the verdict token it emitted)
    p_vuln = p_tok if pred == 1 else 1.0 - p_tok
    return p_vuln if gold == 1 else 1.0 - p_vuln


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default=os.path.expanduser("~/probe_out"))
    ap.add_argument("--units", default=os.path.expanduser(
        "~/data/units_v2/units_paths.jsonl"))
    ap.add_argument("--unit-set", default="units_paths")
    ap.add_argument("--llm-json", default=os.path.expanduser(
        "~/routing/graphrouter/configs/LLM_Descriptions.json"))
    ap.add_argument("--out", default=os.path.expanduser(
        "~/routing/graphrouter/data/router_data.csv"))
    ap.add_argument("--groups-out", default=os.path.expanduser(
        "~/routing/graphrouter/data/split_groups.json"))
    ap.add_argument("--no-embed", action="store_true",
                    help="skip query embedding (schema smoke-test only)")
    a = ap.parse_args()

    units = {}
    for line in open(a.units):
        if line.strip():
            u = json.loads(line)
            units[u["unit_id"]] = u

    llm_desc = json.load(open(a.llm_json))
    # column order == the order LLM nodes appear in llm_description_embedding.pkl
    llm_order = [(name, spec["short"]) for name, spec in llm_desc.items()]

    probes: dict[str, dict[str, dict]] = {}
    for _, short in llm_order:
        path = os.path.join(a.probe_dir, f"{short}__{a.unit_set}.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"missing probe output for {short}: {path}")
        probes[short] = {}
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                probes[short][r["unit_id"]] = r

    # keep only candidates every model measured -- a partially measured query
    # makes the argmax over LLMs undefined
    common = set(units)
    for short in probes:
        common &= set(probes[short])
    dropped = len(units) - len(common)
    if dropped:
        print(f"dropped {dropped} candidates not measured by every model")

    # group by task so each task's rows are contiguous, as their loader assumes
    by_task: dict[str, list[str]] = defaultdict(list)
    for uid in sorted(common):
        by_task[units[uid].get("query_cwe") or "CWE-UNKNOWN"].append(uid)
    print("candidates per task:",
          {k: len(v) for k, v in sorted(by_task.items(), key=lambda kv: -len(kv[1]))})

    queries, rows = [], []
    for task in sorted(by_task, key=lambda t: -len(by_task[t])):
        for uid in by_task[task]:
            u = units[uid]
            queries.append(dict(unit_id=uid, task=task, project=u["project"],
                                code=u["code"]))
            for name, short in llm_order:
                p = probes[short][uid]
                rows.append(dict(
                    task_id=task,
                    task_description=CWE_TEXT.get(task, FALLBACK),
                    query_id=uid,
                    project=u["project"],
                    ground_truth=int(u["gold_label"]),
                    metric="graded_correctness",
                    llm=name,
                    effect=graded_effect(p),
                    cost=float(p["seconds"]),
                ))

    # cost normalised WITHIN each task, matching construct_router_data.py
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["task_id"]].append(r["cost"])
    lohi = {t: (min(v), max(v)) for t, v in by_t.items()}
    for r in rows:
        lo, hi = lohi[r["task_id"]]
        r["cost"] = 0.0 if hi <= lo else (r["cost"] - lo) / (hi - lo)

    # query embeddings: their encoder, all-MiniLM-L6-v2
    if a.no_embed:
        embs = [[] for _ in queries]
    else:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"embedding {len(queries)} slices ...")
        embs = enc.encode([q["code"] for q in queries],
                          batch_size=32, show_progress_bar=False).tolist()
        tdesc = sorted({r["task_id"] for r in rows})
        temb = dict(zip(tdesc, enc.encode([CWE_TEXT.get(t, FALLBACK)
                                           for t in tdesc]).tolist()))

    emb_of = {q["unit_id"]: embs[i] for i, q in enumerate(queries)}

    def emb_cell(vec) -> str:
        r"""Serialise an embedding the way THEIR parser reads it back.

        prepare_data_for_GNN does, per cell:
            inter = re.sub(r'\s+', ', ', inter.strip()); json.loads(inter)[0]

        Two consequences. It wants a NESTED list, because it takes [0]. And any
        whitespace becomes ", ": json.dumps' default "0.1, 0.2" turns into
        "0.1, , 0.2" and fails to parse. Their own writer emitted numpy's
        str(), which is whitespace-separated with no commas, so the regex was
        the separator. Writing with no whitespace at all makes the regex a
        no-op and parses correctly under both readers."""
        return json.dumps([list(vec)], separators=(",", ":"))

    import csv
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "task_id", "task_description", "task_description_embedding",
            "query", "query_embedding", "ground_truth", "metric", "llm",
            "effect", "cost", "query_id", "project"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "task_id": r["task_id"],
                "task_description": r["task_description"],
                "task_description_embedding": emb_cell(
                    temb[r["task_id"]] if not a.no_embed else []),
                "query": r["query_id"],
                "query_embedding": emb_cell(emb_of[r["query_id"]]),
                "ground_truth": r["ground_truth"],
                "metric": r["metric"],
                "llm": r["llm"],
                "effect": round(r["effect"], 6),
                "cost": round(r["cost"], 6),
                "query_id": r["query_id"],
                "project": r["project"],
            })

    with open(a.groups_out, "w") as fh:
        json.dump(dict(
            num_llms=len(llm_order),
            llm_order=[n for n, _ in llm_order],
            queries=[dict(query_id=q["unit_id"], task=q["task"],
                          project=q["project"]) for q in queries]), fh)

    n_eff = [r["effect"] for r in rows]
    print(f"wrote {a.out}: {len(rows)} rows = {len(queries)} queries x {len(llm_order)} LLMs")
    print(f"  effect: min={min(n_eff):.3f} mean={sum(n_eff)/len(n_eff):.3f} max={max(n_eff):.3f}")

    # --- can routing change anything at all? Three views, weakest to strongest.
    k = len(llm_order)
    ties = sum(1 for i in range(0, len(rows), k)
               if len({round(rows[i + j]["effect"], 6) for j in range(k)}) == 1)
    print(f"  queries where all models tie on graded effect: {ties} "
          f"({ties/max(len(queries),1):.1%})")
    print("    ^ graded effect is continuous, so this is near 0 by construction "
          "and is NOT the test of whether the router has anything to learn.")

    # The real test. A path only matters to the router if swapping the model
    # swaps the KEEP/DROP verdict -- that is the only thing the final table sees.
    shorts = [s for _, s in llm_order]
    unanimous = correct_unanimous = wrong_unanimous = 0
    for q in queries:
        verdicts = {probes[s][q["unit_id"]]["pred_label"] for s in shorts}
        if len(verdicts) == 1:
            unanimous += 1
            v = next(iter(verdicts))
            gold = int(units[q["unit_id"]]["gold_label"])
            if v == gold:
                correct_unanimous += 1
            else:
                wrong_unanimous += 1
    live = len(queries) - unanimous
    print(f"  paths where all {k} models give the SAME verdict: {unanimous} "
          f"({unanimous/max(len(queries),1):.1%})")
    print(f"    of those, all-correct {correct_unanimous}, all-wrong {wrong_unanimous}")
    print(f"  paths the router can actually change: {live} "
          f"({live/max(len(queries),1):.1%})")
    if live < 0.05 * len(queries):
        print("  ** The models disagree on under 5% of paths. The router has "
              "almost nothing to route. Report this as the finding rather than "
              "training on it. **")
    print(f"wrote {a.groups_out}")


if __name__ == "__main__":
    main()
