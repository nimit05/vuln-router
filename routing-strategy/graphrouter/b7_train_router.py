#!/usr/bin/env python3
"""B7 -- train GraphRouter, leave one project out, and emit the routing decisions.

Runs UPSTREAM's model, loss, masking and scenarios unchanged (third_party/
GraphRouter at UPSTREAM_COMMIT.txt). The only substitution is `split_data`, for
the reasons in b7_split_patch.py: their split is positional and assumes tasks of
equal size, which both leaks same-repo paths across the split and mis-indexes
our wildly uneven CWE tasks (1,669 paths vs 16).

One fold per project. Each fold trains on every other project and routes only
the held-out project's paths, so no path is ever routed by a model that saw its
repo. The routes from all folds are unioned into one decision per path, which is
what `c1_score_table.py --routes` consumes.

Inference is done here rather than by calling their `test()`, because `test()`
returns only aggregate means -- we need the argmax per query, i.e. WHICH model
was chosen for each path. The forward pass below is theirs, called directly.

`wandb` is stubbed: upstream logs to it every epoch, and a research run should
not require a network account to reproduce.
"""
from __future__ import annotations
import argparse, json, os, sys


class _NoWandb:
    """Upstream calls wandb.log() each epoch. Nothing here needs a service."""
    def log(self, *a, **k):
        pass
    def init(self, *a, **k):
        pass
    def login(self, *a, **k):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gr-root", default="third_party/GraphRouter")
    ap.add_argument("--router-data", default="data/gr/router_data.csv")
    ap.add_argument("--groups", default="data/gr/split_groups.json")
    ap.add_argument("--llm-json", default="data/gr/LLM_Descriptions.json")
    ap.add_argument("--llm-emb", default="data/gr/llm_description_embedding.pkl")
    ap.add_argument("--scenario", default="Performance First",
                    choices=["Performance First", "Balance", "Cost First"])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1881)
    ap.add_argument("--out", default="data/gr/routes.json")
    ap.add_argument("--model-path", default="data/gr/router_fold.pth")
    a = ap.parse_args()

    root = os.path.abspath(a.gr_root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "model"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # multi_task_graph_router imports data_processing.utils for four json/pickle
    # helpers. That module imports bert_score and litellm at top level -- one for
    # a QA metric, one for calling hosted LLMs -- and the training path calls
    # neither. Stubbing them keeps third_party byte-identical to upstream instead
    # of installing two large dependencies to satisfy dead imports.
    import types
    for dead in ("bert_score", "litellm"):
        if dead in sys.modules:
            continue
        stub = types.ModuleType(dead)
        def _unavailable(*a, _n=dead, **k):
            raise RuntimeError(f"{_n} is stubbed: the training path does not "
                               f"use it. Install it if you need upstream's QA "
                               f"metric construction or their LLM calls.")
        stub.score = _unavailable          # bert_score.score
        stub.completion = _unavailable     # litellm.completion
        sys.modules[dead] = stub

    import torch
    import numpy as np
    import graph_nn
    import model.multi_task_graph_router as mtgr
    import b7_split_patch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    groups = json.load(open(a.groups))
    llm_order = groups["llm_order"]
    # split_groups records upstream's display names ("Phi-4-mini (3.8b)"), but
    # the probe table and c1_score_table key on the short name ("phi-3.8b").
    # Emit short names so the routes file joins without a translation step.
    desc = json.load(open(a.llm_json))
    short_of = {name: spec["short"] for name, spec in desc.items()}
    missing = [n for n in llm_order if n not in short_of]
    if missing:
        raise SystemExit(f"{a.llm_json} has no entry for {missing}; it must "
                         f"describe exactly the LLMs split_groups was built with")
    projects_of_query = [q["project"] for q in groups["queries"]]
    query_ids = [q["query_id"] for q in groups["queries"]]

    projects = b7_split_patch.apply(mtgr, a.groups)
    print(f"{len(query_ids)} queries, {len(llm_order)} LLMs, "
          f"{len(projects)} projects -> {len(projects)} folds")

    config = {
        "seed": a.seed,
        "model_path": a.model_path,
        "train_epoch": a.epochs,
        "scenario": a.scenario,
        "llm_num": len(llm_order),
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "train_mask_rate": 0.5,
        "batch_size": 32,
        "num_task": len({q["task"] for q in groups["queries"]}),
        "split_ratio": [0.7, 0.1, 0.2],     # unused by the patched split
        "embedding_dim": 8,
        "edge_dim": 3,
    }
    os.makedirs(os.path.dirname(a.model_path) or ".", exist_ok=True)

    routes: dict[str, str] = {}
    router = None
    for fold, held in enumerate(projects, 1):
        test_q = [i for i, p in enumerate(projects_of_query) if p == held]
        if not test_q:
            print(f"[{fold}/{len(projects)}] {held[:44]}: no queries, skipped")
            continue
        b7_split_patch.CURRENT["held_out"] = held
        print(f"[{fold}/{len(projects)}] hold out {held[:44]} "
              f"({len(test_q)} paths)", flush=True)

        if router is None:
            # First fold builds everything. Their __init__ reads the CSV and
            # json-parses 9,028 embedding cells, then splits and trains.
            router = mtgr.graph_router_prediction(
                router_data_path=a.router_data, llm_path=a.llm_json,
                llm_embedding_path=a.llm_emb, config=config, wandb=_NoWandb())
            # Effect and cost straight from the dataframe -- the same two lines
            # prepare_data_for_GNN uses. Taking them off the router instead
            # would capture values split_data has ALREADY weighted by the
            # scenario during this first fold.
            base_effect = np.array(router.data_df["effect"].tolist())
            base_cost = np.array(router.data_df["cost"].tolist())
        else:
            # Later folds reuse the parsed data and rebuild ONLY the model.
            # Re-reading the CSV per fold dominated runtime -- 14 folds of
            # re-parsing 9,028 x 384 floats -- and changes nothing, since the
            # data is identical across folds and only the split moves.
            #
            # The model IS rebuilt from scratch every fold. Carrying weights over
            # would let fold N start from a network that has already seen fold
            # N+1's project, which is exactly the leak leave-one-project-out
            # exists to prevent.
            router.set_seed(config["seed"])
            router.GNN_predict = graph_nn.GNN_prediction(
                query_feature_dim=router.query_dim,
                llm_feature_dim=router.llm_dim,
                hidden_features_size=config["embedding_dim"],
                in_edges_size=config["edge_dim"],
                wandb=_NoWandb(), config=config, device=device)
            # split_data mutates effect_list by the scenario weighting, so it
            # must start from the untouched values each fold or the weighting
            # compounds: 0.5*effect on fold 2 would be applied to fold 1's
            # already-weighted numbers.
            router.effect_list = base_effect.copy()
            router.cost_list = base_cost.copy()
            router.split_data()
            router.train_GNN()

        gp = router.GNN_predict
        data = router.data_for_test
        mask = torch.tensor(data.edge_mask, dtype=torch.bool)
        can_see = torch.logical_or(gp.valide_mask, gp.train_mask)
        gp.model.eval()
        with torch.no_grad():
            pred = gp.model(task_id=data.task_id, query_features=data.query_features,
                            llm_features=data.llm_features, edge_index=data.edge_index,
                            edge_mask=mask, edge_can_see=can_see,
                            edge_weight=data.combined_edge)
        choice = torch.argmax(pred.reshape(-1, len(llm_order)), 1).tolist()
        if len(choice) != len(test_q):
            raise RuntimeError(f"fold {held}: {len(choice)} predictions for "
                               f"{len(test_q)} held-out queries")
        for qi, c in zip(test_q, choice):
            routes[query_ids[qi]] = short_of[llm_order[c]]

    from collections import Counter
    print(f"\nrouted {len(routes)} paths")
    print("chosen model share:",
          {k: f"{v} ({100*v/max(len(routes),1):.1f}%)"
           for k, v in Counter(routes.values()).most_common()})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(routes, open(a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
