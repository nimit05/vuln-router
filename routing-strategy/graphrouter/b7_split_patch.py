#!/usr/bin/env python3
"""B7 -- patch GraphRouter's split, and NOTHING else.

Their `split_data()` (model/multi_task_graph_router.py) slices by ROW POSITION:

    query_per_task = num_query / num_task
    for task_id in range(num_task):
        start = task_id * query_per_task * num_llms
        train = rows [start, start + 0.7*query_per_task*num_llms)
        val   = the next 0.1, test = the next 0.2

Two things break on our data.

1. **It assumes every task has the same number of queries.** Theirs do; ours do
   not. Our tasks are CWEs and they hold 1,669 / 309 / 265 / 16 paths, so a
   single `query_per_task` is wrong for all four. Rows past the average are
   silently assigned to no split at all, and CWE-78 (16 paths) would be indexed
   as if it had 564.

2. **It splits inside a project.** Paths from one repo share sources, sinks and
   idioms, and ff4j alone contributes 1,413 of 2,257. A positional split puts
   near-duplicate paths on both sides, and any router scores well for the wrong
   reason. Protocol section 4 forbids it.

So the split becomes **leave-one-project-out**: hold out one project entirely,
train on the rest, rotate over all 14 projects that have paths. That also gives
a spread rather than one draw, which matters when 14 projects is the whole
universe.

**Everything else is left exactly as upstream wrote it** -- GeneralConv, two
layers, embedding_dim 8, the loss, the masking, and the three scenarios
(1.0*effect - 0.0*cost, 0.5/0.5, 0.2/0.8). The point of the comparison is their
model on our graph, so the model must not be quietly improved.

Applied by monkey-patching rather than editing the vendored file, so
`third_party/GraphRouter` stays a verbatim copy of upstream at
UPSTREAM_COMMIT.txt and the diff between their method and ours is this file.
"""
from __future__ import annotations
import json
import numpy as np
import torch

# Which project is held out on THIS fold. A module-level holder because upstream
# runs split_data from inside graph_router_prediction.__init__, so there is no
# call site to pass an argument through -- the runner sets this immediately
# before constructing the model for each fold.
CURRENT = {"held_out": None}


def make_leave_one_project_out(groups_path: str):
    """Return a `split_data` that holds out one project at a time.

    `groups_path` is `split_groups.json` from b6b_router_data.py: the project
    each query belongs to, in the row order the CSV was written in. Row order is
    what upstream indexes on, so the two files must be generated together."""
    groups = json.load(open(groups_path))
    projects = [q["project"] for q in groups["queries"]]
    order = groups["llm_order"]

    def split_data(self, held_out: str | None = None):
        n_llms = self.num_llms
        if n_llms != len(order):
            raise ValueError(
                f"split_groups.json was built for {len(order)} LLMs "
                f"({order}) but the router has {n_llms}")
        if len(projects) != self.num_query:
            raise ValueError(
                f"split_groups.json has {len(projects)} queries, the CSV has "
                f"{self.num_query}. Regenerate both from one b6b run.")

        held = held_out
        if held is None:
            held = CURRENT.get("held_out") or getattr(self, "held_out_project", None)
        if held is None:
            raise ValueError("no held-out project set; assign "
                             "router.held_out_project before split_data()")

        test_q = [i for i, p in enumerate(projects) if p == held]
        train_q = [i for i, p in enumerate(projects) if p != held]
        if not test_q:
            raise ValueError(f"no queries for held-out project {held}")

        # A validation slice is still needed for their early stopping, and it
        # must also be project-disjoint from training or it measures the same
        # leak the test split was rewritten to avoid.
        train_projects = sorted({projects[i] for i in train_q})
        rng = np.random.RandomState(self.config["seed"])
        val_proj = set(rng.permutation(train_projects)[:max(1, len(train_projects) // 7)])
        val_q = [i for i in train_q if projects[i] in val_proj]
        train_q = [i for i in train_q if projects[i] not in val_proj]

        def rows(qs):
            # each query owns num_llms contiguous rows, same as upstream
            return [q * n_llms + j for q in qs for j in range(n_llms)]

        self.train_idx = rows(train_q)
        self.validate_idx = rows(val_q)
        self.test_idx = rows(test_q)
        self.held_out_project = held
        self.query_per_task = None      # meaningless here; upstream only used it
                                        # to build the positional indices above

        # --- upstream's scenario weighting and label, copied verbatim so the
        # --- objective is unchanged. Only the index construction above differs.
        self.combined_edge = np.concatenate(
            (self.cost_list.reshape(-1, 1), self.effect_list.reshape(-1, 1)), axis=1)
        self.scenario = self.config["scenario"]
        if self.scenario == "Performance First":
            self.effect_list = 1.0 * self.effect_list - 0.0 * self.cost_list
        elif self.scenario == "Balance":
            self.effect_list = 0.5 * self.effect_list - 0.5 * self.cost_list
        else:
            self.effect_list = 0.2 * self.effect_list - 0.8 * self.cost_list
        effect_re = self.effect_list.reshape(-1, self.num_llms)
        self.label = np.eye(self.num_llms)[np.argmax(effect_re, axis=1)].reshape(-1, 1)

        # Edge index arrays and the three masks, also verbatim from upstream's
        # split_data. They are part of that method, so replacing it means
        # rebuilding them -- omitting them leaves mask_train undefined and
        # train_GNN fails, or worse, trains on a stale mask from a previous fold.
        self.edge_org_id = [num for num in range(self.num_query)
                            for _ in range(self.num_llms)]
        self.edge_des_id = list(range(self.edge_org_id[0],
                                      self.edge_org_id[0] + self.num_llms)) * self.num_query

        self.mask_train = torch.zeros(len(self.edge_org_id))
        self.mask_train[self.train_idx] = 1
        self.mask_validate = torch.zeros(len(self.edge_org_id))
        self.mask_validate[self.validate_idx] = 1
        self.mask_test = torch.zeros(len(self.edge_org_id))
        self.mask_test[self.test_idx] = 1

    return split_data, sorted(set(projects))


def apply(module, groups_path: str):
    """Swap the method in. Returns the project list to rotate over."""
    fn, projects = make_leave_one_project_out(groups_path)
    module.graph_router_prediction.split_data = fn
    return projects
