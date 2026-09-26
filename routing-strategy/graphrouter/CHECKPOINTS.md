# GraphRouter reproduction — checkpoints

Reproducing the Phase A / B / C architecture: a heterogeneous graph of CWE
nodes, candidate nodes (CodeQL alerts sliced with a CPG) and LLM nodes, trained
by edge prediction, used at inference to pick one model per candidate.

Anchor: Feng, Shen & You, *GraphRouter: A Graph-based Router for LLM Selections*,
**ICLR 2025**.

## Hardware assignment

| Work | Where | Why |
|---|---|---|
| Phase A (encoders), B1–B5, B7 (GNN training), Phase C | **gpu0** (`:8022`) | Untimed. A100-80GB, currently idle, has the CodeQL install, the 24 databases and the source trees, and (unlike the cluster) working outbound HTTPS for pip/HF |
| **B6 only** — probing every candidate against every LLM | **csecluster `gpu-A100`** | The one step whose wall-clock is a *measurement*. `--gres` gives an exclusive device; a shared GPU would record the neighbour's load as our cost |
| — | ~~gpu7~~ | Occupied: another tenant's vLLM holds 80.6/81.9 GB |

gpu0's home is shared with other users' directories, so all work lands in
`~/nimit/graphrouter/` and touches nothing else.

## Assets already on gpu0 (from the IRIS reproduction — not rebuilt)

* CodeQL **2.15.5** (the IRIS-patched release) at `IRIS/codeql/codeql`
* **24 CodeQL databases**, 7.5 GB, `IRIS/data/codeql-dbs/`
* CWE-Bench-Java sources, `project_info.csv` (CVE→CWE), `fix_info.csv` (ground truth)
* **4,531 LLM-augmented alerts** in `IRIS/output/*/v15_full17/cwe-*wLLM/results.sarif`,
  4,092 carrying a `codeFlows` dataflow path. Vanilla CodeQL yields only 29 —
  the augmented set is the usable candidate stream.

## Checkpoints

Each is a file on disk. A checkpoint is "done" when its artefact exists and the
stated invariant holds; nothing downstream starts before then.

| # | Checkpoint | Artefact | Invariant | Status |
|---|---|---|---|---|
| A1 | CWE hierarchy from MITRE | `cwe_parents.json` | every alert's CWE resolves to a chain ending at ROOT | DONE |
| A2 | CWE nodes encoded | `cwe_nodes.npz` | one vector per CWE in A1, from MITRE description text | DONE |
| A3 | LLM nodes encoded | `llm_nodes.npz` | one vector per pool model, description + price, same encoder | blocked on B6 costs |
| B0 | IRIS's own accounting, recomputed | `iris_truth.json` | produced by THEIR evaluator on TODAY's SARIFs, not the stale cached JSON | DONE |
| B1 | Paths extracted | `paths.jsonl` | one row per surviving codeFlow; CWE tag + path + sink on every row | DONE — 2,259 |
| B2 | Paths labelled | `paths_labelled.jsonl` | TP at `file:class:method`, IRIS's rule | DONE — 344 (15.23%) |
| **GATE** | **Reconciled with IRIS** | `b1_validate.py` exit 0 | per-project paths, TP and recall all match `iris_truth.json` | **PASSED 16/16** |
| B3 | CPG slices | `slices.jsonl` | every path keeps a row; unsliceable ones flagged, not dropped | DONE — 2,257/2,259, 0 TP lost |
| B4 | Candidate features | `candidates.npz` | code embedding + structural features, no NaNs | folded into B6b |
| B5 | Candidate to CWE edges | `edges_cwe.npz` | every candidate attached to a CWE node present in A2 | folded into B6b |
| B6 | Candidate to LLM edges | `probe_<model>.jsonl` | correctness + GPU-seconds per (path, model), one token budget, one card | rebuilt units staged; sweep swap pending |
| B7 | GNN trained | `router.pt` | beats the no-graph baseline on a held-out project split | not started |
| C1 | The table | `docs/RESULTS_TABLE.md` | every row from one snapshot, IRIS's own `metrics()` | harness written and smoke-tested |

## What the corrected set changed

The old candidate set counted SARIF results, not dataflow paths, and labelled by
fix-line range rather than enclosing method. Both are wrong against IRIS's own
scorer. Rebuilt: 98 true positives became 344, `rocketmq` left the pool because
IRIS never scored it, and the gate now passes on all 16 projects. Full account in
`docs/04-iris-reconciliation.md`.

The headline consequence: **all 16 projects, and every configuration, detect the
same 10.** Ten projects contain a true-positive path; six do not. Detection is
saturated before the router runs, so the comparison is entirely about
false-discovery rate — 82.12% unfiltered, 65.69% for the paper's GPT-4 row,
9.09% for a per-path oracle.

## Split discipline

Splits are **by project**, never by alert. Alerts from one repo share sources,
sinks and idioms; a random alert-level split puts near-duplicates on both sides
and every router scores well for the wrong reason (protocol §4).

## Known risks, recorded before starting

1. **Label skew.** Each project carries one CVE, so among thousands of alerts a
   handful are true positives. B2 reports the rate; if it is under ~1%, the
   edge attributes are nearly constant and B7 has nothing to learn.
2. **18 projects is a small split universe.** Project-level splits give roughly
   13 train / 5 test. Report over repeated splits with spread, not one draw.
3. **Alerts are concentrated.** Two projects hold 3,312 of 4,531 alerts, so a
   project split moves large blocks of data at once.
