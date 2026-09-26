# Hybrid LLM reproduction — checkpoints

Reproducing the ICLR 2024 router: a two-class head over (route small, route
large), trained on a quality-gap target, thresholded at test time to trade
quality for cost without retraining.

Anchor: Ding, Mallick, Wang, Sim, Mukherjee, Ruhle, Lakshmanan & Awadallah,
*Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing*, **ICLR 2024**.
Upstream: `third_party/HybridLLM` (microsoft/best-route-llm) at
UPSTREAM_COMMIT.txt, verbatim.

Benchmark: **IRIS / CWE-Bench-Java**, the same 2,259 dataflow paths and the
same 16 projects the GraphRouter track used, so the two routers are comparable
row for row. Nothing in the candidate stream was rebuilt.

## Why this benchmark and not PrimeVul

The IRIS track already has the three things a router comparison needs and the
PrimeVul track does not yet: a gate that reconciles our candidate set against
the upstream evaluator (`b1_validate.py`, 16/16), a published table to sit
beside, and IRIS's own `metrics()` imported rather than reimplemented. Running
Hybrid LLM anywhere else first would have meant rebuilding all three.

## What is reused, unchanged

| Asset | Source | Why not rebuilt |
|---|---|---|
| 2,259 paths, 344 true positives | `b1`/`b2`, gate-passed | rebuilding risks re-opening a settled reconciliation |
| CPG slices + structural counts | `b3` | the router's only pre-call features |
| per (path, model) verdict, seconds, verdict logprob | `b6`, `data/gr/probe5` | the probe is the expensive step and it is already paid for |
| `metrics()` | `../IRIS/reproduction/score_subset.py` | imported, so arithmetic cannot drift |

**`data/gr/probe5` is the snapshot, not `probe_csecluster`.** On csecluster,
deepseek-6.7b has 1,376 of 2,257 rows with no parsed verdict at all, which
`probe.py` records as a confident *not vulnerable* — defect 1 of
`docs/05-probe-defects.md`. probe5 has 11. The cost columns on probe5 come from
a shared box and are therefore optimistic in absolute terms; they are used only
as ratios between models measured in the same run.

## Checkpoints

| # | Checkpoint | Artefact | Invariant | Status |
|---|---|---|---|---|
| H0 | Upstream vendored | `third_party/HybridLLM/` | tree byte-identical to UPSTREAM_COMMIT.txt | DONE |
| H1 | Label rules extracted | `hl_labels.py` | `det_2cls` and `prob_2cls` agree at n_samples=1 (self-check) | DONE |
| H2 | Pair table built | `data/hl/pairs__<s>__<l>__rho<r>.jsonl` | upstream's schema, every routable path present, no silent-zero verdict scored as confident | DONE — 2,257 |
| H3 | Router trained | `*.scores.json` | leave-one-project-out, vectoriser fitted on train only | DONE (linear backbone) |
| H4 | Frontier scored | `docs/RESULTS_HYBRIDLLM.md` | IRIS's own `metrics()`, all 16 projects in the denominator | DONE |
| H5 | DeBERTa backbone | — | upstream's `train_router.py` on the H2 folds | NOT STARTED — needs a booked GPU |
| H6 | Router v2: per-project rank features + small-model confidence | `h6_router_v2.py`, `*.scores.json` | only small-side signals in the matrix; ranks computed within a project | DONE |
| H7 | Project floor | `h7_floor.py`, `docs/RESULTS_FLOOR.md` | floor ranks by the small model's `p_vuln`, computable at inference | DONE |
| H8 | Controls: floor-only, matched-budget random | `h8_controls.py`, `docs/RESULTS_CONTROLS.md` | random control spends the SAME number of large calls, floor included | DONE |

## Two adaptations, both load-bearing

**1. Quality is the asymmetric security loss, not BARTScore.** There is no
reference answer for "is this path a real vulnerability", and a missed CVE and
a spurious alert are not the same event. `q = 1 - l/c_FN` puts a correct
verdict at 1, a missed vulnerability at 0 and a false alarm at `1 - 1/rho`.
At `rho = 1` this is plain accuracy and reproduces the paper's symmetric
setting exactly.

*Consequence worth stating:* quality then takes three values whose **order**
does not depend on rho, so `det_2cls` at `t = 0` is invariant to the
false-negative price. Upstream's tolerance `t` is the only channel through
which asymmetric cost reaches the objective at all.

**2. The quality distribution comes from the verdict logprob, not resampling.**
`prob_2cls` needs several quality samples per model. The probe recorded one
greedy verdict plus `decision_logprob`, so `p_vuln` is read off that token and
`--n-samples` deterministic Bernoulli draws are laid down. Cheaper than
re-running the pool, reproducible, and available for any open-weight model.
Rows with no parsed verdict are scored at `p = 0.5` and flagged, never as a
confident benign.

## Split discipline

Leave-one-project-out, for the reason in `graphrouter/b7_split_patch.py`. The
TF-IDF vectoriser is fitted on the training fold only; fitting it on all rows
leaks the held-out project's vocabulary, which is the same leak the split
exists to prevent.

## Order to run

```sh
PY=/opt/anaconda3/bin/python3      # needs sklearn, numpy, scipy

$PY hybridllm/h1_build_pairs.py --small qwen-1.5b --large granite-8b --rho 10
$PY hybridllm/h2_train_router.py \
      --pairs data/hl/pairs__qwen-1.5b__granite-8b__rho10.jsonl \
      --loss-type det_2cls --match-t 0.0
$PY hybridllm/h3_frontier.py \
      --scores data/hl/pairs__qwen-1.5b__granite-8b__rho10__det_2cls_t0.scores.json \
      --routes-at 0.5 --out docs/RESULTS_HYBRIDLLM_PAIR.md
$PY hybridllm/h4_sweep_pairs.py --out docs/RESULTS_HYBRIDLLM.md
```

`h3 --routes-at` writes a `{path_id: model}` file in the format
`graphrouter/c1_score_table.py --routes` already reads, so Hybrid LLM drops into
the GraphRouter table as one more row.

## H6–H8: what the second round changed, and what it did not

Design and full numbers in `docs/Hybrid_Router_plan.md`.

**H6.** Two candidate fixes for the LOPO-AUC-at-chance result of §3 of
`docs/07-hybrid-llm.md`: per-project rank normalisation of the structural
features, and small-model confidence signals (verdict logprob, token-logprob
spread, parse failures) read after a cascade's cheap call. Best single cell is
`struct+small+rank` at LOPO AUC **0.571** on qwen-7b → granite-8b against the
text backbone's 0.471 — but the ordering does not survive a change of backbone
(GBM prefers plain `struct`, 0.557), of pair, or of loss (`prob_2cls` prefers
text, 0.595). **Not a demonstrated fix.**

**H7.** The project floor sends the k riskiest paths per project to the large
model regardless of the router, which is the missing per-project term in a
per-path objective. With it, `τ=0.85, k=10` reaches AvgF1 **0.244 at 1,329.8
GPU-s and 10/16 detected**, against always-granite-8b's 0.207 at 1,943.6 and
10/16: better on quality, cost and detection simultaneously. First routed
configuration in this project to do that.

**H8, the caveat that has to travel with H7.** Against two controls:

* vs **matched-budget random** routing (same number of large calls, same
  floor): router ahead by +0.023 to +0.028 AvgF1, 2.8–5.0 sd over 25 draws.
  Real.
* vs **floor-only** (always-small except the floor): floor-only gets AvgF1
  0.237–0.241 at ~30% of always-granite's cost and loses only on #Detected
  (7–9/16). The router is behind it on AvgF1 in two of four cells.

So the floor, not the learned router, is carrying most of the quality. The
defensible claim is conditional: *where 10/16 detection must be preserved,
router + floor beats always-large on quality and cost.* τ and k were read off
the grid, so H9 (conformal calibration) is required before this is a result,
and the win does not replicate on qwen-1.5b → granite-8b.
