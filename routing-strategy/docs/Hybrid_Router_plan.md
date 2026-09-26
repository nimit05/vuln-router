# Hybrid Router — architecture and plan

A cost-aware router for vulnerability detection, built on Hybrid LLM
(Ding et al., ICLR 2024) and adapted to the two failures this project measured:
GraphRouter's objective is misaligned with a per-project recall metric
(`06-objective-misalignment.md`), and Hybrid LLM's router cannot predict its own
(good) target across repositories (`07-hybrid-llm.md` §3).

Status of each component is in §7. Benchmark, metrics and path set are
unchanged: IRIS / CWE-Bench-Java, 2,259 paths, 16 projects, IRIS's own
`score_subset.metrics`.

---

## 1. What this design has to fix

| Measured problem | Where | Fix here |
|---|---|---|
| Router supervised by `argmax` over 5 models discards the one model that keeps true positives | GraphRouter, AvgF1 0.165 vs 0.207 | Pairwise quality comparison, not argmax (§3) |
| Code text predicts the repository, not the path: LOPO AUC 0.494, leaky 0.865 | Hybrid LLM, all 10 pairs | Per-project rank features + small-model confidence (§4) |
| A non-answer is scored as a confident "not vulnerable" | `05-probe-defects.md` defect 1 | Abstention routes to large (§2, S2) |
| Routing per path ignores a per-project metric | every track | Project floor (§5) |
| `det_2cls` at t = 0 is invariant to the false-negative price ρ | `07-hybrid-llm.md` §5 | Continuous quality via CWE credit (§3) |
| τ picked by hand | — | Conformal risk control on missed vulnerabilities (§6) |

---

## 2. Inference flow

```
 Repository
     │
     ▼
 S0  Candidate generation ......... CodeQL + IRIS source/sink specs
     → units (dataflow paths)       (no static analysis → unit = function)
     │
     ▼
 S1  Feature extraction (no LLM) .. structural: n_hops, n_if, max_nest,
     → x_struct                     slice_loc, n_call, cross_file, …
     rank-normalise within project  → "long FOR THIS REPO", not "this repo"
     │
     ▼
 S2  Small model call ............. qwen-7b, JSON verdict + CWE
     → verdict, p_vuln, logprobs    unparseable ⇒ ABSTAIN, never "benign"
     │
     ▼
 S3  Router ....................... r(x) = P(small suffices | x)
     inputs = x_struct ⊕ small      linear head, soft target
     confidence signals
     │
     ├── r(x) ≥ τ  and not abstain ──► keep small's verdict
     │
     └── r(x) < τ  or abstain ──────► S4  Large model call (granite-8b, later 32B)
                                           │
     ┌─────────────────────────────────────┘
     ▼
 S5  Project floor ................ top-k units per project by small's p_vuln
     → re-route to large            protect per-project recall (#Detected)
     │
     ▼
 S6  Report ....................... kept paths + CWE + model used + GPU-seconds
```

Because the small model always runs, this is a **cascade**. Affordable only
because c_large / c_small is 3.4× here (granite-8b 1943.6 GPU-s vs qwen-7b
569.2 over 2,257 paths). The pre-call variant (router without S2 signals) is
kept as the ablation that says whether confidence is what makes routing
transfer.

---

## 3. Quality and label

**Detection quality**, ρ = c_FN / c_FP:

| outcome | q_det |
|---|---|
| correct verdict | 1 |
| missed vulnerability (FN) | 0 |
| false alarm (FP) | 1 − 1/ρ |

**CWE credit** (only when a true vulnerability is correctly flagged), over the
MITRE tree already parsed in `graphrouter/a1a2_cwe_nodes.py`:

| predicted CWE vs gold | q_cwe |
|---|---|
| exact | 1 |
| parent or child (view 1000) | 0.5 |
| grandparent or sibling | 0.25 |
| else | 0 |

```
q = q_det × (α + (1 − α)·q_cwe)     correctly detected vulnerability
q = q_det                           otherwise
```

α = 0.7 gives finding the bug 70% of the credit even with the wrong CWE.
Making q continuous is what lets ρ move the label at t = 0, which it cannot do
with three discrete quality values.

**Label** (upstream, unmodified in `hl_labels.py`): `y = 1` ("small suffices")
iff `q_small ≥ q_large − t`, with `t` the quality-gap tolerance.
`prob_2cls` relaxes the comparison over sampled quality.

---

## 4. Router inputs

Only small-side signals may enter. `pred_large`, `p_large`, `sec_large`,
`q_small`, `q_large` and `gold` are the answer or half of it.

| group | features |
|---|---|
| structural (S1) | n_hops, cross_file, n_methods, slice_loc, slice_chars, max_nest, n_if, n_loop, n_call, n_catch |
| small-model confidence (S2) | p_vuln, margin = \|p−0.5\|, verdict, decision logprob, out_tokens, retries, parse failures, token-logprob mean/min/std, fraction below −1, seconds |
| transform | per-project percentile rank (average ranks, ties shared) |

Rank normalisation is legitimate at inference: a scan has the whole
repository's candidate set before routing any of it. Ranks for a held-out
project are computed from that project's rows only, so nothing crosses the
fold boundary.

---

## 5. Project floor

IRIS's #Detected counts a project as detected if **one** true-positive path
survives. A per-path router has no term for that, so it can save cost by
dropping the only true path in a project. The floor sends the top-k paths per
project, ranked by the small model's p_vuln, to the large model regardless of
r(x). k is a second knob alongside τ.

---

## 6. Threshold calibration

Conformal risk control (Angelopoulos et al., ICLR 2024): on calibration
projects, choose the cheapest τ such that the router's extra missed
vulnerabilities over always-large stay ≤ ε with probability ≥ 1 − δ
(e.g. ε = 2%, δ = 10%). Report the price of the guarantee — how much of the
saving it costs.

---

## 7. Status

| # | Component | Artefact | Status |
|---|---|---|---|
| S0–S1 | paths, slices, structural counts | `data/gr/slices.jsonl` | DONE (gate `b1_validate.py`, 16/16) |
| S2 | probe: verdict + logprobs | `data/gr/probe5/` | DONE, CPU-reusable |
| S3 | router v2: rank + confidence features | `hybridllm/h6_router_v2.py` | DONE (H6) |
| S4 | large model | granite-8b | DONE; 32B swap pending, csecluster |
| S5 | project floor | `hybridllm/h7_floor.py` | DONE (H7) |
| — | controls: floor-only, matched random | `hybridllm/h8_controls.py` | DONE (H8) |
| S6 | frontier scoring | `hybridllm/h3_frontier.py` | DONE, reused unchanged |
| §3 | CWE credit in quality | `hl_labels.py` + re-probe | NOT STARTED — needs a CWE prompt (§9) |
| §6 | conformal τ | — | NOT STARTED |

Run order (CPU, local, minutes):

```sh
PY=/opt/anaconda3/bin/python3
$PY hybridllm/h6_router_v2.py --pairs data/hl/pairs__qwen-7b__granite-8b__rho10.jsonl \
      --loss-type det_2cls --match-t 0 --emit struct+small+rank
$PY hybridllm/h7_floor.py  --scores <the .scores.json h6 wrote> --out docs/RESULTS_FLOOR.md
$PY hybridllm/h8_controls.py --scores <same> --out docs/RESULTS_CONTROLS.md
```

---

## 8. Results so far (qwen-7b → granite-8b, ρ=10, det_2cls, t=0)

**The frontier.** First routed row in this project that beats the best single
model on both quality and cost:

| Configuration | %large | GPU-s | #Det | AvgFDR% | AvgF1 |
|---|---|---|---|---|---|
| always-qwen-7b | 0.0 | 569.2 | 4/16 | 79.60 | 0.142 |
| always-granite-8b | 100.0 | 1943.6 | 10/16 | 82.71 | **0.207** |
| **router v2, τ=0.90** | 59.7 | 1502.4 | 9/16 | 80.45 | **0.228** |
| **router v2, τ=0.95** | 79.4 | 1733.0 | 9/16 | 80.12 | **0.231** |
| label-following (oracle) | 16.6 | 798.8 | 10/16 | 57.08 | 0.405 |
| pair oracle (upper bound) | — | — | 10/16 | 50.98 | 0.445 |

τ=0.90 is +0.021 AvgF1 at 77% of always-large's compute. The earlier Hybrid LLM
sweep had **no** interior optimum at all — its best point was τ=1.00, which is
not routing. **But #Detected falls to 9/16**, which is what §5 exists to fix.

**With the project floor (S5), one cell dominates always-granite-8b on all
three axes at once** — quality, cost and detection:

| Configuration | forced | %large | GPU-s | #Det | AvgFDR% | AvgF1 |
|---|---|---|---|---|---|---|
| always-qwen-7b | — | 0.0 | 569.2 | 4/16 | 79.60 | 0.142 |
| always-granite-8b | — | 100.0 | 1943.6 | 10/16 | 82.71 | 0.207 |
| **router + floor, τ=0.85 k=10** | 115 | 47.9 | **1329.8** | **10/16** | 79.00 | **0.244** |
| router + floor, τ=0.70 k=10 | 115 | 26.3 | 1118.5 | 9/16 | 78.65 | 0.247 |
| floor-only (always-small) k=10 | 115 | 5.1 | 590.4 | 9/16 | 79.14 | 0.237 |
| floor-only (always-small) k=5 | 60 | 2.7 | 578.6 | 7/16 | 78.03 | 0.241 |

+0.037 AvgF1 over always-granite-8b at **68% of its GPU-seconds**, same 10/16
detection. Full grid in `RESULTS_FLOOR.md`.

**The controls say the floor earns most of it.** `RESULTS_CONTROLS.md`:

| cell | router+floor | floor-only | random, matched budget | router margin |
|---|---|---|---|---|
| τ=0.85 k=10 | 0.244 (10/16) | 0.237 (9/16) | 0.216 ± 0.006 (10/16) | +0.007 / **+0.028 (4.5 sd)** |
| τ=0.70 k=10 | 0.247 (9/16) | 0.237 (9/16) | 0.224 ± 0.006 (10/16) | +0.010 / **+0.023 (3.6 sd)** |
| τ=0.90 k=5 | 0.232 (10/16) | 0.241 (7/16) | 0.209 ± 0.008 (10/16) | **−0.010** / +0.023 (2.8 sd) |
| τ=0.95 k=5 | 0.234 (10/16) | 0.241 (7/16) | 0.206 ± 0.006 (10/16) | **−0.007** / +0.028 (5.0 sd) |

Read honestly, in three parts:

* The router **does** beat a matched-budget random router on every cell, by
  +0.023 to +0.028 AvgF1 at 2.8–5.0 standard deviations over 25 draws. Its
  picks are better than chance at the same spend.
* The router does **not** reliably beat floor-only, which reaches 0.237–0.241
  at **30% of always-granite's cost**. On two of four cells the router is
  behind it on AvgF1.
* What the router buys that the floor cannot is **detection**: floor-only tops
  out at 7–9 of 16 projects, and only router+floor reaches 10/16.

So the deployable claim is narrow and defensible: *at a budget where 10/16
detection must be preserved, the router + floor beats always-large on quality
and cost.* The broader claim — that the learned router is the source of the
gain — is not supported.

**It does not replicate on the cheaper pair.** qwen-1.5b → granite-8b, same
recipe: best cell is τ=0.85 k=0 at AvgF1 0.246 but only 7/16 detected, and **no
cell** beats always-granite on quality, cost and detection together.

**The AUC lift is real but not consistent.** LOPO AUC against the router's own
label:

| features | linear, qwen-7b→granite-8b | GBM, same pair | linear, qwen-1.5b→granite-8b |
|---|---|---|---|
| text (H2/H4 backbone) | 0.471 | — | 0.483 |
| struct | 0.464 | **0.557** | **0.533** |
| struct+rank | 0.546 | 0.553 | 0.519 |
| small only | 0.483 | 0.493 | 0.472 |
| struct+small | 0.504 | 0.550 | 0.521 |
| struct+small+rank | **0.571** | 0.537 | 0.517 |

Across the other eight pairs (linear, `det_2cls`), `struct+rank` ranges
0.468–0.572 and `struct+small+rank` 0.457–0.558, with neither config ahead of
plain `struct` more often than not.

Best config differs per backbone and per pair, and every value sits in
0.46–0.60. Honest reading: **these are fold-noise-sized differences on 14
projects, not a demonstrated fix.** Under `prob_2cls` the text backbone wins
instead (0.595). This is consistent with the control result above: the routing
gain is mostly the floor and the budget, not a router that has learned to tell
hard paths from easy ones.

---

## 9. Next steps, in order

1. **Stability** — all ten pairs × both losses × both backbones, seed sweep,
   per-fold AUC spread, and the floor grid on each. Decides whether §8's
   dominating cell is a result or a draw from noise. This is now the blocking
   item: one cell in one pair is not a finding.
2. **Conformal τ (§6)** — turns the frontier into a deployable guarantee, and
   removes the "τ and k chosen on the test set" objection that §8 currently
   carries. τ=0.85 and k=10 were selected by looking at the grid; a calibrated
   rule must pick them from held-out projects instead.
3. **Floor ablation** — k is currently tuned globally. Try a per-project k
   proportional to candidate count, and rank by something better than the small
   model's p_vuln.
4. **CWE credit (§3)** — needs a prompt that emits a CWE for every model.
   Current coverage in `probe5` is uneven (qwen-7b 459/2257, granite-8b
   2183/2257), so this needs a re-probe on csecluster. Note that on
   CWE-Bench-Java the CWE is a function of the project, so the CWE part must be
   **evaluated elsewhere** (PrimeVul vulnerable rows, or Juliet).
5. **32B large model** — the 3.4× cost gap here understates what a real
   cost-aware router saves. csecluster, TP=2, A100-PCIE-40GB.

## 10. Compute

Steps 1–3 are **CPU-only and local** — they replay `data/gr/probe5`, which is
already paid for, and take minutes on a laptop. No GPU was used for anything in
§8. Steps 4–5 need GPUs and go to **csecluster** via `sbatch` (standing
approval); gpu0/gpu7 stay untouched.
