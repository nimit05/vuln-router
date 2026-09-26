# Cost-Aware LLM Routing for Software Vulnerability Detection

A self-contained study of **routing strategies**: given a pool of LLMs with
different price/latency and different error profiles, decide *per code unit*
which model to ask, so as to maximise detection quality under a budget.

Every architectural decision is anchored to a paper at **ICLR / NeurIPS / ICML**
(or FOCS/TMLR where that is the canonical venue). Anchors are listed inline in
the docs and collected in [`docs/references.bib`](docs/references.bib), with the
venue of each verified against the published proceedings.

| Doc | Contents |
|---|---|
| [`docs/01-problem.md`](docs/01-problem.md) | task, benchmark, model pool, cost model, formal objective |
| [`docs/02-strategies.md`](docs/02-strategies.md) | the **strategy zoo** — 10 routing strategies, one anchor each |
| [`docs/03-protocol.md`](docs/03-protocol.md) | evaluation protocol, splits, metrics, failure modes to avoid |

## Layout

```
src/vulnrouter/
  data.py          benchmark loader (PrimeVul default, pluggable)
  features.py      pre-call features on a code unit -> unit_id-indexed frame
  probe.py         run one model over units -> the measurement table
  table.py         load the table; dense (unit x model) loss/cost matrices
  strategies.py    the zoo; every strategy is `route(table, lam) -> model per unit`
  metrics.py       F1, VD-Score, pair accuracy, asymmetric security loss
  evaluate.py      frontier, AIQ, gates G1-G3, bootstrap CIs, align()
configs/pool.yaml  the model pool under study
tests/smoke.py     end-to-end run on a synthetic table
data/              probe outputs (gitignored)
```

## Run the offline machinery now

```bash
../.venv/bin/python tests/smoke.py
```

Exercises every gate and both frontier paths on a synthetic table with known
structure, so the analysis code is verified before any GPU time is spent.

**One trap the library now guards.** `table.load()` pivots by `unit_id`, so row
order is the *sorted* unit order (`u0, u1, u10, u100`), not generation order.
Features must therefore be passed as a `unit_id`-indexed DataFrame and go through
`evaluate.align()`. Passing a positional array silently misaligns every feature
against every cost, and the symptom is a gate reporting "no signal" -- which was
caught exactly this way during the smoke test.

## Status

Design and offline machinery are written. **No measurements exist yet** — the
critical path is `probe.py` over the pool on csecluster, which produces the
table every strategy consumes. See `docs/03-protocol.md` §5.
