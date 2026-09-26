# Three defects in the probe chain, and why two of them were dangerous

All three were found on 2026-09-10 while turning the corrected path set into the
comparison table. Two produce numbers that look reasonable and are wrong, which
is the failure mode worth writing down.

## 1. Truncation becomes a confident "not vulnerable"

`vulnrouter/probe.py` ends its retry loop with:

```python
pred = int(round(sum(verdicts) / len(verdicts))) if verdicts else 0
```

When every attempt fails to parse, `verdicts` is empty and the path is recorded
as **0 -- not vulnerable**. Nothing in the row says the model never answered;
`parse_failures` is set, but `pred_label` looks like any other decision.

At `max_tokens=256`, deepseek-6.7b hit this on **608 of 2,257 paths (26.9%)**.
Every one of those rows had exactly 768 output tokens, which is 3 attempts times
the 256 cap: the model writes a preamble and never reaches the verdict object.
granite-8b was truncated on 14.9% of attempts.

Why this is dangerous rather than merely wrong: **85% of paths are negatives**,
so a model that silently defaults to "not vulnerable" scores about 85-90%
accuracy. In the archived per-alert run deepseek showed 89.6% accuracy while
keeping only 8% of alerts, and that reads as a precise, decisive model. It was
mostly the default. A router trained to maximise accuracy would have learned to
pick it, producing a filter that drops nearly everything -- excellent
false-discovery rate, near-zero detection.

This is the same shape as the bias artefact that failed gate G1 on the PrimeVul
track. Prediction bias imitates skill whenever the base rate is lopsided.

**Fix applied:** re-ran the two affected models at `max_tokens=1024`. granite
came back with 0 truncated attempts and 0 missing verdicts, max 589 tokens.

**Why the other three were not re-run:** qwen-1.5b, qwen-7b and phi-3.8b never
exceeded **26** output tokens. A cap only changes a model that reaches it, so
their answers are identical at 256 and at 1024. Re-running them would have cost
an hour to reproduce byte-identical files. Stated here because "all models must
share one token budget" is a real constraint (their cost is measured seconds)
and this is the argument that the budget change does not break it.

**Still open:** the silent default should become an explicit abstention rather
than a 0, so a non-answer can never be mistaken for a judgement.

## 2. Embedding cells written in a format their own parser rejects

`prepare_data_for_GNN` reads each embedding cell as:

```python
inter = re.sub(r'\s+', ', ', inter.strip())
inter = json.loads(inter)[0]
```

It expects a **nested** list, and it turns any whitespace into `", "`. Upstream
wrote these cells with numpy's `str()`, which is whitespace-separated with no
commas, so the regex WAS the separator. `json.dumps` produces `"0.1, 0.2"`, and
the regex rewrites that to `"0.1, , 0.2"`, which does not parse.

Fixed on our side, in `b6b_router_data.py`, by writing
`json.dumps([vec], separators=(",", ":"))` -- nested, and with no whitespace, so
the regex is a no-op and both readers agree. `third_party/GraphRouter` stays
byte-identical to upstream.

## 3. Two dead imports block the training path

`data_processing/utils.py` imports `bert_score` and `litellm` at module level,
for a QA metric and for hosted-LLM calls. `multi_task_graph_router` imports that
module for four json/pickle helpers, so both are pulled in even though the
training path calls neither.

Stubbed in `b7_train_router.py` rather than installed, so the vendored tree stays
verbatim and reproducing the run does not require two large dependencies for
imports that are dead here. The stubs raise if anything ever does call them.

## Environment notes for gpu0

* vLLM 0.27.1 needs **Python 3.12**; it annotates `array.array[int]`, and on the
  3.11 env it dies with `TypeError: type 'array.array' is not subscriptable`.
  Built `~/nimit/vllmenv` at 3.12, pinned to the same 0.27.1 the cluster
  container runs so `engine_version` matches on both machines.
* The env's `bin/` must be on `PATH`, not merely used for absolute-path launching:
  vLLM's inductor backend shells out to `ninja`, and the EngineCore subprocess
  inherits `PATH`. Without it the model loads, captures CUDA graphs, allocates KV
  cache, and only then dies on `FileNotFoundError: 'ninja'`.
* `probe_client.py` resolves the project package as `<parent-of-script-dir>/src`,
  so it must sit one level below a directory containing `src/vulnrouter`.
* gpu0 is a **shared** box, so it does not give the exclusive GPU that
  `run_probe.sbatch` relies on for honest cost. `run_probe_gpu0.sh` samples every
  process on the card once a minute into `contention.log` so a suspect cost
  window can be checked rather than silently trusted.
