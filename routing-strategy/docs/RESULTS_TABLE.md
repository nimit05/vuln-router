# GraphRouter vs IRIS on CWE-Bench-Java

**16 projects, 2,259 dataflow paths, 344 true-positive paths.**
Metrics are IRIS Sec 3.6, computed by importing their own `score_subset.metrics`
rather than reimplementing it. Paper rows are the published per-project CSVs
restricted to these same 16 projects, so every row is like-for-like.

Our router replaces IRIS's posthoc LLM filter. Same input alerts, same ground
truth, same metrics; a configuration is a keep/drop decision over the same 2,259
paths. The path set is reconciled against IRIS's own evaluator project by
project (`b1_validate.py`, 16/16 exact on paths, true positives and recall).

## The table

| Configuration | #Det | rate% | AvgFDR% | AvgF1 |
|---|---|---|---|---|
| paper: CodeQL | 3/16 | 18.75 | 70.00 | 0.086 |
| paper: IRIS + DeepSeekCoder-7B | 10/16 | 62.50 | 81.31 | 0.215 |
| paper: IRIS + GPT-4 | 10/16 | 62.50 | 65.69 | 0.366 |
| ours: no filter (all paths) | 10/16 | 62.50 | 82.12 | 0.212 |
| ours: always-qwen-1.5b | 3/16 | 18.75 | 87.18 | 0.105 |
| ours: always-qwen-7b | 4/16 | 25.00 | 79.60 | 0.142 |
| ours: always-phi-3.8b | 8/16 | 50.00 | 82.97 | 0.202 |
| **ours: always-granite-8b** | 10/16 | 62.50 | 82.71 | **0.207** |
| **ours: GraphRouter (routed)** | 7/16 | 43.75 | 85.64 | **0.165** |
| ours: per-path oracle (upper bound) | 10/16 | 62.50 | 40.06 | **0.529** |

Model pool: Qwen2.5-Coder-1.5B-Instruct, Qwen2.5-Coder-7B-Instruct,
Phi-4-mini-instruct, granite-3.1-8b-instruct. vLLM 0.27.1, bf16, temperature 0,
one A100-SXM4-80GB, uncontended (verified by per-minute sampling of every process
on the card: 362 samples, none with a second process).

**deepseek-coder-6.7b-instruct was probed and then excluded as irreproducible.**
See the section below; this is the most important caveat on the table.

## Four things the table says

**1. Detection is saturated before any router runs.** Exactly 10 of 16 projects
contain a true-positive path at all, so 10/16 is a ceiling every configuration
shares, including both published IRIS rows. No filter can find an alert that is
not in its input. **The comparison is about false-discovery rate, not
#Detected** -- a reframing the corrected path set forced, and one the original
target table did not anticipate.

**2. Our best single model lands just short of the paper's row.**
always-granite-8b reaches AvgF1 0.207 against the published IRIS+DeepSeekCoder-7B
at 0.215 on the same 16 projects. Close, with an open-weight 8B model and no
tuning.

**3. The trained router underperforms the best single model** -- 0.165 against
0.207, and below even the no-filter baseline at 0.212. It is not undertraining:
20 epochs gave 0.177, 1,000 gave 0.165.

**4. The headroom is real and large.** The per-path oracle reaches AvgF1 0.529
and cuts AvgFDR from 82.12% to 40.06% while keeping full 10/16 detection, well
past the published GPT-4 row at 0.366. The models ARE complementary: on 91.2% of
paths the choice of model changes the verdict. What is missing is an objective
that can exploit it.

## Why the router loses

GraphRouter supervises with `label = eye(n)[argmax(effect)]`: "which model is
right about THIS path". IRIS's #Detected is per PROJECT and needs only one
true-positive path to survive in each. With 1,913 of 2,259 paths being false
alerts, those targets pull opposite ways.

| configuration | per-path accuracy | TP paths kept | keeps |
|---|---|---|---|
| always-granite-8b | 14.9% | 303/344 | 96.7% |
| always-phi-3.8b | 38.9% | 245/344 | 67.6% |
| GraphRouter (routed) | 58.3% | 113/344 | 36.4% |
| always-qwen-1.5b | 69.1% | 104/344 | 24.9% |
| always-qwen-7b | 70.2% | 66/344 | 20.4% |

Accuracy and true-positive retention invert. A filter that drops nearly
everything is right about most paths, because most paths are false.

| model | mean effect | on TP | on FP | wins argmax | router picks |
|---|---|---|---|---|---|
| granite-8b | 0.250 | **0.759** | 0.159 | **11.9%** | **0%** |
| phi-3.8b | 0.471 | 0.576 | 0.452 | 38.1% | 28.6% |
| qwen-7b | 0.447 | 0.588 | 0.421 | 31.6% | 54.0% |
| qwen-1.5b | 0.406 | 0.628 | 0.367 | 18.4% | 17.4% |

granite-8b is the **best** model on true vulnerabilities and the **worst** on
average, because the negative majority dominates the mean. It wins the training
label on 11.9% of paths and the router selects it **never** -- discarding the one
model whose behaviour the deployment metric rewards.

Full argument, with the epoch control, in `06-objective-misalignment.md`.

## deepseek-6.7b is excluded, and why that matters

An earlier draft of this table listed `always-deepseek-6.7b` at AvgF1 0.223 as
the best configuration, above the paper's own DeepSeekCoder-7B row. **That result
was not real and is withdrawn.**

The model was probed twice on identical units, identical prompt, identical vLLM
0.27.1, temperature 0, differing only in GPU: A100-SXM4-80GB and A100-PCIE-40GB.
On the 871 paths where both runs produced a parseable verdict:

| | |
|---|---|
| keep rate, SXM4-80GB | 0.693 |
| keep rate, PCIE-40GB | 0.413 |
| observed agreement | 0.451 |
| agreement expected by chance at those keep rates | 0.466 |
| **Cohen's kappa** | **-0.029** |

Cohen's kappa measures agreement beyond what the two keep rates would produce by
coincidence: 1 is perfect, 0 is none. At **-0.029** the two runs are
statistically independent. The same model, on the same inputs, under greedy
decoding, produces decisions on one card carrying no information about its
decisions on the other.

Its verdicts also carry almost no signal against ground truth: precision 0.187 on
one card and 0.250 on the other, against a base rate of 0.201 -- at or below
chance on one of them.

The other four models are stable across the same two cards:

| model | decisions identical across GPUs |
|---|---|
| qwen-1.5b | 100.0% |
| phi-3.8b | 100.0% |
| qwen-7b | 100.0% |
| granite-8b | 99.8% |

So this is specific to deepseek-6.7b on this task, not a general reproducibility
problem. The mechanism: it ignores the JSON instruction and emits long prose, and
the parser takes the last balanced object in the text, which is often incidental.
Small numerical differences between GPU architectures change the prose and flip
the verdict. On the PCIE card it produced no parseable verdict on 61% of paths
even at a 1024-token budget, with only 26 rows actually truncated -- the rest
simply never emitted a readable answer.

**Consequences.** The primary configuration is four models. The routed row is the
four-model router, because the five-model router selected deepseek on 66.9% of
paths and was therefore reading noise. A five-model table could be recovered by
giving deepseek a prompt it will follow, but changing the prompt for one model
and not the others breaks the comparability the rest of this table rests on.

## Reproducing

```
b0_iris_truth.py     IRIS's evaluator over the current SARIFs -> iris_truth.json
b1_extract_paths.py  one row per surviving codeFlow           -> paths.jsonl
b2_label_paths.py    TP at file:class:method                  -> paths_labelled.jsonl
b1_validate.py       THE GATE: must match iris_truth.json     -> exit 0
b3_slice.py          slice along the path                     -> slices.jsonl
b3b_make_units.py    probe units                              -> units_paths.jsonl
run_probe_gpu0.sh    models x 2,257 units, one card           -> probe_out/*.jsonl
finish_chain.sh      A3 -> b6b -> b7 (leave-one-project-out)  -> routes.json
c1_score_table.py    keep/drop -> IRIS metrics                -> this table
c2_objective_analysis.py                                      -> the diagnosis above
```

## Caveats

* **max_tokens is 256 for the three qwen/phi models and 1024 for granite-8b.**
  The three never exceeded 26 output tokens, so a cap they never reach cannot
  change their answers or their cost. granite was re-run at 1024 because it did
  touch the cap.
* **A non-answer is scored as "not vulnerable".** `probe.py` defaults an
  unparseable response to 0. That is what made deepseek's failure look like a
  confident, precise model rather than missing data. The default should become an
  explicit abstention; it has not been changed yet.
* **Cost is measured GPU-seconds**, min-max normalised per task inside
  GraphRouter, which discards the spread (0.109s to 0.861s per candidate across
  the four models) and keeps only the ordering.
* **Upstream is unmodified** apart from `split_data`, replaced with
  leave-one-project-out because their positional split leaks same-repo paths and
  assumes equal-size tasks (ours are 1,669 / 307 / 265 / 16).
* **14 folds, one seed.** Project-level splits over 16 projects are a small
  universe and the numbers move with the draw; a seed sweep is not yet run.
* **CWE is a function of project** here -- one CVE per project, one query per CWE
  -- so the CWE-node extension can only shift routing between projects.
