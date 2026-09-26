# The router loses to a single model, and the reason is the objective

Provisional: measured on four of five models. deepseek-6.7b was still re-running
when this was written, and the numbers will be refreshed with five.

## What the table shows

| Configuration | #Det | AvgFDR% | AvgF1 |
|---|---|---|---|
| no filter, all 2,259 paths | 10/16 | 82.12 | 0.212 |
| always-granite-8b | 10/16 | 82.71 | 0.207 |
| always-phi-3.8b | 8/16 | 82.97 | 0.202 |
| always-qwen-7b | 4/16 | 79.60 | 0.142 |
| always-qwen-1.5b | 3/16 | 87.18 | 0.105 |
| **GraphRouter (routed)** | **7/16** | **85.64** | **0.165** |
| per-path oracle (upper bound) | 10/16 | 40.06 | 0.529 |
| paper: IRIS + GPT-4 | 10/16 | 65.69 | 0.366 |

The routed row is below the best single model. The oracle is far above it. So the
information needed to route well is present in the model pool, and the trained
router is not extracting it.

## It is not undertraining

| epochs | routed AvgF1 |
|---|---|
| 20 | 0.177 |
| 1000 | 0.165 |

Fifty times the training made it slightly worse. Both runs used the same seed and
upstream's own loss, masking and scenario weighting.

## It is the objective

GraphRouter supervises with, per query,

```python
label = eye(n_llms)[argmax(effect)]
```

which asks "**which model is right about THIS path**". IRIS's `#Detected` is per
PROJECT and needs only **one** true-positive path to survive filtering in each
project. Those two targets diverge as soon as the base rate is skewed, and here
1,913 of 2,259 paths are false alerts.

### Evidence 1: accuracy and recall are anti-correlated

| configuration | per-path accuracy | TP paths kept | keeps |
|---|---|---|---|
| always-granite-8b | 14.9% | 303/344 | 96.7% |
| always-phi-3.8b | 38.9% | 245/344 | 67.6% |
| GraphRouter (routed) | 58.3% | 113/344 | 36.4% |
| always-qwen-1.5b | 69.1% | 104/344 | 24.9% |
| always-qwen-7b | 70.2% | 66/344 | 20.4% |

The ordering by accuracy is almost exactly the reverse of the ordering by
true-positive retention. A filter that drops nearly everything is right about
most paths, because most paths are false, and it destroys detection doing so.
Any objective that climbs the accuracy column walks down the retention column.

### Evidence 2: the argmax label discards the high-recall model

| model | mean effect | effect on TP | effect on FP | wins argmax |
|---|---|---|---|---|
| granite-8b | 0.250 | **0.759** | 0.159 | **11.9%** |
| phi-3.8b | 0.471 | 0.576 | 0.452 | 38.1% |
| qwen-7b | 0.447 | 0.588 | 0.421 | 31.6% |
| qwen-1.5b | 0.406 | 0.628 | 0.367 | 18.4% |

granite-8b is the **best** model on true vulnerabilities and the **worst** on
average, because the negative majority dominates the mean. It therefore wins the
training label on only 11.9% of paths, and the trained router selects it **zero**
times. The model whose behaviour the deployment metric rewards is the one the
objective throws away.

### The router did learn something

Split by ground truth, its choices are not random:

* on true vulnerabilities it picks the permissive phi-3.8b 43.0% of the time
* on false alerts it picks the aggressive qwen-7b 57.4% of the time

So it has learned a real signal. The signal is just far too weak to overcome the
pull of the objective: it retains 113 of 344 true-positive paths, and spread
across 16 projects that leaves several projects with none, which is why
`#Detected` falls from 10 to 7.

## Why this matters beyond this table

This is the same hazard that failed gate G1 on the PrimeVul track, one level up.
There, prediction bias in a model imitated skill. Here, the training objective is
itself fooled by the base rate: it cannot distinguish "drops everything" from
"discriminates well", because on an 85%-negative population those score alike.

Routing for vulnerability triage therefore needs an objective aligned with the
deployment metric -- recall-aware, and aggregated per project rather than per
path. GraphRouter's formulation is not that, and no amount of training fixes it.
That is a finding about the method, not a failure of the reproduction.

## What is NOT claimed

* Not that GraphRouter is wrong on its own benchmark. On balanced QA tasks with
  continuous quality metrics, per-path argmax is a reasonable target.
* Not that routing cannot help here. The oracle reaches AvgF1 0.529 against the
  best single model's 0.207, so the headroom is real and large.
* Not a tuned result. Upstream's hyperparameters were left untouched by design;
  only `split_data` was replaced, for the leakage reason in `b7_split_patch.py`.
