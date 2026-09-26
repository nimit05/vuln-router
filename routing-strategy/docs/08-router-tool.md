# `router.py` — the deployable rule, and the test that says whether to use it

Single self-contained script, no dependencies beyond the standard library.
`python3 router.py --help`.

## Commands

| command | what it does |
|---|---|
| `route` | assigns a model to each program. No LLM in the decision. |
| `preflight` | GO / NO-GO on a new corpus, before anyone builds a router. |
| `export` | LLMDFA `.out` logs -> the JSONL `preflight` reads. |
| `selftest` | reproduces the 12 published probe F1 values. |

## The rule

Route the branchiest fraction `q = 0.409` of the corpus to the **small** model,
the rest to the **large** one. `q` is fitted on the `float_*` DBZ corpus and
never re-fitted.

Relative rather than absolute (`n_if >= 12`) per §10 of `ROUTER_V2_PERCASE.md`:
the absolute cut is tied to one corpus's branchiness, and on family 2 the median
`n_if` drops 9 -> 3 and the fixed cut lands in the wrong place. The relative form
is never worse and lifts the weakest matched-count z from +2.4 to +3.0. Below
`--min-corpus` programs there is nothing to rank against, so it falls back to the
absolute cut.

```
python3 router.py route /path/to/java --small Phi-4-mini-instruct \
                                      --large Mistral-Nemo-Instruct-2407
```

## The pre-flight test

Two gates. **Gate 2 decides**; gate 1 is a screen.

**Gate 1 — do the models disperse on the metric you are routing for?**
A cheap screen, reported for precision, recall and F1. It is *advisory*: it
compares models **marginally**, but routing needs them to differ
**conditionally**. Two models equally good overall can still be good on
different cases -- which is exactly the routable case. So gate 1 only decides
when no pre-call feature was supplied to test.

| corpus | precision spread | gate 1 |
|---|---|---|
| XSS | 0.7 pts (97.5–98.1%) | FAIL |
| OSCI | 2.2 pts (84.0–86.1%) | FAIL |
| DBZ | 13.3 pts (37.6–51.0%) | PASS |

It is also confounded when models sit at different operating points: on PrimeVul
recall spans 3.8%–88.9%, so precision differences are just the precision/recall
trade-off, not skill. The tool prints a CONFOUNDED warning and defers to gate 2.

**Gate 2 — does a pre-call rule beat a random split of the same size?**
Apply the rule, then compare against 2,000 random splits sending the same
*number* of cases to the small model. A rule carrying no information scores what
an arbitrary split of that size scores.

```
python3 router.py export ../LLMDFA/reproduction/logs --bug dbz --out dbz.jsonl
python3 router.py preflight dbz.jsonl --bench ../LLMDFA/benchmark
```

## Gate 2 is NOT the shuffled-oracle test — correction to E17

`PROGRESS.md` E17 states step 2 as *"test against the shuffled-oracle null; if
the real oracle is at or below the shuffle, stop."* **That criterion rejects DBZ**,
the one corpus in either project where routing pays:

| corpus | real oracle | shuffled oracle | rule works? |
|---|---|---|---|
| DBZ (in-sample) | 0.796 | 0.916 | **yes, z +5.9** |
| off815 | 0.805 | 0.940 | **yes, z +5.5** |

§9 already recorded that the oracle sits below the shuffle *in every condition*,
off815 included. So a fake oracle gap does not imply a rule cannot work — the two
measure different things:

- the **shuffled oracle** asks whether *per-case model choice* has headroom;
- the **matched-count random split** asks whether *this particular rule* carries
  information.

The rule wins on DBZ while the oracle gap is fake, because the rule is not trying
to pick the best model per case — it is exploiting a *distribution-level*
crossover (the small model emits ~half the false alarms on branchy programs).

The shuffled oracle stays in the output as **INFO**: it says *do not quote the
oracle as headroom or as a target*. It does not gate.

## Reproduced numbers

`selftest` checks all 12 against `ROUTER_V2_PERCASE.md` at tol 0.002:

| probe | Phi | Nemo | router abs 12 | router relative |
|---|---|---|---|---|
| off815 | 0.575 | 0.639 | 0.699 | 0.699 (cut 13) |
| off1481 | 0.594 | 0.649 | 0.691 | 0.701 (cut 9) |
| fam2 | 0.640 | 0.698 | 0.714 | 0.723 (cut 9) |

`preflight` reproduces E17's headline: off815 **+9.8 precision points at +0.0
recall change**, z +5.5, beaten by 0.0% of 2,000 splits.

## Scope

Validated on LLMDFA divide-by-zero over Juliet Java, two held-out form families,
two inference engines. It does **not** transfer to IRIS (E14). Run `preflight`
before assuming it applies anywhere else.


---

## Cross-dataset testing (2026-09-20)

Run against every measurement set in the two projects. Eight defects found and
fixed; the four LLMDFA verdicts below reproduce `PROGRESS.md` E17 exactly.

| corpus | cases | models | gate 1 (precision) | gate 2 | verdict |
|---|---|---|---|---|---|
| XSS | 666 | 4 | 0.7 pts FAIL | z **−1.7** | NO-GO |
| OSCI | 444 | 4 | 2.2 pts FAIL | z **+0.5** | NO-GO |
| **DBZ** | 286 | 4 | 13.3 pts PASS | z **+5.9** | **GO** |
| **off815** | 111 | 4 | 6.5 pts PASS | z **+5.5** | **GO** |
| IRIS paths | 2,257 | 5 | 4.7 pts FAIL | z −5.8 (`in_tokens`) | NO-GO |
| PrimeVul | 5,820 | 5 | 9.0 pts, CONFOUNDED | z −16.8 (`in_tokens`) | NO-GO |

off815 reproduces E17's headline: **+9.8 precision points at +0.0 recall**,
beaten by 0.0% of 2,000 matched-count random splits. DBZ reproduces **+12.9
precision points** over the nearest-recall model. The reversed direction fails
on all four LLMDFA corpora, confirming §9's "reversing it loses every time".

### Defects found

| # | found on | defect |
|---|---|---|
| 1 | Juliet tree | `[a-g]$` stripping invented phantom programs (`AbstractTestCas`). Now only the Juliet `_\d+[a-g]` multi-part form collapses. |
| 2 | Juliet tree | `route` echoed its raw return value through `sys.exit`. |
| 3 | LLMDFA logs | `export` filtered on the log *filename*; the bug type lives in the header. |
| 4 | **PrimeVul** | A model with no positive predictions was reported at 0% precision rather than undefined, inflating the gate-1 spread **24.1 -> 8.9 pts** and turning a NO-GO into a PASS. |
| 5 | **PrimeVul** | Precision from a handful of predictions counted as real. `deepseek-6.7b` predicts positive 4 times in 5,820. Now `--min-predictions` (default 30). |
| 6 | **PrimeVul** | Precision spread confounded by operating point (recall 3.8%–88.9%). Now warns. |
| 7 | **IRIS** | Gate 2 needed Java source, so it could not run at all off the Juliet corpora. Now `--feature <numeric jsonl field>`. |
| 8 | **IRIS** | **Gate 2 permitted post-hoc direction selection.** On IRIS the stated direction gives z −5.8 and the reversal gives **z +7.8** -- a "GO" obtainable by trying both. Both directions are now always printed, with the warning that the direction must be fixed *before* the run. |
| 9 | synthetic | **Gate 1 as a hard stop rejects genuinely routable corpora.** A corpus where m1 is good on high `k` and m2 on low `k` has *identical* marginal precision and was being stopped -- yet gate 2 scores it z **+11.5**. Gate 1 is now advisory. |

Defect 9 is the one that matters for the writeup: it says the E17 pre-flight
test's step 1 cannot be a gate on its own, only a screen. Steps 1 and 2 are not
two hurdles of the same kind -- step 2 subsumes step 1 wherever a feature exists.

### Reproducing

```
python3 router.py export ../LLMDFA/reproduction/logs ../LLMDFA/reproduction/logs_router \
        --bug dbz --out dbz.jsonl
python3 router.py preflight dbz.jsonl --bench ../LLMDFA/benchmark
python3 router.py selftest
```
