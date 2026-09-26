"""Probe runner: ask one model about every unit and emit the measurement table.

This is the critical path -- nothing else in the project can run until it has.
It is also the one step that is expensive to redo, so every signal any strategy
might need is logged on the first pass (docs/03-protocol.md §5). Retrofitting
logprobs or sample verdicts later means re-running every model.

Serving is left to the caller (an OpenAI-compatible endpoint, e.g. vLLM), so
this module has no inference dependency and can be unit-tested offline.
"""
from __future__ import annotations
import json, re, time
from dataclasses import dataclass, asdict
from typing import Callable, Iterable

PROMPT = """You are a security auditor. Analyse the C/C++ function below.

Answer with a single JSON object and nothing else:
{{"vulnerable": true|false, "cwe": "CWE-###" or null}}

Function:
```c
{code}
```"""

# S2 needs a second, decorrelated framing of the same question -- agreement
# across framings is a stronger difficulty signal than k samples of one framing
# (Yue et al., ICLR 2024, mixture-of-thought).
PROMPT_ALT = """Read the C/C++ function below and list which CWE, if any, it
contains. If it contains none, say CWE-NONE.

Answer with a single JSON object and nothing else:
{{"cwe": "CWE-###" or "CWE-NONE"}}

Function:
```c
{code}
```"""

PROMPT_COT = """You are a security auditor. Analyse the C/C++ function below.

Think step by step, briefly:
1. What untrusted or attacker-influenced values enter this function?
2. Where do they reach a memory operation, index, allocation size, or free?
3. Is there a check that makes that operation safe?

Keep the reasoning under 120 words and do NOT quote code in it. Then, on the
last line, give a single JSON object and nothing after it:
{{"vulnerable": true|false, "cwe": "CWE-###" or null}}

Function:
```c
{code}
```"""

PROMPT_SLICE = """You are triaging a static-analysis alert. Below is the alert
and the interprocedural dataflow slice it flagged -- the methods the tainted
value passes through, source first, sink last.

Decide whether this is a REAL vulnerability (the flagged flow is genuinely
exploitable) or a FALSE ALARM (the flow is guarded, sanitised, or not
attacker-controlled).

Answer with a single JSON object and nothing else:
{{"vulnerable": true|false, "cwe": "CWE-###" or null}}

```java
{code}
```"""

# Greedy from the first brace to the last would swallow any code the model
# quotes, so the verdict is read from the LAST balanced object in the text.
_JSON = re.compile(r"\{[^{}]*\}", re.S)


def parse(text: str) -> tuple[int | None, str | None, bool]:
    """-> (label, cwe, parse_failed). A parse failure is DATA, not an error:
    it is a real way a model fails and it costs real retries, so it is recorded
    and fed to gate G2 rather than silently dropped."""
    ms = _JSON.findall(text or "")
    if not ms:
        return None, None, True
    obj = None
    for cand in reversed(ms):           # the verdict is the LAST object emitted
        try:
            obj = json.loads(cand)
            break
        except json.JSONDecodeError:
            continue
    if obj is None:
        return None, None, True
    cwe = obj.get("cwe")
    if "vulnerable" in obj:
        return int(bool(obj["vulnerable"])), cwe, False
    if cwe is not None:
        return int(str(cwe).upper() != "CWE-NONE"), cwe, False
    return None, None, True


@dataclass
class ProbeRow:
    unit_id: str
    model: str
    commit: str
    cwe: str | None
    gold_label: int
    pred_label: int
    pred_cwe: str | None
    seconds: float
    in_tokens: int
    out_tokens: int
    n_retries: int
    parse_failures: int
    sample_verdicts: list[int]        # S2
    decision_logprob: float | None    # S3 -- the verdict token specifically
    token_logprobs: list[float] | None  # S3 -- the whole completion, for quantiles
    engine: str
    engine_version: str
    hardware: str


def run(units: Iterable[dict], model: str, complete: Callable,
        engine: str, engine_version: str, hardware: str,
        n_samples: int = 1, max_retries: int = 2,
        prompt: str | None = None, prompt_alt: str | None = None) -> list[dict]:
    """`complete(prompt) -> (text, in_tokens, out_tokens, logprobs)`.

    `logprobs` is None or {"decision": float|None, "tokens": [float, ...]}.
    Both are kept: S3's token-level rule needs the sequence, while the verdict
    token is the one the decision actually turns on and is worth isolating.

    Cost is wall-clock over the WHOLE unit including retries and samples, not a
    per-token price. That is the point of gate G2: a weaker model fails more,
    retries more, and therefore costs more -- so cost has to be measured here,
    not looked up from a price sheet afterwards.
    """
    out = []
    for u in units:
        t0 = time.perf_counter()
        # pred_cwe MUST be re-initialised per unit. It used to be read back
        # with locals().get(), which silently carried the PREVIOUS unit's CWE
        # onto any unit whose every attempt failed to parse.
        verdicts, fails, retries, tin, tout = [], 0, 0, 0, 0
        lp, pred_cwe = None, None
        for s in range(n_samples):
            tmpl = (prompt or PROMPT) if s % 2 == 0 else (prompt_alt or PROMPT_ALT)
            prompt_text = tmpl.format(code=u["code"])
            for attempt in range(max_retries + 1):
                text, i_tok, o_tok, dlp = complete(prompt_text)
                tin += i_tok; tout += o_tok
                label, cwe, failed = parse(text)
                if not failed:
                    verdicts.append(label)
                    if lp is None:
                        lp, pred_cwe = dlp, cwe
                    break
                fails += 1
                retries += attempt == max_retries and 0 or 1
        secs = time.perf_counter() - t0
        # abstain -> predict benign, the conservative choice for FP but the
        # expensive one for FN; it is recorded via parse_failures either way
        pred = int(round(sum(verdicts) / len(verdicts))) if verdicts else 0
        out.append(asdict(ProbeRow(
            unit_id=u["unit_id"], model=model, commit=u.get("commit", u["unit_id"]),
            cwe=u.get("cwe"), gold_label=int(u["gold_label"]), pred_label=pred,
            pred_cwe=pred_cwe, seconds=secs,
            in_tokens=tin, out_tokens=tout, n_retries=retries,
            parse_failures=fails, sample_verdicts=verdicts,
            decision_logprob=(lp or {}).get("decision"),
            token_logprobs=(lp or {}).get("tokens"), engine=engine,
            engine_version=engine_version, hardware=hardware)))
    return out
