#!/usr/bin/env python3
"""Thinking-budget frontier + self-consistency probe. stdlib only.

Uses the raw /v1/completions endpoint with Qwen3 chat markers written by hand,
because budget forcing needs exact control over where thinking is cut off and
how the answer is then extracted:

  stage 1: <|im_start|>user ... <|im_start|>assistant\n<think>\n   -> up to B tokens
  stage 2: if no verdict emerged, re-prompt with the partial reasoning plus
           "</think>\n\nFinal answer" -> 48 tokens

That is s1-style budget forcing (Muennighoff et al., 2025): cap the reasoning,
then make the model commit. Cost is the sum of both stages, so a truncated unit
is charged for what it actually used.

Budget 0 is Qwen3's own no-think form (empty <think></think> block prefilled).
Mode sc:<k>@<B> draws k samples at budget B for a majority vote, and records
every sample so vote disagreement can be tested as a confidence signal.
"""
import argparse, json, math, os, re, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

QUESTION = """You are a security auditor. Analyse the C/C++ function below.

Answer with a single JSON object and nothing else:
{{"vulnerable": true|false, "cwe": "CWE-###" or null}}

Function:
```c
{code}
```"""

HEAD = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
MAX_CODE_CHARS = 20000
FORCE = "\n</think>\n\nFinal answer, JSON only: "
VERDICT = re.compile(r'"?vulnerable"?\s*[:=]\s*"?(true|false)"?', re.I)
FALLBACK = re.compile(r'\b(not vulnerable|no vulnerability|is vulnerable|vulnerable)\b', re.I)


def post(base, body, timeout=900):
    req = urllib.request.Request(base + "/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def verdict_of(text):
    m = VERDICT.findall(text)
    if m:
        return int(m[-1].lower() == "true")
    m = FALLBACK.findall(text)
    if m:
        last = m[-1].lower()
        return 0 if last.startswith(("not", "no ")) else 1
    return None


def p_true(choice):
    """p(vulnerable) from the true/false token, renormalised over {true,false}."""
    lp = choice.get("logprobs") or {}
    toks, tops = lp.get("tokens") or [], lp.get("top_logprobs") or []
    for i in range(len(toks) - 1, -1, -1):
        if toks[i].strip().lower() in ("true", "false"):
            d = tops[i] if i < len(tops) else None
            if not d:
                return None
            pt = pf = 0.0
            for k, v in d.items():
                ks = k.strip().lower()
                if ks == "true":
                    pt = max(pt, math.exp(v))
                elif ks == "false":
                    pf = max(pf, math.exp(v))
            return pt / (pt + pf) if pt + pf > 0 else None
    return None


def one_shot(base, served, code, budget, temp, seed=None):
    """One reasoning trace at the given budget, with forcing if it overruns."""
    q = QUESTION.format(code=code[:MAX_CODE_CHARS])
    head = HEAD.format(q=q)
    if budget == 0:                                  # Qwen3 no-think form
        prompt = head + "<think>\n\n</think>\n\n"
        body = dict(model=served, prompt=prompt, max_tokens=64, temperature=0.0,
                    logprobs=5)
        r = post(base, body)
        ch = r["choices"][0]
        return dict(pred=verdict_of(ch["text"]), p_vuln=p_true(ch),
                    tokens=r["usage"]["completion_tokens"], think_tokens=0,
                    forced=False, tail=ch["text"][-160:])
    prompt = head + "<think>\n"
    body = dict(model=served, prompt=prompt, max_tokens=budget, temperature=temp,
                top_p=0.95, logprobs=5)
    if seed is not None:
        body["seed"] = seed
    r = post(base, body)
    ch = r["choices"][0]
    text, used = ch["text"], r["usage"]["completion_tokens"]
    after = text.split("</think>")[-1] if "</think>" in text else ""
    pred = verdict_of(after) if after else None
    if pred is not None:
        return dict(pred=pred, p_vuln=p_true(ch), tokens=used,
                    think_tokens=used - len(after) // 4, forced=False, tail=after[-160:])
    # overran the budget: make it commit
    body2 = dict(model=served, prompt=prompt + text + FORCE, max_tokens=48,
                 temperature=0.0, logprobs=5)
    r2 = post(base, body2)
    ch2 = r2["choices"][0]
    return dict(pred=verdict_of(ch2["text"]), p_vuln=p_true(ch2),
                tokens=used + r2["usage"]["completion_tokens"], think_tokens=used,
                forced=True, tail=ch2["text"][-160:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True, help="budget:<N>  or  sc:<k>@<N>")
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--base", default="http://127.0.0.1:18080/v1")
    ap.add_argument("--served", default="qwen3-8b")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.6)
    a = ap.parse_args()

    if a.mode.startswith("budget:"):
        k, budget = 1, int(a.mode.split(":")[1])
    else:
        ks, bs = a.mode[3:].split("@")
        k, budget = int(ks), int(bs)

    units = [json.loads(l) for l in open(a.units)]
    if a.pairs:
        keep = set(sorted({u["pair_id"] for u in units})[: a.pairs])
        units = [u for u in units if u["pair_id"] in keep]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            done.add(json.loads(l)["unit_id"])
    todo = [u for u in units if u["unit_id"] not in done]
    print(f"{a.mode}: {len(units)} units, {len(todo)} to do", flush=True)
    lock, n = threading.Lock(), [0]

    def work(u):
        row = dict(unit_id=u["unit_id"], pair_id=u["pair_id"], gold_label=u["gold_label"],
                   project=u.get("project"), mode=a.mode, budget=budget, k=k,
                   code_chars=len(u["code"]))
        t0 = time.time()
        try:
            runs = [one_shot(a.base, a.served, u["code"], budget,
                             0.0 if k == 1 and budget == 0 else a.temp,
                             seed=(i if k > 1 else None)) for i in range(k)]
            preds = [r["pred"] for r in runs]
            votes = [p for p in preds if p is not None]
            row.update(
                pred=(1 if sum(votes) * 2 > len(votes) else 0) if votes else None,
                vote_frac=(sum(votes) / len(votes)) if votes else None,
                n_abstain=sum(p is None for p in preds),
                preds=preds,
                p_vuln=next((r["p_vuln"] for r in runs if r["p_vuln"] is not None), None),
                p_all=[r["p_vuln"] for r in runs],
                tokens=sum(r["tokens"] for r in runs),
                think_tokens=sum(r["think_tokens"] for r in runs),
                n_forced=sum(r["forced"] for r in runs),
                seconds=time.time() - t0, tail=runs[0]["tail"], error=None)
        except Exception as e:
            row.update(pred=None, p_vuln=None, tokens=0, error=repr(e)[:300],
                       seconds=time.time() - t0)
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            n[0] += 1
            if n[0] % 100 == 0:
                print(f"  {n[0]}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    print("done", flush=True)


if __name__ == "__main__":
    main()
