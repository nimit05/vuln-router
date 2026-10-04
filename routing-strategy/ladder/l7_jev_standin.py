#!/usr/bin/env python3
"""L7 -- a free, zero-shot stand-in for Jev (TypeSafe AI). stdlib only.

Jev-router gives Jev the query plus one capability description per model tier
(weakest to strongest) and reads back a probability for each tier; nothing is
learned from outcomes. Here an LLM we host (Qwen3-8B, no thinking) plays Jev: same
input, one multiple-choice question, and the probability of each option letter is
read from the first answer token's log-probabilities (renormalised over the options).

Two sets of descriptions for the professor's ladder:
  generic   what a normal Jev user writes: size and "how complex a case it handles"
  informed  the same, plus what each model scored on OWASP (the only data we have
            before seeing the target benchmark)

Output jsonl per unit: unit_id, variant, probs {tier: p}, seconds, in_tokens.
"""
import argparse, json, math, os, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

TIERS = ["qwen3-1.7b-nothink", "qwen3-8b-nothink", "qwen3-32b-nothink", "gpt-oss-120b-low"]
LETTERS = "ABCD"
DESCR = {
    "generic": [
        "Qwen3-1.7B, smallest and cheapest: simple, short code with obvious patterns.",
        "Qwen3-8B, cheap: moderate code with a standard, easy-to-follow data flow.",
        "Qwen3-32B, about 4x the cost of B: complex code with multi-step data flow across several methods.",
        "gpt-oss-120b, most expensive: very complex or subtle cases that need deep reasoning.",
    ],
    "informed": [
        "Qwen3-1.7B, cheapest: at chance on vulnerability triage in our tests; only for trivial cases.",
        "Qwen3-8B, cheap: moderate triage skill (70% accuracy on the OWASP Benchmark).",
        "Qwen3-32B, about 4x the cost of B: good triage skill (76% accuracy on OWASP).",
        "gpt-oss-120b, about 5x the cost of B: best on OWASP (90% accuracy).",
    ],
}
PROMPT = """You are a router. A static analyser flagged the code below as a possible security vulnerability. \
Choose which model should check whether the alert is real. Pick the cheapest model that is capable enough \
for this code.

Models, from cheapest and weakest to most expensive and strongest:
{options}

Code:
```
{code}
```

Answer with a single letter: {letters}."""


def ask(base, served, prompt):
    body = {"model": served, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
            "temperature": 0.0, "logprobs": True, "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    return resp, time.time() - t


def probs_of(resp):
    lp = (resp["choices"][0].get("logprobs") or {}).get("content") or []
    p = {c: 0.0 for c in LETTERS}
    for alt in (lp[0].get("top_logprobs") or []) if lp else []:
        s = alt["token"].strip().upper()
        if s in p and not p[s]:
            p[s] = math.exp(alt["logprob"])
    z = sum(p.values())
    return {TIERS[i]: (p[c] / z if z else 0.25) for i, c in enumerate(LETTERS)}, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--variant", choices=list(DESCR), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:18091/v1")
    ap.add_argument("--served", default="qwen3-8b")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    units = [json.loads(l) for l in open(a.units)]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["unit_id"] for l in open(a.out)}
    todo = [u for u in units if u["unit_id"] not in done]
    opts = "\n".join(f"{LETTERS[i]}: {d}" for i, d in enumerate(DESCR[a.variant]))
    lock, n = threading.Lock(), [0]
    print(f"{a.variant}: {len(todo)} of {len(units)} units", flush=True)

    def work(u):
        row = dict(unit_id=u["unit_id"], variant=a.variant)
        try:
            resp, sec = ask(a.base, a.served, PROMPT.format(options=opts, code=u["code"],
                                                            letters=", ".join(LETTERS)))
            probs, mass = probs_of(resp)
            row.update(probs=probs, option_mass=round(mass, 4), seconds=sec,
                       in_tokens=resp["usage"]["prompt_tokens"], error=None)
        except Exception as e:
            row.update(probs=None, error=repr(e)[:300])
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            n[0] += 1
            if n[0] % 500 == 0:
                print(f"  {n[0]}/{len(todo)} {time.strftime('%H:%M:%S')}", flush=True)

    t = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    print(f"done {a.variant} {len(todo)} units in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
