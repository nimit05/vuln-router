#!/usr/bin/env python3
"""Qwen3 think vs no-think probe over PrimeVul pairs. stdlib only.

One row per (unit, mode): verdict, p_vuln from the true/false token logprobs,
completion tokens, thinking tokens, latency. Resumable.
"""
import argparse, json, math, os, re, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = """You are a security auditor. Analyse the C/C++ function below.

Answer with a single JSON object and nothing else:
{{"vulnerable": true|false, "cwe": "CWE-###" or null}}

Function:
```c
{code}
```"""

PROMPT_COT = """You are a security auditor. Analyse the C/C++ function below.

Think step by step: what untrusted values enter, where they reach a memory
operation, index, size, or free, and whether a check makes it safe. Keep the
reasoning under 300 words. Then, on the last line, give a single JSON object:
{{"vulnerable": true|false, "cwe": "CWE-###" or null}}

Function:
```c
{code}
```"""

MODES = {
    "direct": dict(temperature=0.0, max_tokens=64),
    "cot": dict(temperature=0.0, max_tokens=1024, prompt="cot"),
    "nothink": dict(enable_thinking=False, temperature=0.0, max_tokens=64),
    "think": dict(enable_thinking=True, temperature=0.6, top_p=0.95, top_k=20, max_tokens=6144),
}
MAX_CODE_CHARS = 20000


def call(base, served, code, mode):
    cfg = MODES[mode]
    body = {
        "model": served,
        "messages": [{"role": "user", "content": (PROMPT_COT if cfg.get("prompt") == "cot" else PROMPT).format(code=code[:MAX_CODE_CHARS])}],
        "max_tokens": cfg["max_tokens"], "temperature": cfg["temperature"],
        "logprobs": True, "top_logprobs": 5,
    }
    if "enable_thinking" in cfg:
        body["chat_template_kwargs"] = {"enable_thinking": cfg["enable_thinking"]}
    for k in ("top_p", "top_k"):
        if k in cfg:
            body[k] = cfg[k]
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    return resp, time.time() - t


def parse(resp):
    ch = resp["choices"][0]
    text = ch["message"].get("content") or ""
    reasoning = ch["message"].get("reasoning_content") or ch["message"].get("reasoning") or ""
    lps = (ch.get("logprobs") or {}).get("content") or []
    toks = [x["token"] for x in lps]
    # locate end of thinking inside the token stream
    joined, end_think = "", 0
    for i, tk in enumerate(toks):
        joined += tk
        if "</think>" in joined and end_think == 0:
            end_think = i + 1
    answer = text.split("</think>")[-1]
    m = re.findall(r'"vulnerable"\s*:\s*(true|false)', answer)
    pred = None if not m else int(m[-1] == "true")
    # p_vuln: last true/false token after thinking; renormalise over {true,false}
    p = None
    for x in reversed(lps[end_think:]):
        if x["token"].strip() in ("true", "false"):
            pt = pf = None
            for alt in x.get("top_logprobs") or []:
                s = alt["token"].strip()
                if s == "true" and pt is None:
                    pt = math.exp(alt["logprob"])
                if s == "false" and pf is None:
                    pf = math.exp(alt["logprob"])
            pt, pf = pt or 0.0, pf or 0.0
            if pt + pf > 0:
                p = pt / (pt + pf)
            break
    return dict(pred=pred, p_vuln=p, finish=ch.get("finish_reason"),
                out_tokens=resp["usage"]["completion_tokens"], in_tokens=resp["usage"]["prompt_tokens"],
                think_tokens=end_think if end_think else (len(toks) if "</think>" not in answer and reasoning == "" and ch.get("finish_reason") == "length" else 0),
                answer_tail=answer[-200:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=list(MODES), required=True)
    ap.add_argument("--pairs", type=int, default=0, help="first N pair_ids (sorted); 0=all")
    ap.add_argument("--base", default="http://127.0.0.1:18080/v1")
    ap.add_argument("--served", default="qwen3-8b")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
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
    lock = threading.Lock()
    n = [0]

    def work(u):
        row = dict(unit_id=u["unit_id"], pair_id=u["pair_id"], gold_label=u["gold_label"],
                   gold_cwe=u.get("gold_cwe"), project=u.get("project"), mode=a.mode,
                   code_chars=len(u["code"]), truncated=len(u["code"]) > MAX_CODE_CHARS)
        try:
            resp, sec = call(a.base, a.served, u["code"], a.mode)
            row.update(parse(resp), seconds=sec, error=None)
        except Exception as e:
            row.update(pred=None, p_vuln=None, error=repr(e)[:300])
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            n[0] += 1
            if n[0] % 50 == 0:
                print(f"  {n[0]}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    print("done", flush=True)


if __name__ == "__main__":
    main()
