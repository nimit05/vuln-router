#!/usr/bin/env python3
"""D1 -- probe a reasoning model over IRIS's dataflow paths. stdlib only.

GraphRouter v2, step 1: the positive control. Every model in the B6 pool sits at
chance on these paths (IRIS AUC 0.41-0.52, PROGRESS.md E1), so no objective can
route between them. This asks whether a model that can actually triage exists
at all -- Qwen3-32B, think and no-think -- before a router is built around it.

Same box, engine and prompt as the B6 pool (gpu0, vLLM 0.27.1, PROMPT_SLICE
imported from vulnrouter.probe, not copied), so the new node drops into the
same table. Two differences, both forced and both recorded per row:

* **Requests are concurrent.** Sequential batch-1 think mode is ~55 s a path,
  34 GPU-hours for the full set. `seconds` here is therefore NOT the batch-1
  cost the pool rows carry; `--workers 1` on a small calibration set gives the
  per-token rate that converts out_tokens into a comparable cost.
* **Think mode samples** at Qwen's recommended temperature 0.6 (greedy loops in
  think mode). One draw, as in E18.

p_vuln is read from the last true/false token after </think>, renormalised over
{true, false} using the top-5 logprobs. A row with no parsed verdict keeps
pred=None -- never silently "not vulnerable" (05-probe-defects.md, defect 1).
"""
import argparse, json, math, os, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.expanduser("~/nimit/graphrouter/src"))
from vulnrouter.probe import PROMPT_SLICE                     # noqa: E402

MODES = {
    "nothink": dict(enable_thinking=False, temperature=0.0, max_tokens=64),
    "think": dict(enable_thinking=True, temperature=0.6, top_p=0.95, top_k=20,
                  max_tokens=6144),
    # gpt-oss always writes an analysis channel before the final answer; "low"
    # effort keeps it short. Needs a budget for that analysis, and --plain.
    "low": dict(enable_thinking=False, temperature=0.0, max_tokens=2048,
                reasoning_effort="low"),
    # models that explain before the JSON despite the prompt (phi-4): same greedy
    # decode, room for the explanation; the verdict is read from the last JSON
    "verbose": dict(enable_thinking=False, temperature=0.0, max_tokens=1024),
}


def call(base, served, code, mode, plain=False):
    cfg = MODES[mode]
    body = {
        "model": served,
        "messages": [{"role": "user", "content": PROMPT_SLICE.format(code=code)}],
        "max_tokens": cfg["max_tokens"], "temperature": cfg["temperature"],
        "logprobs": True, "top_logprobs": 5,
    }
    if not plain:          # Qwen3 only; Mistral's template rejects the kwarg (HTTP 400)
        body["chat_template_kwargs"] = {"enable_thinking": cfg["enable_thinking"]}
    for k in ("top_p", "top_k", "reasoning_effort"):
        if k in cfg:
            body[k] = cfg[k]
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        resp = json.load(r)
    return resp, time.time() - t


def parse(resp):
    ch = resp["choices"][0]
    text = ch["message"].get("content") or ""
    lps = (ch.get("logprobs") or {}).get("content") or []
    toks = [x["token"] for x in lps]
    joined, end_think = "", 0
    for i, tk in enumerate(toks):
        joined += tk
        if "</think>" in joined:
            end_think = i + 1
            break
    answer = text.split("</think>")[-1]
    m = re.findall(r'"vulnerable"\s*:\s*(true|false)', answer)
    pred = None if not m else int(m[-1] == "true")
    p = None
    for x in reversed(lps[end_think:]):
        if x["token"].strip() in ("true", "false"):
            pt = pf = 0.0
            for alt in x.get("top_logprobs") or []:
                s = alt["token"].strip()
                if s == "true" and not pt:
                    pt = math.exp(alt["logprob"])
                if s == "false" and not pf:
                    pf = math.exp(alt["logprob"])
            if pt + pf > 0:
                p = pt / (pt + pf)
            break
    return dict(pred_label=pred, p_vuln=p, finish=ch.get("finish_reason"),
                in_tokens=resp["usage"]["prompt_tokens"],
                out_tokens=resp["usage"]["completion_tokens"],
                think_tokens=end_think, answer_tail=answer[-160:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--ids", default=None, help="JSON list of unit_ids to keep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=list(MODES), required=True)
    ap.add_argument("--model", default="qwen3-32b", help="short name for the table")
    ap.add_argument("--served", default="qwen3-32b")
    ap.add_argument("--base", default="http://127.0.0.1:18083/v1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plain", action="store_true",
                    help="no chat_template_kwargs (non-Qwen3 models)")
    a = ap.parse_args()

    units = [json.loads(l) for l in open(a.units) if l.strip()]
    if a.ids:
        keep = set(json.load(open(a.ids)))
        units = [u for u in units if u["unit_id"] in keep]
    if a.limit:
        units = units[: a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                done.add(json.loads(l)["unit_id"])
            except Exception:
                pass
    todo = [u for u in units if u["unit_id"] not in done]
    print(f"{a.model}/{a.mode}: {len(units)} units, {len(todo)} to do, "
          f"{a.workers} workers", flush=True)
    lock, n = threading.Lock(), [0]

    def work(u):
        row = dict(unit_id=u["unit_id"], model=f"{a.model}-{a.mode}", mode=a.mode,
                   project=u.get("project"), gold_label=int(u["gold_label"]),
                   workers=a.workers, engine="vllm", engine_version="0.27.1",
                   hardware="NVIDIA-A100-SXM4-80GB")
        try:
            resp, sec = call(a.base, a.served, u["code"], a.mode, a.plain)
            row.update(parse(resp), seconds=sec, error=None)
        except Exception as e:
            row.update(pred_label=None, p_vuln=None, error=repr(e)[:300])
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            n[0] += 1
            if n[0] % 25 == 0:
                print(f"  {n[0]}/{len(todo)} {time.strftime('%H:%M:%S')}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    print("done", flush=True)


if __name__ == "__main__":
    main()
