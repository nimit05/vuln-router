#!/usr/bin/env python3
"""Run vulnrouter.probe against a local vLLM OpenAI-compatible endpoint.

Deliberately sequential, batch size 1. Batching would raise throughput several
fold and destroy the measurement: `seconds` is the project's primary cost axis
(problem doc §1.4), and under batching a unit's wall-clock reflects whatever
else shared its batch. Per-unit cost has to be measured on an uncontended GPU,
one unit at a time, or gate G2 is regressing noise.

stdlib only -- the host venv on csecluster has no openai package and compute
nodes cannot reach PyPI.
"""
from __future__ import annotations
import argparse, json, os, sys, time, urllib.request, urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vulnrouter.probe import run, PROMPT_COT, PROMPT_SLICE         # noqa: E402

_VERDICT = {"true", "false", "True", "False"}


def make_complete(base_url: str, served: str, max_tokens: int, timeout: int):
    url = f"{base_url}/chat/completions"

    def complete(prompt: str):
        body = json.dumps({
            "model": served,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": 1,
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer EMPTY"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                obj = json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # A dead call is DATA: it costs real seconds and forces a retry,
            # which is exactly the endogenous-cost effect gate G2 tests for.
            return f"<request failed: {e}>", 0, 0, None

        ch = obj["choices"][0]
        text = ch["message"]["content"] or ""
        usage = obj.get("usage") or {}
        toks = [t.get("logprob") for t in
                ((ch.get("logprobs") or {}).get("content") or [])
                if t.get("logprob") is not None]
        decision = next((t["logprob"] for t in
                         ((ch.get("logprobs") or {}).get("content") or [])
                         if (t.get("token") or "").strip() in _VERDICT), None)
        return (text,
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
                {"decision": decision, "tokens": toks})

    return complete


def wait_healthy(base_url: str, minutes: int = 30) -> None:
    root = base_url.rsplit("/v1", 1)[0]
    for _ in range(minutes * 4):
        try:
            urllib.request.urlopen(f"{root}/health", timeout=5)
            return
        except Exception:
            time.sleep(15)
    raise SystemExit("vLLM never became healthy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="short name for the table")
    ap.add_argument("--served", required=True, help="--served-model-name")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--prompt-style", default="json", choices=["json", "cot", "slice"],
                    help="cot = reason briefly, then emit the JSON verdict")
    ap.add_argument("--resume", action="store_true",
                    help="skip unit_ids already in --out and append")
    ap.add_argument("--chunk", type=int, default=25,
                    help="units per flush; smaller = less lost to a kill")
    ap.add_argument("--engine", default="vllm")
    ap.add_argument("--engine-version", required=True)
    ap.add_argument("--hardware", required=True)
    a = ap.parse_args()

    units = [json.loads(l) for l in open(a.units) if l.strip()]
    if a.limit:
        units = units[:a.limit]

    # Resume. A 6k-unit pass is over an hour of GPU time; a node failure, a
    # preemption or a walltime overrun must not throw all of it away. Output is
    # appended per chunk to a STABLE path (not a per-job directory), so a
    # requeued job picks up exactly where the killed one stopped.
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if a.resume and out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)["unit_id"])
            except Exception:
                pass          # a torn last line from a kill mid-write
        units = [u for u in units if u["unit_id"] not in done]
        print(f"resume: {len(done)} units already probed, {len(units)} left",
              flush=True)
    print(f"{len(units)} units -> {a.model}", flush=True)

    wait_healthy(a.base_url)
    complete = make_complete(a.base_url, a.served, a.max_tokens, a.timeout)
    by_id = {u["unit_id"]: u for u in units}
    t0, rows = time.perf_counter(), []
    with open(out, "a" if a.resume else "w") as fh:
        for i in range(0, len(units), a.chunk):
            batch = run(units[i:i + a.chunk], a.model, complete,
                        engine=a.engine, engine_version=a.engine_version,
                        hardware=a.hardware, n_samples=a.n_samples,
                        max_retries=a.max_retries,
                        prompt={"cot": PROMPT_COT, "slice": PROMPT_SLICE}.get(a.prompt_style))
            for r in batch:
                u = by_id[r["unit_id"]]
                # carry the fields the table needs but probe.py does not model
                r.update(project=u.get("project"), pair_id=u.get("pair_id"),
                         split=u.get("split"), gold_cwe=u.get("gold_cwe"),
                         # ProbeRow.cwe reads units["cwe"]; sample_units.py names
                         # it gold_cwe to keep its fit-split-only status legible
                         cwe=u.get("gold_cwe"))
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            os.fsync(fh.fileno())          # NFS: flush alone can sit in cache
            rows += batch
            print(f"  {i + len(batch)}/{len(units)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    n = len(rows)
    acc = sum(r["pred_label"] == r["gold_label"] for r in rows) / max(n, 1)
    pf = sum(r["parse_failures"] > 0 for r in rows)
    secs = sum(r["seconds"] for r in rows)
    print(f"done {n} units in {time.perf_counter()-t0:.0f}s | "
          f"acc={acc:.3f} parse_failures_on={pf} ({pf/max(n,1):.1%}) | "
          f"gpu-seconds={secs:.0f} mean={secs/max(n,1):.2f}s", flush=True)
    print(f"wrote {a.out} ({n} new rows, {len(done)} resumed)", flush=True)
