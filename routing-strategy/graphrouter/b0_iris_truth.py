#!/usr/bin/env python3
"""B0 -- regenerate IRIS's per-project accounting by running IRIS's OWN evaluator.

Why this step exists. The cached `output/<proj>/<variant>/cwe-*wLLM-final/
results.json` files do NOT describe the SARIFs sitting next to them. Running
IRIS's `EvaluationPipeline.evaluate_sarif_result` over the current SARIFs gives
different, larger numbers on 6 of 16 projects (ff4j 1413 paths vs the cached
1229; ESAPI 198 vs 165). The cached JSON is an artefact of an earlier, smaller
query run that a later re-run never refreshed.

Using the cached numbers as ground truth would have been a silent error: our
extractor would have looked wrong when it was right, and the IRIS baseline rows
of the final table would have been computed off a different alert set than the
router rows. Every row must come from one snapshot of one set of files.

So: truth is recomputed here with THEIR code, on TODAY's files, and everything
downstream -- the reconciliation gate and the IRIS baseline rows -- reads this
output rather than the cached JSON. The cached values are carried along in
`stale_*` fields purely so the discrepancy stays documented.

Needs pandas + tqdm; on gpu0 use ~/miniconda3/envs/llmdfahf/bin/python.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iris", default=os.path.expanduser(
        "~/nimit/vuln-pred-results/IRIS"))
    ap.add_argument("--variant", default="v15_full17")
    ap.add_argument("--out", default=os.path.expanduser(
        "~/nimit/graphrouter/data/iris_truth.json"))
    a = ap.parse_args()

    sys.path.insert(0, a.iris)
    sys.path.insert(0, os.path.join(a.iris, "src"))
    import pandas as pd
    from modules.evaluation_pipeline import EvaluationPipeline

    bench = os.path.join(a.iris, "data", "cwe-bench-java", "data")
    fixed = pd.read_csv(os.path.join(bench, "fix_info.csv"))

    out = {}
    pat = os.path.join(a.iris, "output", "*", a.variant, "cwe-*wLLM-final",
                       "results.json")
    for f in sorted(glob.glob(pat)):
        proj = f.split(os.sep + "output" + os.sep)[1].split(os.sep)[0]
        cwe = re.search(r"cwe-(\d+)wLLM-final", f).group(1)
        base = os.path.join(a.iris, "output", proj, a.variant)
        sarif = os.path.join(base, f"cwe-{cwe}wLLM", "results.sarif")
        if not os.path.exists(sarif):
            print(f"  ! no SARIF for {proj}")
            continue
        ep = EvaluationPipeline(
            project_fixed_methods=fixed[fixed["project_slug"] == proj],
            class_locs_path=os.path.join(base, "fetch_class_locs", "results.csv"),
            func_locs_path=os.path.join(base, "fetch_func_locs", "results.csv"),
            project_source_code_dir=os.path.join(
                bench, "..", "project-sources", proj),
            test_run=False)
        r = ep.evaluate_sarif_result(sarif)
        cached = (json.load(open(f)).get("vanilla_result") or {})
        r["cwe"] = f"CWE-{int(cwe)}"
        r["stale_num_paths"] = cached.get("num_paths")
        r["stale_num_tp_paths_method"] = cached.get("num_tp_paths_method")
        r["stale_num_results"] = cached.get("num_results")
        out[proj] = r

    drift = [p for p, r in out.items() if r["num_paths"] != r["stale_num_paths"]]
    print(f"recomputed {len(out)} projects; "
          f"{len(drift)} differ from the cached results.json")
    print(f"{'project':46} {'paths now/cached':>20} {'tp now/cached':>16}")
    for p, r in sorted(out.items()):
        print(f"{p[:46]:46} {r['num_paths']:>9}/{str(r['stale_num_paths']):<10} "
              f"{r['num_tp_paths_method']:>7}/{str(r['stale_num_tp_paths_method']):<8}")
    tot = sum(r["num_paths"] for r in out.values())
    tot_tp = sum(r["num_tp_paths_method"] for r in out.values())
    det = sum(1 for r in out.values() if r["recall_method"])
    print(f"\ntotals: {tot} paths, {tot_tp} tp, detected {det}/{len(out)}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
