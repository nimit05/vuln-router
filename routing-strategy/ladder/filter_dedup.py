#!/usr/bin/env python3
"""Filter stage on unique code slices of projects with a real path: many IRIS paths share the same slice (65,671 paths of the
Qwen3-8B spec run are 29,851 distinct slices), and the filter prompt depends only on the slice.

  make    one unit per distinct slice over all spec runs -> data/filter_units.jsonl
  expand  copy each slice's score back to every path of every run -> probe/<model>__<run>.jsonl,
          and split the dedup run's GPU time over the runs by their share of distinct slices
          (timing.jsonl rows with units_file = run, so l11 reads cost per path as before)
"""
import argparse, hashlib, json, os


def key(code):
    return "u_" + hashlib.sha1(code.encode()).hexdigest()[:20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["make", "expand"])
    ap.add_argument("--dir", default="/tmp/nimit/scaled")
    ap.add_argument("--runs", default="s_q8_full,s_q32_full")
    ap.add_argument("--models", default="qwen3-32b-nothink,qwen3-8b-nothink")
    ap.add_argument("--all", action="store_true", help="also score projects with no real path")
    a = ap.parse_args()
    runs = [r for r in a.runs.split(",") if os.path.exists(f"{a.dir}/data/{r}/units_paths.jsonl")]
    units = {r: [json.loads(l) for l in open(f"{a.dir}/data/{r}/units_paths.jsonl")] for r in runs}

    # A project whose spec run found no real path cannot change any result (that CVE is missed
    # whatever the ranking), so only projects with a real path are scored; the GPU time the rest
    # would cost is estimated from their size at the measured seconds per character.
    det = {r: {u["project"] for u in units[r] if int(u["gold_label"]) == 1} for r in runs}
    slices = {r: {key(u["code"]): len(u["code"]) for u in units[r]} for r in runs}

    if a.step == "make":
        seen = {}
        for r in runs:
            for u in units[r]:
                if a.all or u["project"] in det[r]:
                    seen.setdefault(key(u["code"]), u["code"])
        with open(f"{a.dir}/data/filter_units.jsonl", "w") as f:
            for k, code in seen.items():
                f.write(json.dumps({"unit_id": k, "project": "dedup", "gold_label": 0, "code": code}) + "\n")
        for r in runs:
            print(f"{r}: {len(units[r])} paths, {len(slices[r])} distinct slices, "
                  f"{len(det[r])} projects with a real path")
        print(f"to score: {len(seen)} distinct slices, {sum(map(len, seen.values())) / 1e6:.0f}M characters")
        return

    timing = {}
    tp = f"{a.dir}/probe/timing.jsonl"
    for l in open(tp):
        t = json.loads(l)
        if t["units_file"] == "dedup":
            timing[t["model"]] = t
    for m in a.models.split(","):
        rows = {}
        for l in open(f"{a.dir}/probe/{m}__dedup.jsonl"):
            x = json.loads(l)
            rows[x["unit_id"]] = x
        scored_chars = sum(c for r in runs for k, c in slices[r].items() if k in rows) or 1
        rate = timing[m]["wall_s"] / scored_chars if m in timing else None   # GPU-s per character
        for r in runs:
            miss = 0
            with open(f"{a.dir}/probe/{m}__{r}.jsonl", "w") as f:
                for u in units[r]:
                    x = rows.get(key(u["code"]))
                    if x is None:
                        miss += 1
                        x = {"pred_label": None, "p_vuln": None,
                             "error": "not scored (no real path in project)" if u["project"] not in det[r] else "not scored"}
                    y = dict(x, unit_id=u["unit_id"], project=u["project"], gold_label=int(u["gold_label"]))
                    f.write(json.dumps(y) + "\n")
            if rate is not None:     # every distinct slice of this run once, scored or estimated
                wall = rate * sum(slices[r].values())
                with open(tp, "a") as f:
                    f.write(json.dumps({"model": m, "units_file": r, "wall_s": round(wall, 1), "units": len(units[r]),
                                        "workers": 32, "dedup": True, "estimated_part": round(
                                            rate * sum(c for k, c in slices[r].items() if k not in rows), 1)}) + "\n")
            print(f"{m} {r}: {len(units[r])} paths written, {miss} without a score")


if __name__ == "__main__":
    main()
