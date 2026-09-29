#!/usr/bin/env python3
"""O1 -- OWASP Benchmark Java alerts as triage units, in the IRIS unit format.

Second benchmark for the metric claim. Every one of the 2,740 test programs has a
known answer (expectedresults-1.2.csv: category, real vulnerability, CWE). CodeQL's
standard java-security-extended suite is run over the whole benchmark; each alert
that lands in a BenchmarkTest file AND carries that test's CWE becomes one unit:
"is this alert a real vulnerability?", labelled by the answer key.

Code shown to the model mirrors the IRIS units (PROMPT_SLICE): the CodeQL alert
header (rule + CWE + message), then the test's source with the same
"// file:line  method" marker. The 11 categories play the role of IRIS's projects
when scoring (within-category AUC, per-category metrics).

Output: units_owasp.jsonl with unit_id, code, gold_label, project (= category),
gold_cwe, test.
"""
import argparse, collections, csv, json, os, re

CATS = {"pathtraver": 22, "cmdi": 78, "xss": 79, "sqli": 89, "ldapi": 90,
        "crypto": 327, "hash": 328, "weakrand": 330, "trustbound": 501,
        "securecookie": 614, "xpathi": 643}


def rule_cwes(rule: dict) -> set:
    tags = (rule.get("properties") or {}).get("tags") or []
    out = set()
    for t in tags:
        m = re.match(r"external/cwe/cwe-0*(\d+)", t)
        if m:
            out.add(int(m.group(1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/nimit/owasp"))
    ap.add_argument("--out", default=os.path.expanduser("~/nimit/scale/data/units_owasp.jsonl"))
    a = ap.parse_args()
    bj = os.path.join(a.root, "BenchmarkJava")
    truth = {}
    for row in csv.reader(open(os.path.join(bj, "expectedresults-1.2.csv"))):
        if row and not row[0].startswith("#"):
            truth[row[0].strip()] = (row[1].strip(), row[2].strip() == "true", int(row[3]))
    sarif = json.load(open(os.path.join(a.root, "owasp.sarif")))
    run = sarif["runs"][0]
    rules = {}
    for comp in [run["tool"]["driver"]] + (run["tool"].get("extensions") or []):
        for r in comp.get("rules") or []:
            rules[r["id"]] = r
    src = {}
    def source(path):
        if path not in src:
            src[path] = open(os.path.join(bj, path), errors="ignore").read()
        return src[path]

    units, seen = [], set()
    stat = collections.Counter()
    for res in run["results"]:
        loc = res["locations"][0]["physicalLocation"]
        uri = loc["artifactLocation"]["uri"]
        m = re.search(r"(BenchmarkTest\d{5})\.java$", uri)
        if not m:
            stat["not_in_a_test"] += 1; continue
        test = m.group(1)
        if test not in truth:
            stat["test_not_in_key"] += 1; continue
        cat, real, cwe = truth[test]
        rule = rules.get(res.get("ruleId")) or {}
        if cwe not in rule_cwes(rule):
            stat["cwe_mismatch"] += 1; continue
        key = (test, res.get("ruleId"))
        if key in seen:                         # one unit per (test, rule)
            stat["duplicate"] += 1; continue
        seen.add(key)
        line = loc.get("region", {}).get("startLine", 0)
        msg = (res.get("message") or {}).get("text", "").strip()
        code = source(uri)
        body = code[code.find("public class"):] if "public class" in code else code
        text = (f"// CodeQL alert: {res.get('ruleId')}  (CWE-{cwe})\n// {msg}\n\n"
                f"// {uri}:{line}  doPost\n{body}")
        units.append(dict(unit_id=f"owasp:{test}:{res.get('ruleId')}", code=text,
                          gold_label=int(real), project=f"owasp-{cat}", gold_cwe=f"CWE-{cwe}",
                          test=test))
        stat["unit"] += 1
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for u in units:
            fh.write(json.dumps(u) + "\n")
    by = collections.Counter((u["project"], u["gold_label"]) for u in units)
    print(dict(stat))
    print(f"wrote {len(units)} units ({sum(u['gold_label'] for u in units)} real) -> {a.out}")
    for c in CATS:
        print(f"  {c:13} real {by[('owasp-' + c, 1)]:4}  false {by[('owasp-' + c, 0)]:4}")
    tests_flagged = len({u["test"] for u in units})
    print(f"tests with >=1 matching alert: {tests_flagged} of {len(truth)}")


if __name__ == "__main__":
    main()
