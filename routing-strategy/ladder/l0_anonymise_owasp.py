#!/usr/bin/env python3
"""L0 -- OWASP units with the benchmark's identity hidden (memorisation control).

OWASP Benchmark has been public since 2015, so a model may recognise it rather
than read the code. This rewrites every string that names the benchmark:
BenchmarkTest00001 -> RequestHandler, org.owasp.benchmark.* -> com.example.app.*,
benchmark.properties -> app.properties. The real OWASP ESAPI library
(org.owasp.esapi) is kept, since production code uses it too. The code logic,
the CodeQL alert header and the labels are unchanged.

If a model's AUC falls a lot on these units, its skill was recognition, not
code reading (same check as d6_anonymise.py on IRIS).
"""
import argparse, json, re

SUBS = [
    (r"BenchmarkTest\d{5}", "RequestHandler"),
    (r"org\.owasp\.benchmark", "com.example.app"),
    (r"org/owasp/benchmark", "com/example/app"),
    (r"benchmarkprops", "appprops"),
    (r"benchmark\.properties", "app.properties"),
    (r"/benchmark/", "/app/"),
    (r"\b[Bb]enchmark\b", "app"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="data/owasp_ladder/units_owasp.jsonl")
    ap.add_argument("--out", default="data/owasp_ladder/units_owasp_anon.jsonl")
    a = ap.parse_args()
    n = left = 0
    with open(a.out, "w") as f:
        for l in open(a.units):
            u = json.loads(l)
            for pat, rep in SUBS:
                u["code"] = re.sub(pat, rep, u["code"])
            left += bool(re.search(r"owasp\.benchmark|BenchmarkTest|[Bb]enchmark", u["code"]))
            f.write(json.dumps(u) + "\n")
            n += 1
    print(f"{n} units written to {a.out}; {left} still mention the benchmark")


if __name__ == "__main__":
    main()
