#!/usr/bin/env python3
"""SAST Filter Agent: LLM-based contextual review of Semgrep findings, following
the QASecClaw paper's design (batches of 15 files, CWE-aware prompt, structured
JSON verdicts, conservative fail-open on malformed/failed responses).
"""
import argparse
import csv
import json
import re
import time
import urllib.request
from pathlib import Path

CWE_NAMES = {
    22: 'Path Traversal', 78: 'Command Injection', 79: 'Cross-Site Scripting (XSS)',
    89: 'SQL Injection', 90: 'LDAP Injection', 327: 'Weak Cryptography',
    328: 'Weak Hashing', 330: 'Weak Randomness', 501: 'Trust Boundary Violation',
    614: 'Insecure Cookie', 643: 'XPath Injection',
}

CWE_SYNONYMS = {326: 327}


def load_ground_truth(csv_path):
    gt = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            name, category, is_vuln, cwe = row[0], row[1], row[2], row[3]
            gt[name] = {'category': category, 'vulnerable': is_vuln.strip().lower() == 'true', 'cwe': int(cwe.strip())}
    return gt


def extract_test_name(path):
    m = re.search(r'(BenchmarkTest\d+)', path)
    return m.group(1) if m else None


def extract_cwes(finding):
    cwes = set()
    for c in finding.get('extra', {}).get('metadata', {}).get('cwe', []):
        m = re.search(r'CWE-(\d+)', c)
        if m:
            num = int(m.group(1))
            cwes.add(num)
            if num in CWE_SYNONYMS:
                cwes.add(CWE_SYNONYMS[num])
    return cwes


def build_review_set(gt, semgrep_json_path):
    """Files where Semgrep flagged the file's own ground-truth CWE (TP+FP)."""
    data = json.load(open(semgrep_json_path))
    file_findings = {}  # name -> list of raw finding dicts
    file_cwes = {}       # name -> set of cwes found
    for r in data['results']:
        name = extract_test_name(r['path'])
        if not name:
            continue
        file_findings.setdefault(name, []).append(r)
        file_cwes.setdefault(name, set()).update(extract_cwes(r))

    review = []
    for name, info in gt.items():
        cwe = info['cwe']
        if cwe in file_cwes.get(name, set()):
            matching = [r for r in file_findings[name] if cwe in extract_cwes(r) or CWE_SYNONYMS.get(cwe) in extract_cwes(r)]
            rep = matching[0] if matching else file_findings[name][0]
            review.append({
                'test': name,
                'cwe': cwe,
                'rule': rep['check_id'],
                'message': rep['extra'].get('message', ''),
                'line': rep['start']['line'],
            })
    return review


def read_source(src_root, test_name):
    p = Path(src_root) / f"{test_name}.java"
    return p.read_text(errors='replace') if p.exists() else None


def build_batch_prompt(batch, src_root):
    parts = [
        "You are a security code reviewer verifying candidate SAST findings from Semgrep.",
        "For EACH finding below, read the full source file and decide if the reported vulnerability",
        "is a TRUE POSITIVE (exploitable, no effective sanitization/validation/parameterization before the sink)",
        "or a FALSE POSITIVE (input is sanitized, validated, parameterized, encoded, or otherwise safe before reaching the sink).",
        "",
        "Respond with a JSON array containing EXACTLY ONE object per finding below, no other text:",
        '[{"test": "BenchmarkTestNNNNN", "verdict": "true_positive"}, ...]',
        "The verdict field must be exactly \"true_positive\" or \"false_positive\".",
        "You MUST include one entry for every single test name listed below, in the same order.",
        "",
        "=== FINDINGS TO REVIEW ===",
    ]
    included = []
    for item in batch:
        src = read_source(src_root, item['test'])
        if src is None:
            continue
        included.append(item)
        cwe_name = CWE_NAMES.get(item['cwe'], f"CWE-{item['cwe']}")
        parts.append(f"\n--- Finding for {item['test']} ---")
        parts.append(f"CWE: {item['cwe']} ({cwe_name})")
        parts.append(f"Semgrep rule: {item['rule']}")
        parts.append(f"Flagged line: {item['line']}")
        parts.append(f"Message: {item['message']}")
        parts.append(f"Source code:\n```java\n{src}\n```")
    return "\n".join(parts), included


VERDICT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "test": {"type": "string"},
            "verdict": {"type": "string", "enum": ["true_positive", "false_positive"]},
        },
        "required": ["test", "verdict"],
    },
}


def call_ollama(prompt, model, host, timeout=300, think=False):
    # Thinking-capable models (qwen3, qwen3.6) combined with a constrained JSON
    # `format` schema put the answer in the "thinking" field and leave "response"
    # empty. Disable thinking explicitly, and fall back to "thinking" if needed.
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": VERDICT_SCHEMA,
        "think": bool(think),
        "options": {"temperature": 0, "num_ctx": 32768},
    }
    payload = json.dumps(body).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    out = data.get("response") or ""
    if not out.strip():
        out = data.get("thinking") or ""
    return out


def parse_verdicts(raw_text, expected_tests):
    try:
        data = json.loads(raw_text)
        if not isinstance(data, list):
            return None
        verdicts = {}
        for item in data:
            t = item.get('test')
            v = item.get('verdict')
            if t is None or v not in ('true_positive', 'false_positive'):
                return None
            verdicts[t] = v
        if set(verdicts.keys()) != set(expected_tests):
            return None
        return verdicts
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--semgrep-json', required=True)
    ap.add_argument('--src-root', required=True)
    ap.add_argument('--model', default='qwen3-coder:30b')
    ap.add_argument('--host', default='http://localhost:11434')
    ap.add_argument('--batch-size', type=int, default=15)
    ap.add_argument('--limit-batches', type=int, default=None)
    ap.add_argument('--sample-file', default=None)
    ap.add_argument('--think', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    gt = load_ground_truth(args.gt)
    if args.sample_file:
        review = json.load(open(args.sample_file))
        print(f"Review set (from sample file): {len(review)} files")
    else:
        review = build_review_set(gt, args.semgrep_json)
        print(f"Review set (TP+FP from baseline): {len(review)} files")

    batches = [review[i:i + args.batch_size] for i in range(0, len(review), args.batch_size)]
    if args.limit_batches:
        batches = batches[:args.limit_batches]
    print(f"Batches: {len(batches)} (batch size {args.batch_size})")

    results = {}  # test_name -> verdict ('true_positive'/'false_positive'/'fail_open')
    fail_open_count = 0

    for bi, batch in enumerate(batches):
        t0 = time.time()
        prompt, included = build_batch_prompt(batch, args.src_root)
        expected = [it['test'] for it in included]
        try:
            raw = call_ollama(prompt, args.model, args.host, think=args.think)
            verdicts = parse_verdicts(raw, expected)
        except Exception as e:
            verdicts = None
            print(f"  batch {bi}: LLM call error: {e}")

        if verdicts is None:
            fail_open_count += 1
            for t in expected:
                results[t] = 'fail_open'
            print(f"  batch {bi+1}/{len(batches)}: FAIL-OPEN (malformed/error), {len(expected)} files retained, {time.time()-t0:.1f}s")
        else:
            for t, v in verdicts.items():
                results[t] = v
            tp_count = sum(1 for v in verdicts.values() if v == 'true_positive')
            print(f"  batch {bi+1}/{len(batches)}: OK, {tp_count}/{len(verdicts)} retained as true_positive, {time.time()-t0:.1f}s")

        with open(args.out, 'w') as f:
            json.dump({'results': results, 'fail_open_batches': fail_open_count, 'batches_done': bi + 1}, f, indent=2)

    print(f"\nDone. {fail_open_count}/{len(batches)} batches fail-opened.")
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
