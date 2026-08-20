#!/usr/bin/env python3
"""Score Semgrep (or QASecClaw-filtered) findings against OWASP Benchmark v1.2 ground truth.

File-level, CWE-matched scoring, matching the OWASP Benchmark convention:
a test file counts as "flagged" only if a finding's CWE matches the file's
ground-truth CWE category.
"""
import argparse
import csv
import json
import re
from pathlib import Path


def load_ground_truth(csv_path):
    gt = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            name, category, is_vuln, cwe = row[0], row[1], row[2], row[3]
            gt[name] = {
                'category': category,
                'vulnerable': is_vuln.strip().lower() == 'true',
                'cwe': int(cwe.strip()),
            }
    return gt


def extract_test_name(path):
    m = re.search(r'(BenchmarkTest\d+)', path)
    return m.group(1) if m else None


# Semgrep registry drift: some rules tag a sibling/parent CWE rather than the
# exact OWASP Benchmark ground-truth CWE for the same weakness class.
CWE_SYNONYMS = {326: 327}


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


def load_semgrep_flags(json_path):
    """Returns dict: test_name -> set of CWE ints flagged in that file."""
    data = json.load(open(json_path))
    flags = {}
    for r in data['results']:
        name = extract_test_name(r['path'])
        if not name:
            continue
        cwes = extract_cwes(r)
        flags.setdefault(name, set()).update(cwes)
    return flags


def score(gt, flags):
    tp = fp = tn = fn = 0
    per_cwe = {}
    for name, info in gt.items():
        cwe = info['cwe']
        vulnerable = info['vulnerable']
        flagged = cwe in flags.get(name, set())
        per_cwe.setdefault(cwe, {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})
        if flagged and vulnerable:
            tp += 1
            per_cwe[cwe]['tp'] += 1
        elif flagged and not vulnerable:
            fp += 1
            per_cwe[cwe]['fp'] += 1
        elif not flagged and vulnerable:
            fn += 1
            per_cwe[cwe]['fn'] += 1
        else:
            tn += 1
            per_cwe[cwe]['tn'] += 1
    return tp, fp, tn, fn, per_cwe


def metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    youden_j = recall - fpr
    return precision, recall, f1, fpr, youden_j


CWE_NAMES = {
    22: 'Path Traversal', 78: 'Command Injection', 79: 'XSS', 89: 'SQL Injection',
    90: 'LDAP Injection', 327: 'Weak Cryptography', 328: 'Weak Hashing',
    330: 'Weak Randomness', 501: 'Trust Boundary Violation', 614: 'Insecure Cookie',
    643: 'XPath Injection',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--semgrep-json', required=True)
    ap.add_argument('--label', default='Tool')
    args = ap.parse_args()

    gt = load_ground_truth(args.gt)
    flags = load_semgrep_flags(args.semgrep_json)
    tp, fp, tn, fn, per_cwe = score(gt, flags)
    precision, recall, f1, fpr, j = metrics(tp, fp, tn, fn)

    print(f"=== {args.label} — Overall (OWASP Benchmark v1.2, {len(gt)} test cases) ===")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f} FPR={fpr:.3f} YoudenJ={j:.3f}")
    print()
    print(f"{'CWE':6} {'Category':22} {'Prec':6} {'Rec':6} {'F1':6} {'TP':4} {'FP':4} {'TN':4} {'FN':4}")
    for cwe in sorted(per_cwe):
        c = per_cwe[cwe]
        p, r, f, _, _ = metrics(c['tp'], c['fp'], c['tn'], c['fn'])
        name = CWE_NAMES.get(cwe, '?')
        print(f"{cwe:<6} {name:22} {p:.3f}  {r:.3f}  {f:.3f}  {c['tp']:<4} {c['fp']:<4} {c['tn']:<4} {c['fn']:<4}")


if __name__ == '__main__':
    main()
