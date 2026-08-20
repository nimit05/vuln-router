#!/usr/bin/env python3
"""Score the QASecClaw-style pipeline: Semgrep baseline flags, with the
SAST Filter Agent's LLM verdicts applied (suppress false_positive verdicts,
fail-open / no-verdict files stay flagged as in the Semgrep baseline)."""
import argparse
import csv
import json
import re

CWE_NAMES = {
    22: 'Path Traversal', 78: 'Command Injection', 79: 'XSS', 89: 'SQL Injection',
    90: 'LDAP Injection', 327: 'Weak Cryptography', 328: 'Weak Hashing',
    330: 'Weak Randomness', 501: 'Trust Boundary Violation', 614: 'Insecure Cookie',
    643: 'XPath Injection',
}
CWE_SYNONYMS = {326: 327}


def load_ground_truth(csv_path):
    gt = {}
    with open(csv_path) as f:
        for row in csv.reader(f):
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


def baseline_flags(gt, semgrep_json_path):
    data = json.load(open(semgrep_json_path))
    file_cwes = {}
    for r in data['results']:
        name = extract_test_name(r['path'])
        if not name:
            continue
        file_cwes.setdefault(name, set()).update(extract_cwes(r))
    flagged = set()
    for name, info in gt.items():
        if info['cwe'] in file_cwes.get(name, set()):
            flagged.add(name)
    return flagged


def apply_filter(flagged, filter_results_path):
    verdicts = json.load(open(filter_results_path))['results']
    final = set()
    for name in flagged:
        v = verdicts.get(name, 'fail_open')  # not reviewed -> treat as fail-open (retain)
        if v == 'false_positive':
            continue  # suppressed
        final.add(name)  # true_positive or fail_open -> retained
    return final, verdicts


def metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    j = recall - fpr
    return precision, recall, f1, fpr, j


def score(gt, final_flags):
    tp = fp = tn = fn = 0
    per_cwe = {}
    for name, info in gt.items():
        cwe = info['cwe']
        vulnerable = info['vulnerable']
        flagged = name in final_flags
        per_cwe.setdefault(cwe, {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})
        if flagged and vulnerable:
            tp += 1; per_cwe[cwe]['tp'] += 1
        elif flagged and not vulnerable:
            fp += 1; per_cwe[cwe]['fp'] += 1
        elif not flagged and vulnerable:
            fn += 1; per_cwe[cwe]['fn'] += 1
        else:
            tn += 1; per_cwe[cwe]['tn'] += 1
    return tp, fp, tn, fn, per_cwe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--semgrep-json', required=True)
    ap.add_argument('--filter-results', required=True)
    ap.add_argument('--model-label', default='LLM SAST Filter Agent')
    args = ap.parse_args()

    gt = load_ground_truth(args.gt)
    flagged = baseline_flags(gt, args.semgrep_json)
    final_flags, verdicts = apply_filter(flagged, args.filter_results)

    n_suppressed = len(flagged) - len(final_flags)
    n_fail_open = sum(1 for v in verdicts.values() if v == 'fail_open')
    print(f"Baseline flagged files (TP+FP): {len(flagged)}")
    print(f"Suppressed as false_positive by LLM filter: {n_suppressed}")
    print(f"Fail-open (retained without review): {n_fail_open}")
    print(f"Final retained (QASecClaw positives): {len(final_flags)}")
    print()

    tp, fp, tn, fn, per_cwe = score(gt, final_flags)
    precision, recall, f1, fpr, j = metrics(tp, fp, tn, fn)
    print(f"=== QASecClaw (Semgrep + {args.model_label}) — Overall ({len(gt)} test cases) ===")
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
