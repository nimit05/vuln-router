"""The control the reasoning result needs: same-yes-rate random guessing.

pair-both-right rewards giving DIFFERENT answers to the two versions of a pair.
A guesser that says "vulnerable" with probability r, independently per function,
scores r*(1-r) -- which is 0.25 at r=0.5. So any configuration must be compared
against its OWN yes-rate's random baseline, analytic and by permutation
(shuffling verdicts across units preserves the rate and destroys within-pair
structure).

Also reported: within-pair agreement, i.e. how often the model gives the SAME
verdict to the buggy and fixed versions. Independent guessing at rate r agrees
r^2 + (1-r)^2 of the time; more agreement than that means the model is reacting
to the function's surface form rather than the defect.
"""
import glob, json
import numpy as np, pandas as pd
rng = np.random.default_rng(0)

def pb(df, col='q'):
    v = df.pivot_table(index='pair_id', columns='gold_label', values=col).dropna()
    return float(((v[1] == 1) & (v[0] == 0)).mean())

def agree(df, col='q'):
    v = df.pivot_table(index='pair_id', columns='gold_label', values=col).dropna()
    return float((v[1] == v[0]).mean())

def report(tag, d):
    d = d.copy(); d['q'] = d.pred.fillna(0).astype(int)
    r = d.q.mean()
    obs, ag = pb(d), agree(d)
    analytic = r * (1 - r)
    ag_ind = r**2 + (1 - r)**2
    nulls = []
    for _ in range(300):
        dn = d.copy(); dn['q'] = rng.permutation(dn.q.values)
        nulls.append(pb(dn))
    print(f'{tag:24s} yes={r:.2f}  pair-both={obs:.3f}  same-rate null={np.mean(nulls):.3f}'
          f'+-{np.std(nulls):.3f}  analytic r(1-r)={analytic:.3f}  '
          f'excess={obs-np.mean(nulls):+.3f} | within-pair agreement {ag:.2f} vs {ag_ind:.2f} if independent')

print('=== thinking budget sweep (Qwen3-8B)')
for b in [0, 256, 512, 1024, 2048, 4096]:
    d = pd.DataFrame([json.loads(l) for l in open(f'budget_{b}.jsonl')]).drop_duplicates('unit_id')
    report(f'budget {b}', d)
print('\n=== E6 runs')
for f, tag in [('qwen3_nothink', 'qwen3 no-think'), ('qwen3_think', 'qwen3 think (unbounded)'),
               ('phi4_direct', 'phi-4 direct'), ('phi4_cot', 'phi-4 CoT')]:
    d = pd.DataFrame([json.loads(l) for l in open(f + '.jsonl')]).drop_duplicates('unit_id')
    report(tag, d)
print('\n=== self-consistency')
d = pd.DataFrame([json.loads(l) for l in open('sc5_1024.jsonl')]).drop_duplicates('unit_id')
report('sc k=5 @1024', d)
print('\n=== the 5-model pool on the same metric (PrimeVul pairs)')
R = '/Users/nimitwadhwa/Documents/projects/vuln-pred-results/routing-strategy/'
t = pd.read_json(R + 'data/table_dedup.jsonl', lines=True)
for m in ['qwen-1.5b', 'phi-3.8b', 'qwen-7b', 'granite-8b', 'deepseek-6.7b']:
    x = t[t.model == m].rename(columns={'pred_label': 'pred'})
    report(m, x[['pair_id', 'gold_label', 'pred']])
