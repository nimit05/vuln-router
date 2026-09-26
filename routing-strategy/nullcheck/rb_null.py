"""Null-oracle test on RouterBench (Hu et al. 2024): is the reported oracle
headroom complementary skill, or diversity of error?

Shuffle each model's per-prompt score WITHIN each benchmark. That preserves
every model's accuracy on every benchmark and destroys any per-prompt skill.
If the shuffled oracle still matches the real one, the oracle is not evidence
that models are complementary.
"""
import numpy as np, pandas as pd
d = pd.read_pickle('rb0.pkl')
MODELS = ['WizardLM/WizardLM-13B-V1.2','claude-instant-v1','claude-v1','claude-v2',
          'gpt-3.5-turbo-1106','gpt-4-1106-preview','meta/code-llama-instruct-34b-chat',
          'meta/llama-2-70b-chat','mistralai/mistral-7b-chat','mistralai/mixtral-8x7b-chat',
          'zero-one-ai/Yi-34B-Chat']
S = d[MODELS].astype(float).values
print('score values:', np.unique(S)[:6], '... n distinct', len(np.unique(S)))
ev = d.eval_name.values
C = d[[m+'|total_cost' for m in MODELS]].astype(float).values
rng = np.random.default_rng(0)

def oracle(S):           # best achievable per prompt
    return S.max(1).mean()
def null_oracle(S, ev, reps=30):
    out=[]
    for _ in range(reps):
        Sn = S.copy()
        for g in np.unique(ev):
            i = np.where(ev==g)[0]
            for j in range(S.shape[1]):
                Sn[i,j] = Sn[rng.permutation(i),j]
        out.append(oracle(Sn))
    return np.mean(out), np.std(out)

best = S.mean(0).max(); bi = S.mean(0).argmax()
o = oracle(S); nm, ns = null_oracle(S, ev)
print(f'\nALL {len(d)} prompts, {len(MODELS)} models')
print(f'  best single model      {best:.4f}  ({MODELS[bi]})')
print(f'  oracle (best per prompt) {o:.4f}   headroom {o-best:+.4f}')
print(f'  SHUFFLE-NULL oracle      {nm:.4f}+-{ns:.4f}   null headroom {nm-best:+.4f}')
print(f'  >>> real complementarity = oracle - null = {o-nm:+.4f}')
print(f'\nper benchmark:')
print(f'{"eval":28s} {"n":>6s} {"best":>7s} {"oracle":>7s} {"null":>7s} {"real-comp":>9s}')
for g in sorted(pd.unique(ev)):
    i = ev==g; Sg=S[i]
    b=Sg.mean(0).max(); og=oracle(Sg); n,_=null_oracle(Sg, ev[i], reps=20)
    print(f'{g[:28]:28s} {i.sum():6d} {b:7.4f} {og:7.4f} {n:7.4f} {og-n:+9.4f}')
# cost side: cheapest-correct oracle vs null, i.e. the frontier RouterBench plots
cheap_cost = np.where(S>0, C, np.nan)
real_cost = np.nanmin(cheap_cost, axis=1)
print(f'\ncost of the cheapest CORRECT model, mean ${np.nanmean(real_cost):.5f} vs '
      f'always-gpt-4 ${C[:,MODELS.index("gpt-4-1106-preview")].mean():.5f}')
