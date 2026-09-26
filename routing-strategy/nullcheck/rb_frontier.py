"""The cost-quality frontier of oracle routing, real vs shuffle-null.

RouterBench's headline object is the frontier: sweep a cost weight lambda, let
the oracle pick argmax_m (score - lambda*cost) per prompt, and plot mean quality
against mean cost. AIQ is the normalised area under it.

If the null frontier sits ON or ABOVE the real one, then the published frontier
is a function of each model's marginal accuracy and price, not of any
per-prompt complementarity a router could learn to exploit.
"""
import numpy as np, pandas as pd
d = pd.read_pickle('rb0.pkl')
MODELS = ['WizardLM/WizardLM-13B-V1.2','claude-instant-v1','claude-v1','claude-v2',
          'gpt-3.5-turbo-1106','gpt-4-1106-preview','meta/code-llama-instruct-34b-chat',
          'meta/llama-2-70b-chat','mistralai/mistral-7b-chat','mistralai/mixtral-8x7b-chat',
          'zero-one-ai/Yi-34B-Chat']
S = d[MODELS].astype(float).values
C = d[[m+'|total_cost' for m in MODELS]].astype(float).values
C = np.nan_to_num(C, nan=np.nanmean(C))
ev = d.eval_name.values
rng = np.random.default_rng(0)

def shuffled(S, ev):
    Sn = S.copy()
    for g in np.unique(ev):
        i = np.where(ev == g)[0]
        for j in range(S.shape[1]):
            Sn[i, j] = Sn[rng.permutation(i), j]
    return Sn

LAMS = np.concatenate([[0.0], np.logspace(-1, 4, 24)])
def frontier(S, C):
    pts = []
    for lam in LAMS:
        pick = (S - lam * C).argmax(1)
        r = np.arange(len(S))
        pts.append((C[r, pick].mean(), S[r, pick].mean()))
    return np.array(sorted(pts))

def aiq(pts, cmax, qmax):
    """Area under the frontier, normalised: 1.0 = perfect quality at zero cost."""
    c, q = pts[:, 0] / cmax, np.clip(pts[:, 1] / qmax, 0, 1)
    # step-interpolate over [0,1] cost and integrate
    grid = np.linspace(0, 1, 501)
    qi = np.interp(grid, c, q, left=q[0], right=q[-1])
    return qi.mean()

Fr = frontier(S, C)
nulls = [frontier(shuffled(S, ev), C) for _ in range(10)]
cmax, qmax = C.max(0).mean(), 1.0
print(f'AIQ real   {aiq(Fr, cmax, qmax):.4f}')
print(f'AIQ null   {np.mean([aiq(f, cmax, qmax) for f in nulls]):.4f} '
      f'+-{np.std([aiq(f, cmax, qmax) for f in nulls]):.4f}')
print('\nfrontier points (cost $, quality) — real vs null mean')
Fn = np.mean(nulls, axis=0)
print(f'{"cost$":>10s} {"real q":>8s} | {"cost$":>10s} {"null q":>8s}')
for i in range(0, len(Fr), 3):
    print(f'{Fr[i,0]:10.5f} {Fr[i,1]:8.4f} | {Fn[i,0]:10.5f} {Fn[i,1]:8.4f}')
# single-model reference points
print('\nsingle models: (cost, acc)')
for j, m in enumerate(MODELS):
    print(f'  {m[:34]:34s} ${C[:,j].mean():.5f} {S[:,j].mean():.4f}')
# how much of the oracle's quality is reachable by a model that knows only marginals?
best_by_acc = S.mean(0).argmax()
print(f'\nbest single {S[:,best_by_acc].mean():.4f} at ${C[:,best_by_acc].mean():.5f}; '
      f'oracle {S.max(1).mean():.4f}; null oracle {shuffled(S,ev).max(1).mean():.4f}')
