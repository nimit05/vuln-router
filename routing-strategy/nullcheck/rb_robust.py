"""Robustness of the E8 claim: 5-shot RouterBench, bootstrap CIs, and a check
that the result is not an artifact of how the shuffle preserves marginals."""
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
MODELS = ['WizardLM/WizardLM-13B-V1.2','claude-instant-v1','claude-v1','claude-v2',
          'gpt-3.5-turbo-1106','gpt-4-1106-preview','meta/code-llama-instruct-34b-chat',
          'meta/llama-2-70b-chat','mistralai/mistral-7b-chat','mistralai/mixtral-8x7b-chat',
          'zero-one-ai/Yi-34B-Chat']
def shuffled(S, ev):
    Sn = S.copy()
    for g in np.unique(ev):
        i = np.where(ev == g)[0]
        for j in range(S.shape[1]):
            Sn[i, j] = Sn[rng.permutation(i), j]
    return Sn
for tag, path in [('0-shot','rb0.pkl'), ('5-shot','rb5.pkl')]:
    d = pd.read_pickle(path).dropna(subset=MODELS)
    cols = [m for m in MODELS if m in d.columns]
    S = d[cols].astype(float).values; ev = d.eval_name.values
    best = S.mean(0).max(); o = S.max(1).mean()
    nulls = [shuffled(S, ev).max(1).mean() for _ in range(30)]
    # bootstrap the gap over prompts
    gaps = []
    for _ in range(200):
        b = rng.integers(0, len(S), len(S))
        Sb, evb = S[b], ev[b]
        gaps.append(Sb.max(1).mean() - shuffled(Sb, evb).max(1).mean())
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    # correlation of errors: mean pairwise correlation of correctness across models
    B = (S > 0.5).astype(float)
    cc = np.corrcoef(B.T); iu = np.triu_indices(len(cols), 1)
    print(f'{tag}: n={len(d)} models={len(cols)}  best={best:.4f}  oracle={o:.4f}  '
          f'null={np.mean(nulls):.4f}  gap={o-np.mean(nulls):+.4f} '
          f'95% CI [{lo:+.4f},{hi:+.4f}]  mean pairwise error corr={cc[iu].mean():+.3f}')
