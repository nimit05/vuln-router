"""The shuffle null has a closed form, so the diagnostic needs no model access.

For binary correctness and independent models with accuracies a_1..a_m, the
oracle ("at least one model is right") is

    null_oracle = 1 - prod_i (1 - a_i)

That is computable from a published per-model accuracy table alone. Here we
check the closed form against the permutation estimate, so the formula can be
trusted where raw predictions are not available.
"""
import glob, json
import numpy as np, pandas as pd
rng = np.random.default_rng(0)

def shuffle_null(B, grp, reps=30):
    out = []
    for _ in range(reps):
        Bn = B.copy()
        for g in np.unique(grp):
            i = np.where(grp == g)[0]
            for j in range(B.shape[1]):
                Bn[i, j] = Bn[rng.permutation(i), j]
        out.append(Bn.any(1).mean())
    return np.mean(out)

def closed_form(B, grp):
    """Per group, 1 - prod(1 - a_ij); then average over prompts."""
    tot = 0.0
    for g in np.unique(grp):
        i = grp == g
        a = B[i].mean(0)
        tot += i.sum() * (1 - np.prod(1 - a))
    return tot / len(B)

print(f'{"dataset":22s} {"oracle":>8s} {"shuffle":>8s} {"closed":>8s} {"excess":>8s}')
# RouterBench
for tag, path in [('RouterBench 0-shot', 'rb0.pkl'), ('RouterBench 5-shot', 'rb5.pkl')]:
    d = pd.read_pickle(path)
    MODELS = [c for c in d.columns if '|' not in c and c not in
              ('sample_id', 'prompt', 'eval_name', 'oracle_model_to_route_to')]
    S = d[MODELS].astype(float).dropna()
    B = (S > 0.5).values; grp = d.loc[S.index, 'eval_name'].values
    print(f'{tag:22s} {B.any(1).mean():8.4f} {shuffle_null(B,grp):8.4f} '
          f'{closed_form(B,grp):8.4f} {B.any(1).mean()-closed_form(B,grp):+8.4f}')
# our two benchmarks
R = '/Users/nimitwadhwa/Documents/projects/vuln-pred-results/routing-strategy/'
MOD = ['qwen-1.5b','phi-3.8b','qwen-7b','granite-8b','deepseek-6.7b']
def ours(name, files, keyproj):
    df = pd.concat([pd.read_json(f, lines=True) for f in files])
    V = df.pivot_table(index='unit_id', columns='model', values='pred_label')
    meta = df.drop_duplicates('unit_id').set_index('unit_id').loc[V.index]
    y = meta.gold_label.values.astype(int)
    B = (V[MOD].values == y[:, None])
    grp = meta[keyproj].values
    print(f'{name:22s} {B.any(1).mean():8.4f} {shuffle_null(B,grp):8.4f} '
          f'{closed_form(B,grp):8.4f} {B.any(1).mean()-closed_form(B,grp):+8.4f}')
ours('IRIS paths', glob.glob(R+'data/gr/probe5/*.jsonl'), 'project')
ours('PrimeVul functions', [R+'data/table_dedup.jsonl'], 'project')
# Qwen3/phi-4 reasoning modes (E6)
for tag, files in [('Qwen3 think/no-think', ['qwen3_nothink.jsonl','qwen3_think.jsonl']),
                   ('phi-4 direct/CoT', ['phi4_direct.jsonl','phi4_cot.jsonl'])]:
    ds = [pd.DataFrame([json.loads(l) for l in open(f)]).drop_duplicates('unit_id').set_index('unit_id') for f in files]
    ix = ds[0].index.intersection(ds[1].index)
    y = ds[0].loc[ix].gold_label.values.astype(int)
    B = np.column_stack([(d.loc[ix].pred.fillna(0).values.astype(int) == y) for d in ds])
    grp = ds[0].loc[ix].project.fillna('na').values
    print(f'{tag:22s} {B.any(1).mean():8.4f} {shuffle_null(B,grp):8.4f} '
          f'{closed_form(B,grp):8.4f} {B.any(1).mean()-closed_form(B,grp):+8.4f}')
