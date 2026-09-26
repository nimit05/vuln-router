"""What does a RouterBench router actually learn: the task, or the prompt?

Three routers, same label (does the cheap model already get this prompt right):
  A  prompt text (TF-IDF char n-grams)          - what papers fit
  B  benchmark identity only (one-hot), no text - knows the task, nothing else
  C  prompt text, scored WITHIN each benchmark  - per-prompt skill with task
                                                  identity held constant
If B ~= A, the router is a task classifier. If C ~= 0.5, it has no per-prompt
skill at all and the whole effect is task-level difficulty.
"""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
d = pd.read_pickle('rb0.pkl')
CHEAP = 'mistralai/mixtral-8x7b-chat'
y = (d[CHEAP].astype(float) > 0.5).astype(int).values
ev = d.eval_name.values
X = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=5, max_features=200000
                    ).fit_transform(d.prompt.astype(str).str[:4000].values)
E1 = OneHotEncoder(handle_unknown='ignore').fit_transform(ev.reshape(-1,1))
kf = KFold(5, shuffle=True, random_state=0)
res = {}
for name, M in [('A text', X), ('B benchmark-id only', E1)]:
    a = []
    for tr, te in kf.split(y):
        m = LogisticRegression(max_iter=1000).fit(M[tr], y[tr])
        a.append(roc_auc_score(y[te], m.predict_proba(M[te])[:,1]))
    res[name] = (np.mean(a), np.std(a)); print(f'{name:22s} AUC {np.mean(a):.3f} +- {np.std(a):.3f}')
# C: text router, AUC computed within each benchmark (>=50 prompts, both classes)
oof = np.zeros(len(y))
for tr, te in kf.split(y):
    oof[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:,1]
per = []
for g in np.unique(ev):
    i = ev == g
    if i.sum() >= 50 and 0 < y[i].mean() < 1:
        per.append(roc_auc_score(y[i], oof[i]))
print(f'C text, WITHIN benchmark   AUC {np.mean(per):.3f} +- {np.std(per):.3f} '
      f'over {len(per)} benchmarks')
# does the task-level rate alone explain the label? (base-rate router)
rate = pd.Series(y).groupby(ev).transform('mean').values
print(f'D task base-rate only     AUC {roc_auc_score(y, rate):.3f}  (no learning at all)')
