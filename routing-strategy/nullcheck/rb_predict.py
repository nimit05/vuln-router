"""Is the routing label predictable OUT of domain on RouterBench?

Label: which model wins on this prompt (cheapest correct model, RouteLLM-style
binary version: does the cheap model already get it right?). Backbone: TF-IDF
char n-grams + logistic regression on the prompt text.

Two splits, and the gap between them is the whole point:
  random      - prompts from the same benchmark on both sides (what papers do)
  leave-one-benchmark-out - a genuinely new task family at test time
"""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import roc_auc_score
d = pd.read_pickle('rb0.pkl')
CHEAP, STRONG = 'mistralai/mixtral-8x7b-chat', 'gpt-4-1106-preview'
y = (d[CHEAP].astype(float) > 0.5).astype(int).values      # 1 = cheap model suffices
txt = d.prompt.astype(str).str[:4000].values
ev = d.eval_name.values
print(f'n={len(y)}  cheap-suffices rate={y.mean():.3f}  '
      f'strong acc={(d[STRONG].astype(float)>0.5).mean():.3f}')
V = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=5, max_features=200000)
X = V.fit_transform(txt)
for name, splitter, groups in [('random 5-fold (leaky across benchmarks)', KFold(5, shuffle=True, random_state=0), None),
                               ('leave-benchmark-out 5-fold', GroupKFold(5), ev)]:
    aucs = []
    for tr, te in splitter.split(X, y, groups):
        m = LogisticRegression(max_iter=1000, C=1.0).fit(X[tr], y[tr])
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    print(f'  {name:42s} AUC {np.mean(aucs):.3f} +- {np.std(aucs):.3f}')
# what does the leaky signal actually encode? predict the benchmark id itself
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder().fit(ev)
tr, te = next(iter(KFold(5, shuffle=True, random_state=0).split(X)))
m = LogisticRegression(max_iter=400).fit(X[tr], le.transform(ev)[tr])
print(f'  benchmark identity recoverable from prompt text: accuracy '
      f'{m.score(X[te], le.transform(ev)[te]):.3f} over {len(le.classes_)} classes')
