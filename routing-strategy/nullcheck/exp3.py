from common import *
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
rng=np.random.default_rng(2)
P,V,S,meta=primevul(); y=meta.loc[P.index,'gold_label'].values.astype(int)
meta=meta.loc[P.index]; cwe=meta['cwe'].fillna('NA').values
# pair-rank per CWE (buggy scored above its fix), heterogeneity vs null
pid=meta.pair_id.values
d=pd.DataFrame({'pid':pid,'y':y,'cwe':cwe})
for m in ['granite-8b','phi-3.8b','qwen-7b']:
    d['p']=P[m].values
    g=d.groupby('pid').filter(lambda x:len(x)==2 and x.y.sum()==1)
    r=g.groupby('pid').apply(lambda x:pd.Series({'cwe':x.cwe.iloc[0],'w':float(x[x.y==1].p.iloc[0]>x[x.y==0].p.iloc[0])+0.5*float(x[x.y==1].p.iloc[0]==x[x.y==0].p.iloc[0])}))
    top=r.groupby('cwe').w.agg(['size','mean']).query('size>=30').sort_values('mean')
    # null: shuffle cwe labels across pairs -> spread of per-CWE means expected by chance
    sd_real=top['mean'].std()
    sds=[]
    for _ in range(300):
        rr=r.copy(); rr['cwe']=rng.permutation(rr.cwe.values)
        t=rr.groupby('cwe').w.agg(['size','mean']).loc[top.index]; sds.append(t['mean'].std())
    print(f'{m}: per-CWE pair-rank range {top["mean"].min():.2f}..{top["mean"].max():.2f} over {len(top)} CWEs; spread sd={sd_real:.3f} vs chance {np.mean(sds):.3f} (p={np.mean(np.array(sds)>=sd_real):.2f})')
# code length slices
L=meta.in_tokens.values if 'in_tokens' in meta else None
q=pd.qcut(L,4,labels=False)
print('AUC by input-length quartile (granite, qwen-7b, phi):')
for b in range(4):
    i=q==b; print(' Q',b,[round(roc_auc_score(y[i],P[m].values[i]),3) for m in ['granite-8b','qwen-7b','phi-3.8b']], 'pos',round(y[i].mean(),3))
# stacked: all 5 scores + length, train valid -> test
X=np.column_stack([P[MODELS].values,V[MODELS].values,np.log1p(L)])
tr=meta.split.values=='valid'; te=~tr
clf=LogisticRegression(max_iter=2000).fit(X[tr],y[tr]); print('stacked 5-model + length, valid->test AUC',round(roc_auc_score(y[te],clf.predict_proba(X[te])[:,1]),3))
clf=LogisticRegression(max_iter=2000).fit(np.log1p(L[tr])[:,None],y[tr]); print('length ALONE AUC',round(roc_auc_score(y[te],clf.predict_proba(np.log1p(L[te])[:,None])[:,1]),3))
# G2: is cost per unit non-constant?
print('cost variation (seconds) per model: CV across units, corr with in_tokens')
df=pd.read_json(R+'data/table_dedup.jsonl',lines=True)
for m in MODELS:
    x=df[df.model==m]; print(f' {m:14s} CV={x.seconds.std()/x.seconds.mean():.2f} p90/p10={x.seconds.quantile(.9)/x.seconds.quantile(.1):.1f} corr(in_tok)={np.corrcoef(x.seconds,x.in_tokens)[0,1]:.2f} corr(out_tok)={np.corrcoef(x.seconds,x.out_tokens)[0,1]:.2f}')
