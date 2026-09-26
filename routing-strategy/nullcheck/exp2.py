from common import *
from null_oracle import iris_metrics
import itertools
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(1)
P,V,S,meta=iris(); y=meta.loc[P.index,'gold_label'].values.astype(int); proj=meta.loc[P.index,'proj'].values
Vm=V[MODELS].values.astype(int)
def shuf(col):
    c=col.copy()
    for g in np.unique(proj):
        i=np.where(proj==g)[0]; c[i]=c[rng.permutation(i)]
    return c
print('--- A. pair oracle vs skill-free pair oracle (IRIS AvgF1)')
for a,b in [('qwen-7b','granite-8b'),('qwen-1.5b','granite-8b'),('phi-3.8b','granite-8b'),('qwen-1.5b','qwen-7b')]:
    va,vb=V[a].values.astype(int),V[b].values.astype(int)
    o=np.where((va==y)|(vb==y),y,1-y)
    ns=[]
    for _ in range(40):
        sa,sb=shuf(va),shuf(vb); ns.append(iris_metrics(np.where((sa==y)|(sb==y),y,1-y)==1,y,proj)[0])
    print(f'{a}->{b}: oracle {iris_metrics(o==1,y,proj)[0]:.3f}  skill-free {np.mean(ns):.3f}+-{np.std(ns):.3f}')
print('--- B. metric gaming: keep k RANDOM paths per project (no model at all)')
for k in [1,2,3,5,10]:
    fs=[]
    for _ in range(200):
        keep=np.zeros(len(y),bool)
        for g in np.unique(proj):
            i=np.where(proj==g)[0]; keep[rng.choice(i,min(k,len(i)),replace=False)]=True
        fs.append(iris_metrics(keep,y,proj))
    fs=np.array(fs); print(f'random k={k}: AvgF1 {fs[:,0].mean():.3f}+-{fs[:,0].std():.3f}  #Det {fs[:,1].mean():.1f}')
print('reference: always-granite 0.207, router+floor best 0.247, floor-only k=10 0.237')
# oracle-in-hindsight k: top-k by qwen-7b p
for k in [1,5,10]:
    keep=np.zeros(len(y),bool)
    for g in np.unique(proj):
        i=np.where(proj==g)[0]; keep[i[np.argsort(-P['qwen-7b'].values[i])[:k]]]=True
    print(f'qwen-7b top-{k}/project: AvgF1 {iris_metrics(keep,y,proj)[0]:.3f}')
print('--- C. per-project path counts and positives')
d=pd.DataFrame({'proj':proj,'y':y}).groupby('proj').y.agg(['size','sum']); print(d.to_string())
