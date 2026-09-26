from common import *
from sklearn.metrics import f1_score
rng=np.random.default_rng(0)
def iris_metrics(keep,y,proj):
    # IRIS Sec3.6: per project; paths kept = predicted vulnerable
    f1s=[];fdr=[];det=0
    for g in np.unique(proj):
        k=keep[proj==g]; yy=y[proj==g]
        n=k.sum(); tp=(k&(yy==1)).sum()
        rec=1.0 if tp>0 else 0.0; det+=rec
        prec=tp/n if n else 0.0
        if n: fdr.append(1-prec)
        f1s.append(2*prec*rec/(prec+rec) if prec+rec else 0)
    return np.mean(f1s),det
def oracle_pred(V,y):  # per unit: correct if any model correct
    anyc=(V==y[:,None]).any(1)
    return np.where(anyc,y,1-y)
for name,(P,V,S,meta) in [('IRIS',iris()),('PrimeVul',primevul())]:
    y=meta.loc[P.index,'gold_label'].values.astype(int); Vm=V[MODELS].values.astype(int)
    grp=meta.loc[P.index,'proj' if name=='IRIS' else 'project'].values
    best=max(f1_score(y,Vm[:,j]) for j in range(5))
    o=oracle_pred(Vm,y); real=f1_score(y,o)
    extra=''
    if name=='IRIS':
        extra=f' | IRIS AvgF1 oracle={iris_metrics(o==1,y,grp)[0]:.3f}'
    nulls=[];nullsI=[]
    for r in range(50):
        Vn=Vm.copy()
        for j in range(5):  # shuffle each model's verdicts within group: keeps per-project positive rate, destroys skill
            for g in np.unique(grp):
                idx=np.where(grp==g)[0]; Vn[idx,j]=Vn[rng.permutation(idx),j]
        on=oracle_pred(Vn,y); nulls.append(f1_score(y,on))
        if name=='IRIS': nullsI.append(iris_metrics(on==1,y,grp)[0])
    print(f'{name}: best single F1={best:.3f}  oracle F1={real:.3f}  SKILL-FREE oracle F1={np.mean(nulls):.3f}+-{np.std(nulls):.3f}{extra}'+(f'  skill-free IRIS AvgF1={np.mean(nullsI):.3f}+-{np.std(nullsI):.3f}' if nullsI else ''))
    # fraction of units where at least one model is right
    print('   any-model-right: real', (Vm==y[:,None]).any(1).mean().round(3), 'null', round(np.mean([(1)]),3))
