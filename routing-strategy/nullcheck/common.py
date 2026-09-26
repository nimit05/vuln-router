import pandas as pd, numpy as np, glob, json
R='/Users/nimitwadhwa/Documents/projects/vuln-pred-results/routing-strategy/'
MODELS=['qwen-1.5b','phi-3.8b','qwen-7b','granite-8b','deepseek-6.7b']
def pv(df):
    lp=df.decision_logprob
    pc=np.exp(lp.fillna(np.log(0.5)))
    p=np.where(df.pred_label==1,pc,1-pc)
    p=np.where(df.decision_logprob.isna(),0.5,p)
    return p
def wide(df,key='unit_id'):
    df=df.copy(); df['p']=pv(df)
    P=df.pivot_table(index=key,columns='model',values='p')
    V=df.pivot_table(index=key,columns='model',values='pred_label')
    S=df.pivot_table(index=key,columns='model',values='seconds')
    meta=df.drop_duplicates(key).set_index(key)
    return P,V,S,meta
def iris():
    df=pd.concat([pd.read_json(f,lines=True) for f in glob.glob(R+'data/gr/probe5/*.jsonl')])
    P,V,S,meta=wide(df); sl=pd.read_json(R+'data/gr/slices.jsonl',lines=True).set_index('path_id')
    meta=meta.join(sl[['project','n_hops','cross_file','slice_loc','slice_chars','max_nest','n_if','n_loop','n_call','n_catch','n_methods','rule']],rsuffix='_s')
    meta['proj']=meta['project_s'] if 'project_s' in meta else meta['project']
    return P,V,S,meta
def primevul():
    df=pd.read_json(R+'data/table_dedup.jsonl',lines=True)
    return wide(df)
