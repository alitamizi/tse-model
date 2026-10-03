import pandas as pd, numpy as np
P,N=pd.read_pickle('panel_raw.pkl')
OUT={}
for s,d in P.items():
    d=d[d['last'].notna()&(d['last']>0)&d.adj.notna()&(d.vol>0)].copy()
    if len(d)<150: continue
    f=d.adj/d['last']
    x=pd.DataFrame(index=d.index)
    for a,b in [('o','open'),('h','high'),('l','low'),('c','close'),('last','last')]: x[a]=d[b]*f
    x['h']=x[['o','h','l','c','last']].max(axis=1); x['l']=x[['o','h','l','c','last']].min(axis=1)
    x['value']=d.value; x['vol']=d.vol; x['ntr']=d.ntr; x['mcap']=d.mcap
    bp=d.bp.where(d.bp>0); bp=bp.fillna(d.pcb/d.pcs)
    x['bp']=bp.replace([np.inf,-np.inf],np.nan)
    x['net']=d.net; x['nrp']=(d.net/d.value).where(d.net.notna())
    x['pcb']=d.pcb; x['pcs']=d.pcs; x['float']=d['float']; x['pe']=d.pe
    x['jdate']=d.jdate; x['jy']=d.jdate.str[:4].astype(int)
    x['rawpct']=d.pct
    r=np.log(x.c).diff()
    bad=r.abs()>np.log(1.6)   # residual unadjusted gaps
    x['gap']=bad
    OUT[s]=x
pd.to_pickle((OUT,N),'panel.pkl')
print(len(OUT),'symbols kept')
g=[(s,int(x.gap.sum())) for s,x in OUT.items() if x.gap.sum()>0]; print('residual big gaps:',g)
