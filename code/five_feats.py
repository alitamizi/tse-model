import pandas as pd,numpy as np,lightgbm as lgb,warnings;warnings.filterwarnings('ignore')
X,N=pd.read_pickle('panel.pkl')
H=10; T=0.05
lab=[]
for s,x in X.items():
    c=x.c.values; n=len(c); lk=(((x.h-x.l)/x.c<0.002)&(x.rawpct>0.01)).values; ld=(((x.h-x.l)/x.c<0.002)&(x.rawpct<-0.01)).values; gp=x.gap.values
    yb=np.full(n,np.nan); ys=np.full(n,np.nan); rb=np.full(n,np.nan)
    for i in range(n-H-2):
        if gp[i+1:i+H+2].any(): continue
        e=c[i+1]; seg=c[i+2:i+2+H]
        up=np.where(seg>=e*(1+T))[0]; dn=np.where(seg<=e*(1-T))[0]
        fu=up[0] if len(up) else 99; fd=dn[0] if len(dn) else 99
        if not lk[i+1]: yb[i]=float(fu<fd and fu<99)
        if not ld[i+1]: ys[i]=float(fd<fu and fd<99)
        rb[i]=seg[-1]/e-1
    lab.append(pd.DataFrame({'sym':s,'date':x.index,'yb':yb,'ys':ys,'fr10':rb}))
L=pd.concat(lab)
F=pd.read_pickle('flowF.pkl').reset_index(drop=True)
F=F[(~F.reopen5)&(F.jy>=1387)].merge(L,on=['sym','date'],how='inner').sort_values(['sym','date'])
for col in ['qform_buy','locked','last_at_up','qform_sell','lockdn','last_at_dn']:
    F[col+'_3']=F.groupby('sym')[col].transform(lambda s:s.astype(float).rolling(3).sum())
DROP=['sym','jy','jdate','date','reopen5','u10','u15','u15h30','ft10','ft15','d10','yb','ys','fr10']
FEAT=[c for c in F.columns if c not in DROP]
for c in FEAT: F[c]=F[c].astype('float32')
