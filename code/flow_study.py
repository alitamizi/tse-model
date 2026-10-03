import pandas as pd, numpy as np
X,N=pd.read_pickle('panel.pkl')
parts=[]
for s,x in X.items():
    x=x.copy(); c=x.c; n=len(x)
    cal=pd.Series(x.index,index=x.index).diff().dt.days.fillna(99)
    reopen=(cal>7)
    since_reopen=np.zeros(n,int); k=99
    for i,r in enumerate(reopen.values):
        k=0 if r else k+1; since_reopen[i]=k
    f=pd.DataFrame(index=x.index); f['sym']=s; f['jy']=x.jy; f['jdate']=x.jdate
    f['reopen5']=since_reopen<5
    pc=c.shift(1)
    f['cpct']=c/pc-1; f['lpct']=x['last']/pc-1; f['l_c']=x['last']/c-1          # last vs close
    f['hpct']=x.h/pc-1; f['opct']=x.o/pc-1; f['lo_pct']=x.l/pc-1
    v=x.value
    for k in [3,5,10,20]: f[f'vr{k}']=v/v.shift(1).rolling(k).mean()
    f['vr_prev']=f.vr5.shift(1)
    f['ntr_r']=x.ntr/x.ntr.shift(1).rolling(10).mean()
    f['bp']=x.bp; f['bp_prev5']=x.bp.shift(1).rolling(5).mean(); f['bp_jump']=x.bp/f.bp_prev5
    f['nrp']=x.nrp
    f['pcb_r']=x.pcb/x.pcb.shift(1).rolling(20).median()
    for k in [5,10,20,40,60]: f[f'r{k}']=c/c.shift(k)-1
    f['dma20']=c/c.rolling(20).mean()-1; f['dma50']=c/c.rolling(50).mean()-1
    f['dd60']=c/c.rolling(60).max()-1; f['up60']=c/c.rolling(60).min()-1
    lock=((x.h-x.l)/c<0.002)&(x.rawpct>0.01); f['locked']=lock
    f['lockdn']=((x.h-x.l)/c<0.002)&(x.rawpct<-0.01)
    # labels, tradable entry = first non-locked day in t+1..t+5, at close
    cv=c.values; lk=lock.values
    for H,T,S,name in [(20,0.10,None,'u10'),(20,0.15,None,'u15'),(30,0.15,None,'u15h30'),(30,0.10,0.07,'ft10'),(30,0.15,0.10,'ft15')]:
        y=np.full(n,np.nan)
        for i in range(n-1):
            d=None
            for j in range(i+1,min(n,i+6)):
                if not lk[j]: d=j;break
            if d is None: continue
            seg=cv[d+1:d+1+H]
            if len(seg)<H: continue
            e=cv[d]
            if S is None: y[i]=float(seg.max()>=e*(1+T))
            else:
                up=np.where(seg>=e*(1+T))[0]; dn=np.where(seg<=e*(1-S))[0]
                y[i]=float(len(up)>0 and (len(dn)==0 or up[0]<dn[0]))
        f[name]=y
    # sell side: fall >=10% within 20d from next close
    y=np.full(n,np.nan)
    for i in range(n-21):
        e=cv[i+1]; y[i]=float(cv[i+2:i+22].min()<=e*0.90)
    f['d10']=y
    g=x.gap.values.astype(float); fw=pd.Series(g[::-1]).rolling(32,min_periods=1).max().values[::-1]; bad=np.r_[fw[1:],0]>0  # gapmask
    for col in ['u10','u15','u15h30','ft10','ft15','d10']: f.loc[bad,col]=np.nan
    parts.append(f)
F=pd.concat(parts); F['date']=F.index
# market-wide daily price limit estimate: 97th pct of |lpct| rounded to 0.5%
lim=F.groupby('date').lpct.apply(lambda s:s.abs().quantile(0.97)); lim=(np.round(lim*200)/200).clip(0.02,0.10)
F['lim']=F.date.map(lim)
F['last_at_up']=F.lpct>=F.lim-0.0035; F['last_at_dn']=F.lpct<=-(F.lim-0.0035)
F['qform_buy']=F.last_at_up&(F.l_c>=0.01)&~F.locked     # buy queue formed intraday: last at limit, close below it
F['qform_sell']=F.last_at_dn&(F.l_c<=-0.01)&~F.lockdn
mk=F.groupby('date').agg(m_r20=('r20','mean'),m_br=('dma50',lambda s:(s>0).mean()),m_bp=('bp','median'),m_q=('qform_buy','mean'))
F=F.join(mk,on='date')
F.to_pickle('flowF.pkl'); print(F.shape); print(lim.groupby(F.drop_duplicates('date').set_index('date').jy).median() if False else lim.resample('YE').median().tail(12))
