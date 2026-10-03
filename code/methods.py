import numpy as np, pandas as pd
COST=0.0125
NAMES={'sma':'میانگین متحرک ۲۰/۵۰','macd':'مکدی (۱۲،۲۶،۹)','rsi':'RSI برگشت از ۳۰','boll':'بولینگر برگشت از باند پایین',
'donch':'شکست کانال دانچیان (۲۰/۱۰)','ichi':'ایچیموکو','super':'سوپرترند (۱۰،۳)','psar':'پارابولیک سار','stoch':'استوکاستیک',
'adx':'ADX / DMI','fibo':'فیبوناچی (بازگشت ۳۸–۶۲٪ روی زیگزاگ)','flow_bp':'قدرت خریدار ≥ ۲ + ارزش بالا','flow_net':'ورود پول حقیقی ≥ ۱۵٪',
'volbrk':'شکست سقف ۲۰روزه با ارزش ۲ برابر','trendflow':'روند صعودی + قدرت خریدار ≥ ۱٫۵','obv':'OBV + میانگین ۲۰'}
def ema(a,n): return pd.Series(a).ewm(span=n,adjust=False).mean().values
def sma(a,n): return pd.Series(a).rolling(n).mean().values
def rsi(c,n=14):
    d=np.diff(c,prepend=c[0]); up=pd.Series(np.maximum(d,0)).ewm(alpha=1/n,adjust=False).mean(); dn=pd.Series(np.maximum(-d,0)).ewm(alpha=1/n,adjust=False).mean()
    return (100-100/(1+up/dn.replace(0,np.nan))).fillna(50).values
def atr(h,l,c,n):
    pc=np.r_[c[0],c[:-1]]; tr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values
def roll_max(a,n): return pd.Series(a).rolling(n).max().values
def roll_min(a,n): return pd.Series(a).rolling(n).min().values
def cross_up(a,b): return (a>b)&(np.r_[np.nan,a[:-1]]<=np.r_[np.nan,b[:-1]] if np.ndim(b) else (a>b)&(np.r_[np.nan,a[:-1]]<=b))
def cup(a,b):
    b=np.broadcast_to(b,a.shape); pa=np.r_[np.nan,a[:-1]]; pb=np.r_[np.nan,b[:-1]]; return (a>b)&(pa<=pb)
def cdn(a,b):
    b=np.broadcast_to(b,a.shape); pa=np.r_[np.nan,a[:-1]]; pb=np.r_[np.nan,b[:-1]]; return (a<b)&(pa>=pb)

def signals(x):
    """returns dict method -> (entry, exit, hold_max, stop) arrays computed with data up to each day only"""
    c=x.c.values; h=x.h.values; l=x.l.values; v=x.value.values.astype(float)
    bp=x.bp.values.astype(float); nrp=x.nrp.values.astype(float); n=len(c); S={}
    s20,s50=sma(c,20),sma(c,50); S['sma']=(cup(s20,s50),cdn(s20,s50),0,0)
    m=ema(c,12)-ema(c,26); sg=ema(m,9); S['macd']=(cup(m,sg),cdn(m,sg),0,0)
    r=rsi(c); S['rsi']=(cup(r,30),cup(r,70),0,0.10)
    mid=sma(c,20); sd=pd.Series(c).rolling(20).std().values; lo=mid-2*sd; up=mid+2*sd
    S['boll']=(cup(c,lo),c>=up,0,0.10)
    S['donch']=(c>np.r_[np.nan,roll_max(h,20)[:-1]], c<np.r_[np.nan,roll_min(l,10)[:-1]],0,0)
    tk=(roll_max(h,9)+roll_min(l,9))/2; kj=(roll_max(h,26)+roll_min(l,26))/2
    sa=np.r_[np.full(26,np.nan),((tk+kj)/2)[:-26]]; sb=np.r_[np.full(26,np.nan),((roll_max(h,52)+roll_min(l,52))/2)[:-26]]
    cloud=np.fmax(sa,sb); ok=(c>cloud)&(tk>kj); S['ichi']=(ok&~np.r_[False,ok[:-1]], c<kj,0,0)
    # supertrend
    a=atr(h,l,c,10); hl=(h+l)/2; ub=hl+3*a; lb=hl-3*a; fu=ub.copy(); fl=lb.copy(); tr=np.ones(n,bool)
    for i in range(1,n):
        fu[i]=ub[i] if (ub[i]<fu[i-1] or c[i-1]>fu[i-1]) else fu[i-1]
        fl[i]=lb[i] if (lb[i]>fl[i-1] or c[i-1]<fl[i-1]) else fl[i-1]
        tr[i]=True if c[i]>fu[i-1] else (False if c[i]<fl[i-1] else tr[i-1])
    S['super']=(tr&~np.r_[True,tr[:-1]], ~tr&np.r_[True,tr[:-1]],0,0)
    # psar
    bull=True; af=.02; ep=h[0]; sar=l[0]; L=np.ones(n,bool)
    for i in range(1,n):
        sar=sar+af*(ep-sar)
        if bull:
            sar=min(sar,l[i-1],l[i-2] if i>1 else l[i-1])
            if l[i]<sar: bull=False; sar=ep; ep=l[i]; af=.02
            elif h[i]>ep: ep=h[i]; af=min(af+.02,.2)
        else:
            sar=max(sar,h[i-1],h[i-2] if i>1 else h[i-1])
            if h[i]>sar: bull=True; sar=ep; ep=h[i]; af=.02
            elif l[i]<ep: ep=l[i]; af=min(af+.02,.2)
        L[i]=bull
    S['psar']=(L&~np.r_[True,L[:-1]], ~L&np.r_[True,L[:-1]],0,0)
    k=100*(c-roll_min(l,14))/(roll_max(h,14)-roll_min(l,14)+1e-12); k=sma(k,3); dd=sma(k,3)
    S['stoch']=(cup(k,dd)&(np.r_[np.nan,k[:-1]]<20), cdn(k,dd)&(np.r_[np.nan,k[:-1]]>80),0,0.10)
    pc=np.r_[c[0],c[:-1]]; ph=np.r_[h[0],h[:-1]]; pl=np.r_[l[0],l[:-1]]
    upm=h-ph; dnm=pl-l; pdm=np.where((upm>dnm)&(upm>0),upm,0); ndm=np.where((dnm>upm)&(dnm>0),dnm,0)
    a14=atr(h,l,c,14); pdi=100*pd.Series(pdm).ewm(alpha=1/14,adjust=False).mean().values/a14; ndi=100*pd.Series(ndm).ewm(alpha=1/14,adjust=False).mean().values/a14
    dx=100*abs(pdi-ndi)/(pdi+ndi+1e-12); adx=pd.Series(dx).ewm(alpha=1/14,adjust=False).mean().values
    ok=(pdi>ndi)&(adx>20); S['adx']=(ok&~np.r_[False,ok[:-1]], pdi<ndi,0,0)
    # fibonacci on online zigzag (10%)
    en=np.zeros(n,bool); ex=np.zeros(n,bool); th=0.10
    lowp=c[0]; highp=c[0]; direction=0; lastL=None; lastH=None; pos=False; stop=0; tgt=0
    for i in range(1,n):
        p=c[i]
        if direction>=0:
            if p>highp: highp=p
            if p<highp*(1-th): lastH=highp; direction=-1; lowp=p
        if direction<=0 and not (direction==-1 and p==lowp and lastH==highp and i and False):
            if direction==-1:
                if p<lowp: lowp=p
                if p>lowp*(1+th): lastL=lowp; direction=1; highp=p
            elif direction==0 and p<lowp: lowp=p
        if not pos and direction==-1 and lastH is not None and lastL is not None and lastH>lastL*1.15:
            rng=lastH-lastL; z_hi=lastH-0.382*rng; z_lo=lastH-0.618*rng
            if z_lo<=l[i]<=z_hi and p>h[i-1] and p>z_lo:
                en[i]=True; pos=True; stop=lastH-0.786*rng; tgt=lastH
        elif pos:
            if p<stop or p>=tgt: ex[i]=True; pos=False
    S['fibo']=(en,ex,0,0)
    vm=pd.Series(v).rolling(20).median().values; vr=v/vm
    S['flow_bp']=((bp>=2)&(vr>=1),np.zeros(n,bool),5,0)
    S['flow_net']=((nrp>=0.15)&(vr>=1.5),np.zeros(n,bool),5,0)
    S['volbrk']=((c>=roll_max(c,20))&(vr>=2),c<s20,0,0)
    s50p=np.r_[np.full(10,np.nan),s50[:-10]]
    S['trendflow']=((c>s50)&(s50>s50p)&(bp>=1.5),c<s20,0,0)
    sgn=np.sign(np.diff(c,prepend=c[0])); obv=np.cumsum(sgn*x.vol.values.astype(float)); oe=ema(obv,20)
    ok=(obv>oe)&(c>s20); S['obv']=(ok&~np.r_[False,ok[:-1]], (obv<oe)&(c<s20),0,0)
    return S

def run_year(c, idx, entry, exit_, hold, stop):
    """trade within index range idx (array of positions). signal at t, fill at close t+1. flat at start, forced exit at last day."""
    trades=[]; pos=False; i0=idx[0]; i1=idx[-1]
    t=i0
    while t<i1:
        if not pos:
            if entry[t] and not np.isnan(c[t]):
                e=t+1; pos=True; ep=c[e]; t=e; continue
        else:
            done=(exit_[t] and t>e) or (hold and t-e>=hold) or (stop and c[t]<ep*(1-stop))
            if done:
                trades.append((e,t+1)); pos=False; t=t+1; continue
        t+=1
    if pos: trades.append((e,i1))
    return trades

def evaluate_trades(c, trades, i_end, next_entries):
    """buy correct if exit>entry; sell correct if price at next buy (or year end) < exit price"""
    nb=nbc=ns=nsc=0; ret=1.0; exp=0
    for k,(e,x) in enumerate(trades):
        nb+=1; nbc+= c[x]>c[e]; ret*= (c[x]/c[e])*(1-COST); exp+=x-e
        if x<i_end:
            ns+=1; nxt=trades[k+1][0] if k+1<len(trades) else i_end
            nsc+= c[nxt]<c[x]
    return nb,nbc,ns,nsc,ret-1,exp
