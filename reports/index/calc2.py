import pandas as pd, numpy as np, json
d = pd.read_pickle('idx.pkl'); c = d.v.astype(float); J = d.j; I = pd.read_pickle('ind.pkl'); br = pd.read_pickle('breadth.pkl')
out = json.load(open('stats.json'))
# ---------- conditional forward returns (analogs)
fw = {h: c.shift(-h) / c - 1 for h in (20, 60, 120)}
mdd = {h: (c[::-1].rolling(h, min_periods=1).min()[::-1].shift(-1) / c - 1) for h in (60,)}
conds = {
 'همه روزها': pd.Series(True, index=c.index),
 'فاصله از MA200 بیش از ۶۰٪': (c / I.ma200 - 1) > 0.6,
 'RSI۱۴ بالای ۸۰': I.rsi > 80,
 'بازده ۶۰ روزه بیش از ۴۵٪': (c / c.shift(60) - 1) > 0.45,
 'سقف تاریخی (در ۵ روز اخیر)': c >= c.cummax().shift(0) * 0.999,
 'همهٔ شرایط امروز (MA200>60٪ و RSI>75 و بازده ۶۰روزه>45٪)': ((c / I.ma200 - 1) > 0.6) & (I.rsi > 75) & ((c / c.shift(60) - 1) > 0.45),
}
rows = []
for k, m in conds.items():
    m = m.fillna(False)
    row = dict(cond=k, n=int(m.sum()))
    for h, f in fw.items():
        v = f[m].dropna(); row[f'f{h}'] = float(v.median()) if len(v) else None; row[f'p{h}'] = float((v > 0).mean()) if len(v) else None
    v = mdd[60][m].dropna(); row['mdd60'] = float(v.median()) if len(v) else None
    # distinct episodes (gap >20 trading days)
    idx = np.where(m.values)[0]; row['episodes'] = int(1 + (np.diff(idx) > 20).sum()) if len(idx) else 0
    ep = []
    if len(idx):
        starts = [idx[0]] + [idx[i + 1] for i in np.where(np.diff(idx) > 20)[0]]
        ep = [J.iloc[s] for s in starts]
    row['ep_starts'] = ep
    rows.append(row)
A = pd.DataFrame(rows); out['analogs'] = rows; print(A.drop(columns='ep_starts').round(3).to_string()); print(A[['cond', 'ep_starts']].to_string())
# ---------- trend-following backtests (signal at close t, applied from t+1)
lr = np.log(c).diff().fillna(0)
def bt(sig, name):
    pos = sig.shift(1).fillna(0); r = pos * lr; eq = np.exp(r.cumsum())
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min(); tim = pos.mean(); trades = int((pos.diff().abs() > 0).sum() / 2)
    return dict(name=name, cagr=float(eq.iloc[-1] ** (1 / yrs) - 1), maxdd=float(dd), time_in=float(tim), trades=trades, eq=eq)
B = [bt(pd.Series(1.0, index=c.index), 'خرید و نگهداری'),
     bt((c > I.ma200).astype(float), 'بالای MA200'),
     bt((I.ma50 > I.ma200).astype(float), 'MA50 بالای MA200'),
     bt((c > I.ma100).astype(float), 'بالای MA100'),
     bt((c / c.shift(240) - 1 > 0).astype(float), 'مومنتوم ۱۲ماهه مثبت'),
     bt((I.macd > I.sig).astype(float), 'MACD بالای سیگنال'),
     bt((c > c.shift(1)).astype(float), 'روز قبل مثبت (مومنتوم روزانه)')]
pd.to_pickle({b['name']: b['eq'] for b in B}, 'bt_eq.pkl')
out['bt'] = [{k: v for k, v in b.items() if k != 'eq'} for b in B]; print(pd.DataFrame(out['bt']).round(3))
# daily autocorrelation decomposition: next-day return after up/down days by size
nx = lr.shift(-1)
bins = pd.cut(lr, [-1, -0.02, -0.01, 0, 0.01, 0.02, 1])
T = nx.groupby(bins).agg(['mean', lambda s: (s > 0).mean(), 'count']); T.columns = ['mean', 'pos', 'n']; print(T.round(4)); out['nextday'] = {str(k): [float(a), float(b), int(n)] for k, (a, b, n) in T.iterrows()}
# streaks
s = np.sign(lr); run = s.groupby((s != s.shift()).cumsum()).cumcount() + 1; cur = int(run.iloc[-1] * s.iloc[-1]); out['streak'] = cur
# ---------- long-term log-linear regression
t = np.arange(len(c)); y = np.log(c.values); b1, b0 = np.polyfit(t, y, 1); res = y - (b0 + b1 * t); sdv = res.std()
out['reg'] = dict(ann=float(np.exp(b1 * 240) - 1), z=float(res[-1] / sdv), fit=float(np.exp(b0 + b1 * t[-1])), up2=float(np.exp(b0 + b1 * t[-1] + 2 * sdv)), dn2=float(np.exp(b0 + b1 * t[-1] - 2 * sdv)))
pd.to_pickle(dict(b0=b0, b1=b1, sd=sdv), 'reg.pkl'); print(out['reg'])
# regression since 1402 low (cycle) 
# ---------- breadth / equal weight
b = br.reindex(c.index).dropna(subset=['EW'])
EWn = b.EW / b.EW.iloc[0]; Cn = c[b.index] / c[b.index].iloc[0]
cc = c[b.index]; rel = (b.EW / b.EW.iloc[-241]) / (cc / cc.iloc[-241]) - 1
out['ew_vs_cw_240'] = float(rel.iloc[-1])
for h in (20, 60, 120, 240):
    out[f'ew_ret{h}'] = float(b.EW.iloc[-1] / b.EW.iloc[-1 - h] - 1)
ad = (b.adv - b.dec).cumsum(); out['ad_now_vs_max'] = float(ad.iloc[-1] - ad.max()); out['ad_at_max'] = bool(ad.iloc[-1] >= ad.iloc[-20:].max())
pctup = (b.adv / b.n); out['pctup_10'] = float(pctup.rolling(10).mean().iloc[-1])
# % stocks above own MA50 needs per stock: compute
import sys
X, N = pd.read_pickle('/home/claude/multi/panel.pkl')
above = {}
for k, x in X.items():
    x = x[x.index >= '2015-01-01']
    if len(x) < 60: continue
    above[k] = (x.c > x.c.rolling(50).mean()).where(x.c.rolling(50).count() >= 50)
A50 = pd.DataFrame(above).mean(axis=1).reindex(c.index); A50.to_pickle('above50.pkl'); out['above50'] = float(A50.iloc[-1]); print('above50', out['above50'])
val = b.val; out['val20'] = float(val.iloc[-20:].mean()); out['val240'] = float(val.iloc[-240:].mean()); out['val_today'] = float(val.iloc[-1])
print({k: out[k] for k in ['ew_vs_cw_240', 'ew_ret20', 'ew_ret60', 'ew_ret240', 'ad_at_max', 'pctup_10', 'val20', 'val240', 'val_today', 'streak']})
json.dump(out, open('stats.json', 'w'), ensure_ascii=False, indent=1, default=str)
