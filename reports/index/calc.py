import pandas as pd, numpy as np, json, sys
sys.path.insert(0, '/home/claude/multi')
d = pd.read_pickle('idx.pkl'); c = d.v.astype(float); J = d.j
lr = np.log(c).diff()
out = {}
# --- market breadth / equal-weight / value from per-stock panel
X, N = pd.read_pickle('/home/claude/multi/panel.pkl')
R = {}; V = {}
for k, x in X.items():
    x = x[x.index >= '2014-12-01']
    if len(x) < 30: continue
    r = x.c.pct_change()
    if 'gap' in x: r[x.gap.fillna(False).astype(bool)] = np.nan
    r[(r.abs() > 0.12)] = np.nan          # capital-raise / adjustment days
    R[k] = r; V[k] = x.value
R = pd.DataFrame(R); V = pd.DataFrame(V)
ew = R.mean(axis=1).fillna(0); EW = (1 + ew).cumprod()
adv = (R > 0).sum(axis=1); dec = (R < 0).sum(axis=1); ntr = R.notna().sum(axis=1)
br = pd.DataFrame(dict(ew=ew, EW=EW, adv=adv, dec=dec, n=ntr, val=V.sum(axis=1)))
br = br[br.n > 50]
br.to_pickle('breadth.pkl')
print('breadth', br.index.min(), br.index.max(), len(br))
# --- basic indicators on the index
I = pd.DataFrame({'c': c})
for n in (20, 50, 100, 200): I[f'ma{n}'] = c.rolling(n).mean()
I['ema12'] = c.ewm(span=12, adjust=False).mean(); I['ema26'] = c.ewm(span=26, adjust=False).mean()
I['macd'] = I.ema12 - I.ema26; I['sig'] = I.macd.ewm(span=9, adjust=False).mean(); I['hist'] = I.macd - I.sig
dlt = c.diff(); up = dlt.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); dn = (-dlt.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
I['rsi'] = 100 - 100 / (1 + up / dn)
I['bbm'] = c.rolling(20).mean(); sd = c.rolling(20).std(); I['bbu'] = I.bbm + 2 * sd; I['bbl'] = I.bbm - 2 * sd; I['bw'] = (I.bbu - I.bbl) / I.bbm
I['vol20'] = lr.rolling(20).std() * np.sqrt(240); I['vol60'] = lr.rolling(60).std() * np.sqrt(240)
I['dd'] = c / c.cummax() - 1
I['lr'] = lr
I.to_pickle('ind.pkl')
L = I.iloc[-1]
out.update(last=float(L.c), date=str(J.iloc[-1]), ma50=float(L.ma50), ma100=float(L.ma100), ma200=float(L.ma200), rsi=float(L.rsi), macd=float(L.macd), sig=float(L.sig),
           bbu=float(L.bbu), bbl=float(L.bbl), bw=float(L.bw), bw_pct=float((I.bw.dropna() < L.bw).mean()), vol20=float(L.vol20), vol60=float(L.vol60), dd=float(L.dd),
           dist200=float(L.c / L.ma200 - 1), dist50=float(L.c / L.ma50 - 1))
# distribution of distance from MA200 historically
dist = (I.c / I.ma200 - 1).dropna(); out['dist200_pct'] = float((dist < out['dist200']).mean()); out['dist200_max'] = float(dist.max()); out['dist200_max_date'] = str(J[dist.idxmax()])
# rsi history extremes
out['rsi_pct'] = float((I.rsi.dropna() < L.rsi).mean())
# --- returns: year table, CAGR
yrs = c.groupby(J.str[:4]).last(); out['yearly'] = {k: float(v) for k, v in (yrs / yrs.shift() - 1).dropna().items()}
first1395 = c[J.str[:4] == '1394'].iloc[-1]
out['cagr'] = float((c.iloc[-1] / c.iloc[0]) ** (365.25 / (c.index[-1] - c.index[0]).days) - 1)
out['ytd1405'] = float(c.iloc[-1] / c[J < '1405'].iloc[-1] - 1)
for h in (5, 20, 60, 120, 240): out[f'ret{h}'] = float(c.iloc[-1] / c.iloc[-1 - h] - 1)
# --- drawdowns episodes
peak = c.cummax(); dd = c / peak - 1
eps = []; in_dd = False
for t in range(len(c)):
    if dd.iloc[t] < 0 and not in_dd: in_dd = True; s = t - 1
    if in_dd and dd.iloc[t] == 0:
        seg = dd.iloc[s:t + 1]; tr = seg.idxmin()
        eps.append(dict(start=J.iloc[s], trough=J[tr], end=J.iloc[t], depth=float(seg.min()), days_down=int(c.index.get_loc(tr) - s), days_total=int(t - s)))
        in_dd = False
if in_dd:
    seg = dd.iloc[s:]; tr = seg.idxmin(); eps.append(dict(start=J.iloc[s], trough=J[tr], end='ادامه دارد', depth=float(seg.min()), days_down=int(c.index.get_loc(tr) - s), days_total=int(len(c) - 1 - s)))
E = pd.DataFrame(eps).sort_values('depth').head(8); out['dd_eps'] = E.to_dict('records')
# --- distribution stats
out['skew'] = float(lr.skew()); out['kurt'] = float(lr.kurt()); out['pos_days'] = float((lr > 0).mean())
out['ac1'] = float(lr.autocorr(1)); out['ac2'] = float(lr.autocorr(2)); out['ac5'] = float(lr.autocorr(5))
# weekly
w = c.resample('W-WED').last().dropna(); wl = np.log(w).diff(); out['wac1'] = float(wl.autocorr(1))
# seasonality: Persian month average return
m = c.groupby(J.str[:7]).last(); mr = (m / m.shift() - 1).dropna(); mm = mr.groupby(mr.index.str[5:7]).agg(['mean', 'median', lambda s: (s > 0).mean(), 'count'])
mm.columns = ['mean', 'median', 'pos', 'n']; out['months'] = mm.round(4).to_dict('index')
# day of week
dow = lr.groupby(lr.index.dayofweek).agg(['mean', lambda s: (s > 0).mean(), 'count']); out['dow'] = {int(k): [float(a), float(b), int(n)] for k, (a, b, n) in dow.iterrows()}
json.dump(out, open('stats.json', 'w'), ensure_ascii=False, indent=1, default=str)
for k, v in out.items():
    if k not in ('months', 'dd_eps', 'yearly', 'dow'): print(k, v)
print(pd.DataFrame(out['dd_eps']).to_string()); print(mm.round(3)); print(dow.round(4))
