import pandas as pd, numpy as np, json
d = pd.read_pickle('idx.pkl'); c = d.v.astype(float); J = d.j; I = pd.read_pickle('ind.pkl')
out = json.load(open('stats.json'))
def zigzag(s, th):
    piv = [(s.index[0], s.iloc[0], 0)]; trend = 0; ext_i, ext_v = s.index[0], s.iloc[0]
    for t, v in s.items():
        if trend >= 0 and v >= ext_v and trend == 1 or (trend == 0 and v > ext_v):
            ext_i, ext_v = t, v; trend = 1 if trend == 0 and v > piv[0][1] * (1 + th) else trend
        if trend == 1:
            if v > ext_v: ext_i, ext_v = t, v
            elif v < ext_v * (1 - th): piv.append((ext_i, ext_v, 1)); trend = -1; ext_i, ext_v = t, v
        elif trend == -1:
            if v < ext_v: ext_i, ext_v = t, v
            elif v > ext_v * (1 + th): piv.append((ext_i, ext_v, -1)); trend = 1; ext_i, ext_v = t, v
        else:
            if v < piv[0][1] * (1 - th): trend = -1; ext_i, ext_v = t, v
            elif v > piv[0][1] * (1 + th): trend = 1; ext_i, ext_v = t, v
    piv.append((ext_i, ext_v, trend))
    return pd.DataFrame(piv, columns=['t', 'v', 'type'])
Z = zigzag(c, 0.12); Z['j'] = Z.t.map(J); Z['chg'] = Z.v.pct_change(); Z['days'] = Z.t.map(lambda t: c.index.get_loc(t)).diff()
print(Z.to_string()); Z.to_pickle('zz12.pkl')
Z7 = zigzag(c[c.index >= '2024-09-01'], 0.07); Z7['j'] = Z7.t.map(J); Z7['chg'] = Z7.v.pct_change(); print(Z7.to_string()); Z7.to_pickle('zz7.pkl')
out['zz12'] = Z[['j', 'v', 'type', 'chg', 'days']].to_dict('records')
# Fibonacci on the last major swing (low of the latest big leg -> current high)
lowi = Z[Z.type == -1].iloc[-1]; hi = c.max(); lo = lowi.v
fib = {f'{r:.3f}': float(hi - (hi - lo) * r) for r in (0.236, 0.382, 0.5, 0.618, 0.786)}
ext = {f'{r:.3f}': float(lo + (hi - lo) * r) for r in (1.272, 1.618)}
# extension of previous up-leg projected from last pullback (if any) using 7% zigzag
out['fib'] = dict(low=float(lo), low_j=lowi.j, high=float(hi), retr=fib, ext=ext); print(out['fib'])
# weekly
w = c.resample('W-WED').last().dropna()
dl = w.diff(); up = dl.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); dn = (-dl.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean(); wr = 100 - 100 / (1 + up / dn)
wm = w.ewm(span=12, adjust=False).mean() - w.ewm(span=26, adjust=False).mean(); ws = wm.ewm(span=9, adjust=False).mean()
pd.DataFrame(dict(w=w, rsi=wr, macd=wm, sig=ws)).to_pickle('weekly.pkl')
out['wrsi'] = float(wr.iloc[-1]); out['wrsi_max'] = float(wr.max()); out['wrsi_max_date'] = str(wr.idxmax().date()); print('weekly rsi', wr.iloc[-1], wr.max(), wr.idxmax())
# daily RSI divergence: compare last two price peaks (5% zigzag) with RSI
Z5 = zigzag(c[c.index >= '2025-12-01'], 0.04); pk = Z5[Z5.type == 1]
divs = []
for i in range(1, len(pk)):
    a, b = pk.iloc[i - 1], pk.iloc[i]
    ra, rb = I.rsi[a.t], I.rsi[b.t]; ma, mb = I.macd[a.t], I.macd[b.t]
    divs.append(dict(p1=J[a.t], p2=J[b.t], v1=float(a.v), v2=float(b.v), rsi1=float(ra), rsi2=float(rb), macd1=float(ma), macd2=float(mb)))
# current vs last peak
if len(pk):
    a = pk.iloc[-1]; divs.append(dict(p1=J[a.t], p2=J.iloc[-1] + ' (امروز)', v1=float(a.v), v2=float(c.iloc[-1]), rsi1=float(I.rsi[a.t]), rsi2=float(I.rsi.iloc[-1]), macd1=float(I.macd[a.t]), macd2=float(I.macd.iloc[-1])))
out['divs'] = divs; print(pd.DataFrame(divs).round(1).to_string())
# Ichimoku (close-only approximation: high=low=close)
hh = lambda n: c.rolling(n).max(); ll = lambda n: c.rolling(n).min()
ten = (hh(9) + ll(9)) / 2; kij = (hh(26) + ll(26)) / 2; sa = ((ten + kij) / 2).shift(26); sb = ((hh(52) + ll(52)) / 2).shift(26)
ich = pd.DataFrame(dict(c=c, ten=ten, kij=kij, sa=sa, sb=sb)); ich.to_pickle('ichi.pkl')
# future cloud (26 ahead)
fsa = ((ten + kij) / 2).iloc[-26:].values; fsb = ((hh(52) + ll(52)) / 2).iloc[-26:].values
out['ichi'] = dict(ten=float(ten.iloc[-1]), kij=float(kij.iloc[-1]), sa=float(sa.iloc[-1]), sb=float(sb.iloc[-1]), fsa=float(fsa[-1]), fsb=float(fsb[-1]))
print(out['ichi'])
# 1399 analog: align current leg start (1405-03-20 restart point, trough after closure) vs 1399 rally start
# find bottom-to-now leg: last 12% zigzag low
start_now = lowi.t
prev_low = Z[(Z.type == -1) & (Z.j < '1399') & (Z.j > '1398')]
print('prev lows 1398', prev_low)
json.dump(out, open('stats.json', 'w'), ensure_ascii=False, indent=1, default=str)
