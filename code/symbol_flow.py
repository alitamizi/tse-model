"""Per-symbol flow-signal library (1400+): which flow signals worked on each symbol, buy and sell side.
Legal-side signals use legal net = -(real net): legal buying shows up as real money leaving."""
import pandas as pd, numpy as np, live
O = pd.read_pickle('five_oos.pkl')  # not used for labels; labels below
exec(open('five_feats.py').read().split("for col in ['qform_buy'")[0])   # builds F with yb/ys labels (5% in 10d)
F = F[F.jy >= 1400].copy()
F['legal'] = -F.nrp
SIG = {
 'جریان-قدرت‌خریدار-حقیقی-بالا': ('buy', F.bp >= 2),
 'جریان-ورود-پول-حقیقی': ('buy', F.nrp >= 0.10),
 'جریان-خرید-حقوقی': ('buy', F.legal >= 0.10),
 'جریان-خرید-حقوقی-در-افت': ('buy', (F.legal >= 0.10) & (F.r5 < 0)),
 'جریان-ارزش-بالا-روز-مثبت': ('buy', (F.vr5 >= 2) & (F.cpct > 0)),
 'جریان-صف-خرید-درون‌روز': ('buy', F.qform_buy == True),
 'جریان-قدرت‌خریدار-حقیقی-پایین': ('sell', F.bp <= 0.5),
 'جریان-خروج-پول-حقیقی': ('sell', F.nrp <= -0.10),
 'جریان-فروش-حقوقی': ('sell', F.legal <= -0.10),
 'جریان-فروش-حقوقی-در-رشد': ('sell', (F.legal <= -0.10) & (F.r5 > 0)),
 'جریان-ارزش-بالا-روز-منفی': ('sell', (F.vr5 >= 2) & (F.cpct < 0)),
 'جریان-صف-فروش-درون‌روز': ('sell', F.qform_sell == True),
}
rows = []
F = F.sort_values(['sym', 'date'])
for name, (side, m) in SIG.items():
    lab = 'yb' if side == 'buy' else 'ys'
    s = F[m.fillna(False) & F[lab].notna()][['sym', 'date', lab]]
    keep = []; last = {}
    for i, sy, d in zip(s.index, s.sym, s.date):
        if sy not in last or (d - last[sy]).days > 14: keep.append(i)
        last[sy] = d
    s = s.loc[keep]
    base = F[F[lab].notna()].groupby('sym')[lab].mean()
    g = s.groupby('sym')[lab].agg(n='size', k='sum').reset_index()
    g['rule'] = name; g['side'] = side; g['base'] = g.sym.map(base)
    rows.append(g)
G = pd.concat(rows); G['sym'] = G.sym.map(live.norm)
G['hit'] = G.k / G.n; G['lift'] = G.hit - G.base
G.to_csv(live.STATE + '/symbol_flow_hist.csv', index=False, float_format='%.3f')
pool = G.groupby('rule').apply(lambda g: pd.Series(dict(n=g.n.sum(), hit=g.k.sum() / g.n.sum(), base=(g.base * g.n).sum() / g.n.sum())), include_groups=False)
print(pool.round(3).to_string())
# how many symbols have each signal as a clearly useful one (n>=6, lift>=0.15)?
print((G[(G.n >= 6) & (G.lift >= 0.15)].groupby('rule').size()).to_string())
