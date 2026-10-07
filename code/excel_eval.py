"""Accuracy of past daily workbooks: return and excess return over the market median, by signal and strength, at 1/3/5 sessions,
plus the 5%-before-5% outcome once 10 sessions have passed. Output: live_state/excel_eval.csv"""
import glob, os, numpy as np, pandas as pd, live
X, N = pd.read_pickle('panel.pkl')
C = pd.DataFrame({live.norm(k): x.c for k, x in X.items()}); C = C.loc[:, ~C.columns.duplicated()]
H = pd.DataFrame({live.norm(k): x.h for k, x in X.items()}); H = H.loc[:, ~H.columns.duplicated()]
L = pd.DataFrame({live.norm(k): x.l for k, x in X.items()}); L = L.loc[:, ~L.columns.duplicated()]
rows = []
for f in sorted(glob.glob('live_state/reports/signals_*.xlsx')):
    d0 = pd.Timestamp(os.path.basename(f)[8:18])
    if d0 not in C.index: continue
    R = pd.read_excel(f, sheet_name='سیگنال‌ها')
    idx = C.index[C.index >= d0]; base = C.loc[d0]
    for _, r in R.iterrows():
        s = r['نماد']
        if s not in C.columns or not base.get(s, np.nan) > 0: continue
        row = dict(day=d0.date().isoformat(), sym=s, signal=r['سیگنال'], tier=r.get('قدرت سیگنال') if isinstance(r.get('قدرت سیگنال'), str) else '', conf=r.get('اطمینان'))
        sgn = -1 if r['سیگنال'] == 'فروش' else 1
        for h in (1, 3, 5, 10):
            if len(idx) > h:
                ret = C.at[idx[h], s] / base[s] - 1; m = float((C.loc[idx[h]] / base - 1).replace([np.inf, -np.inf], np.nan).median())
                row[f'r{h}'] = ret; row[f'x{h}'] = sgn * (ret - m)
        fut = idx[1:11]; res = 'باز'
        for t in fut:
            up, dn = H.at[t, s] / base[s] - 1, 1 - L.at[t, s] / base[s]
            fav, adv = (dn, up) if sgn < 0 else (up, dn)
            if fav >= 0.05 and adv >= 0.05: res = 'هر دو'; break
            if fav >= 0.05: res = 'درست'; break
            if adv >= 0.05: res = 'غلط'; break
        if res == 'باز' and len(fut) == 10: res = 'هیچ‌کدام'
        row['tp5'] = res
        rows.append(row)
E = pd.DataFrame(rows); E.to_csv('live_state/excel_eval.csv', index=False)
agg = {f'r{h}': (f'r{h}', 'mean') for h in (1, 3, 5) if f'r{h}' in E} | {f'x{h}': (f'x{h}', 'mean') for h in (1, 3, 5) if f'x{h}' in E}
T = E.groupby(['day', 'signal', 'tier']).agg(n=('sym', 'size'), **agg, hit=('tp5', lambda s: (s == 'درست').sum()), miss=('tp5', lambda s: (s == 'غلط').sum()), open_=('tp5', lambda s: (s == 'باز').sum()))
pd.set_option('display.width', 250); print(T.round(4).to_string())
