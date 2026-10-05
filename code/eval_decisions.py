"""Score every logged 10-minute decision against later daily closes and keep a running accuracy history.
Output: live_state/decision_eval.csv (one row per first occurrence of sym+rule per day)
        live_state/metrics_daily.csv (accuracy by rule/decision per signal day; the learning curve)"""
import pandas as pd, numpy as np, live, os
LOG = 'live_state/decisions_10m.csv'
if not os.path.exists(LOG): raise SystemExit('no log')
D = pd.read_csv(LOG)
D = D.sort_values(['day', 'time']).drop_duplicates(['day', 'sym', 'rule', 'decision'], keep='first')
X, N = pd.read_pickle('panel.pkl'); by = {live.norm(k): k for k in X}
out = []
for r in D.itertuples():
    k = by.get(r.sym)
    if k is None or not r.last > 0: continue
    x = X[k]; d0 = pd.Timestamp(r.day)
    sell = str(r.strength).startswith('فروش')
    same = x.c[x.index == d0]; fut = x.c[x.index > d0]
    row = dict(day=r.day, time=r.time, sym=r.sym, rule=r.rule, decision=r.decision, side='sell' if sell else 'buy', entry=r.last,
               r_close=same.iloc[0] / r.last - 1 if len(same) else np.nan)
    # multi-day outcome: within 10 sessions, does price move 5% in the signal's direction before 5% against it?
    fx = x[x.index > d0].iloc[:10]; res = 'باز'
    for _, b in fx.iterrows():
        hi, lo = b.h / r.last - 1, b.l / r.last - 1
        fav, adv = (-lo, hi) if sell else (hi, -lo)
        if fav >= 0.05 and adv >= 0.05: res = 'هر دو'; break
        if fav >= 0.05: res = 'درست'; break
        if adv >= 0.05: res = 'غلط'; break
    row['tp5_10d'] = res
    row['mfe10'] = (( -fx.l.min() / r.last + 1) if sell else (fx.h.max() / r.last - 1)) if len(fx) else np.nan
    row['mae10'] = ((fx.h.max() / r.last - 1) if sell else (1 - fx.l.min() / r.last)) if len(fx) else np.nan
    for h in (1, 3, 5, 10):
        row[f'r{h}'] = fut.iloc[h - 1] / r.last - 1 if len(fut) >= h else np.nan
    out.append(row)
E = pd.DataFrame(out)
sgn = np.where(E.side == 'sell', -1, 1)
for c in ['r_close', 'r1', 'r3', 'r5', 'r10']:
    E['ok_' + c] = np.where(E[c].isna(), np.nan, (E[c] * sgn > 0).astype(float))
    E['g_' + c] = E[c] * sgn          # signed gain in the signal's direction
E.to_csv('live_state/decision_eval.csv', index=False)
E['dec'] = E.decision.str.split(':').str[0]
M = E.groupby(['day', 'rule', 'dec']).agg(n=('sym', 'size'), **{f'ok_{c}': (f'ok_{c}', 'mean') for c in ['r_close', 'r1', 'r3', 'r5']},
                                         **{f'g_{c}': (f'g_{c}', 'mean') for c in ['r1', 'r3', 'r5']}).reset_index()
M.to_csv('live_state/metrics_daily.csv', index=False)
pd.set_option('display.width', 250)
T = E.groupby(['rule', 'dec']).agg(n=('sym', 'size'), ok1=('ok_r1', 'mean'), g1=('g_r1', 'mean'), ok3=('ok_r3', 'mean'), g3=('g_r3', 'mean'), ok5=('ok_r5', 'mean'), g5=('g_r5', 'mean'),
                                   tp_ok=('tp5_10d', lambda s: (s == 'درست').sum()), tp_bad=('tp5_10d', lambda s: (s == 'غلط').sum()), tp_open=('tp5_10d', lambda s: (s == 'باز').sum()))
print(T.round(3).to_string())
