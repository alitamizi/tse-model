"""Evaluate intraday features on option underlyings. Run from /home/claude/multi after opt_intraday.py build."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250)
P = pd.read_pickle('live_state/opt_panel.pkl')
P['y60'] = P.p_h60 / P.p - 1; P['yc'] = P.p_end / P.p - 1
last = P.sort_values('T').groupby(['u', 'day']).tail(1).set_index(['u', 'day'])
days = sorted(P.day.unique()); nxt = dict(zip(days[:-1], days[1:]))
P['nday'] = P.day.map(nxt)
nr = (last.close_end / last.yday).rename('nr')    # next-day close-to-close, adjustment-safe
P = P.join(nr, on=['u', 'nday'])
P['yn'] = (P.close_end / P.p) * P.nr - 1
P['tradable'] = P.ob.abs() < 1
for y in ['y60', 'yc', 'yn']:
    P['x' + y[1:]] = P[y] - P.groupby(['day', 'T'])[y].transform('median')
P['cm_pm'] = P.call_m - P.put_m
P['oi_net'] = (P.oi_call - P.oi_put)
P['lag'] = P.call_m - P.r30          # option leads underlying?
F = ['ret_d', 'r10', 'r30', 'flow30', 'bp30', 'flow_d', 'ob', 'cp_val', 'call_m', 'put_m', 'cm_pm', 'lag']
P = P[P.tradable & (P.val30 > 0)]
P.to_pickle('live_state/opt_panel_eval.pkl')
SPLIT = '2026-10-28'
for name, S in [('train', P[P.day < SPLIT]), ('test', P[P.day >= SPLIT])]:
    print(f'== {name}: {S.day.nunique()} days, {len(S)} rows, {S.u.nunique()} underlyings')
    R = []
    for f in F:
        row = {'f': f}
        for y in ['x60', 'xc', 'xn', 'yn']:
            ic = S.groupby('day').apply(lambda g: g[f].corr(g[y], method='spearman'))
            row[y] = round(ic.mean(), 3); row[y + '_t'] = round(ic.mean() / ic.std() * np.sqrt(ic.notna().sum()), 1)
        R.append(row)
    print(pd.DataFrame(R).to_string(index=False))
