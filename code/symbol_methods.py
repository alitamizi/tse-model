"""Blend each symbol's historical method accuracy with live outcomes (live evidence weighted 3x, history 0.5x)."""
import os, pandas as pd, numpy as np, live, methods as M
R = pd.read_pickle('bt.pkl'); R['n'] = R.nb + R.ns; R['k'] = R.nbc + R.nsc
R = R[R.year >= 1400]
H = R.groupby(['sym', 'method'])[['n', 'k']].sum().reset_index(); H['sym'] = H.sym.map(live.norm)
H['rule'] = 'تکنیکال-' + H.method
E = pd.read_csv(os.path.join(live.STATE, 'evaluation.csv')) if os.path.exists(os.path.join(live.STATE, 'evaluation.csv')) else pd.DataFrame(columns=['sym', 'rule', 'result'])
E = E[E.result.isin(['درست', 'غلط'])]
Lv = E.groupby(['sym', 'rule']).result.agg(n_live='size', k_live=lambda s: (s == 'درست').sum()).reset_index()
B = H.merge(Lv, on=['sym', 'rule'], how='outer').fillna({'n': 0, 'k': 0, 'n_live': 0, 'k_live': 0})
B['score'] = (0.5 * B.k + 3 * B.k_live + 1) / (0.5 * B.n + 3 * B.n_live + 2)
B['hist_hit'] = B.k / B.n.replace(0, np.nan); B['live_hit'] = B.k_live / B.n_live.replace(0, np.nan)
B = B.sort_values(['sym', 'score'], ascending=[True, False])
B[['sym', 'rule', 'n', 'hist_hit', 'n_live', 'live_hit', 'score']].to_csv(os.path.join(live.STATE, 'symbol_methods.csv'), index=False, float_format='%.3f')
print('rows', len(B), 'with live evidence', int((B.n_live > 0).sum()))
