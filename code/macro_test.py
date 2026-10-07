import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
M = pd.read_pickle('macro.pkl')
lg = np.log
P = pd.DataFrame(index=M.index)
for h in (5, 20, 60): P[f'usd_r{h}'] = lg(M.usd / M.usd.shift(h))
for h in (20, 60): P[f'nima_r{h}'] = lg(M.nima / M.nima.shift(h))
P['spread'] = M.usd / M.nima - 1; P['spread_d20'] = P.spread.diff(20)
P['usd_yoy'] = M.usd / M.usd.shift(240) - 1
P['real'] = M.rfavg - P.usd_yoy
P['rf'] = M.rfavg; P['rf_d20'] = M.rfavg.diff(20); P['rf_d60'] = M.rfavg.diff(60)
P['usd_val'] = lg(M.mcap_usd) - lg(M.mcap_usd).rolling(480, min_periods=240).mean()      # market cap in USD vs its 2y average
P['flow20'] = M.rnet.rolling(20).sum() / M.val.rolling(20).sum()
IDX = {'tedpix': 'شاخص کل', 'ewt': 'شاخص کل هم‌وزن', 'vwp': 'شاخص قیمت وزنی-ارزشی'}
res = []; oos = []
for ix, nmx in IDX.items():
    s = M[ix]
    P[f'{ix}_m20'] = lg(s / s.shift(20)); P[f'{ix}_m60'] = lg(s / s.shift(60))
    for h in (5, 20, 60):
        y = lg(s.shift(-h) / s)
        for v in ['usd_r5', 'usd_r20', 'usd_r60', 'nima_r20', 'nima_r60', 'spread', 'spread_d20', 'real', 'rf', 'rf_d20', 'rf_d60', 'usd_val', 'flow20']:
            d = pd.concat([y.rename('y'), P[v].rename('x'), P[f'{ix}_m20'].rename('m20'), P[f'{ix}_m60'].rename('m60')], axis=1).dropna()
            if len(d) < 300: continue
            ic = d[['y', 'x']].corr('spearman').iloc[0, 1]
            half = d.index[len(d) // 2]; ic1 = d[d.index < half][['y', 'x']].corr('spearman').iloc[0, 1]; ic2 = d[d.index >= half][['y', 'x']].corr('spearman').iloc[0, 1]
            X = sm.add_constant((d[['x', 'm20', 'm60']] - d[['x', 'm20', 'm60']].mean()) / d[['x', 'm20', 'm60']].std())
            m = sm.OLS(d.y, X).fit(cov_type='HAC', cov_kwds={'maxlags': h + 5})
            res.append(dict(idx=ix, h=h, var=v, ic=ic, ic_first=ic1, ic_second=ic2, t=m.tvalues['x'], beta=m.params['x'], n=len(d)))
            # OOS (expanding, sampled every h days, from 1400): regression y ~ x + own momentum vs y ~ own momentum only
            if h in (20, 60):
                idx_s = d.index[::h]; dd = d.loc[idx_s]
                start = dd.index.searchsorted(pd.Timestamp('2021-03-21'))
                e_full = []; e_base = []
                for i in range(start, len(dd)):
                    tr = dd.iloc[:max(i - int(np.ceil(h / h)), 30)]   # drop the overlapping last obs
                    te = dd.iloc[i]
                    b1 = sm.OLS(tr.y, sm.add_constant(tr[['x', 'm20', 'm60']])).fit(); b0 = sm.OLS(tr.y, sm.add_constant(tr[['m20', 'm60']])).fit()
                    p1 = b1.params['const'] + (b1.params[['x', 'm20', 'm60']] * te[['x', 'm20', 'm60']]).sum()
                    p0 = b0.params['const'] + (b0.params[['m20', 'm60']] * te[['m20', 'm60']]).sum()
                    e_full.append((te.y - p1) ** 2); e_base.append((te.y - p0) ** 2)
                oos.append(dict(idx=ix, h=h, var=v, r2_oos=1 - np.sum(e_full) / np.sum(e_base), n=len(e_full)))
R = pd.DataFrame(res); O = pd.DataFrame(oos)
R.to_csv('macro_ic.csv', index=False); O.to_csv('macro_oos.csv', index=False)
pd.set_option('display.width', 250)
T = R.pivot_table(index='var', columns=['idx', 'h'], values='t').round(1); print('HAC t-stats (controlling own momentum)'); print(T.to_string())
T2 = R.pivot_table(index='var', columns=['idx', 'h'], values='ic').round(2); print('Spearman IC'); print(T2.to_string())
S = R.assign(stable=np.sign(R.ic_first) == np.sign(R.ic_second)).pivot_table(index='var', columns=['idx', 'h'], values='stable'); print('sign stable across halves'); print(S.to_string())
print('OOS R2 vs momentum-only (positive = adds value)'); print(O.pivot_table(index='var', columns=['idx', 'h'], values='r2_oos').round(3).to_string())
