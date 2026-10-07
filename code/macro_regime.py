"""Index-level macro overlay (tested 2026-10-07): free-dollar momentum over 20/60 days predicts the next ~20 sessions of
TEDPIX / equal-weight index (OOS R2 +3..5% over momentum-only, 1400-1405); real rate helps only the equal-weight index.
Rate level/changes, USD/Nima spread and adding macro features to the per-stock model did NOT help and are not used.
Keeps macro/macro_hist.pkl updated daily from the bot (dollar: price_dollar_rl, Nima: ice_transfer_usd_sell, indices.json)."""
import json, os, math, numpy as np, pandas as pd, statsmodels.api as sm
B = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(B, 'macro', 'macro_hist.pkl'); D = os.path.join(B, 'live_state', 'tse-data', 'data')
M = pd.read_pickle(H)
try:
    mj = json.load(open(os.path.join(D, 'macro.json')))['items']; ij = json.load(open(os.path.join(D, 'indices.json')))
    bo = [x for x in ij['بازارها'] if x['بازار'] == 'بورس'][0]
    day = pd.Timestamp(bo['زمان'][:10])
    if bo.get('شاخص کل', 0) > 0 and (day not in M.index):
        row = dict(usd=mj['price_dollar_rl']['قیمت'], nima=mj['ice_transfer_usd_sell']['قیمت'], tedpix=bo['شاخص کل'], ewt=bo['شاخص هم‌وزن'])
        M.loc[day, list(row)] = list(row.values()); M = M.sort_index(); M[['rf', 'rfavg']] = M[['rf', 'rfavg']].ffill()
        M.to_pickle(H)
except Exception as e:
    print('MACRO_UPDATE_ERROR', repr(e)[:200])
lg = np.log; out = {}
X = pd.DataFrame(index=M.index)
X['usd_r20'] = lg(M.usd / M.usd.shift(20)); X['usd_r60'] = lg(M.usd / M.usd.shift(60))
X['real'] = M.rfavg - (M.usd / M.usd.shift(240) - 1)
for ix, nm in [('tedpix', 'شاخص کل'), ('ewt', 'شاخص کل هم‌وزن')]:
    s = M[ix]; X['m20'] = lg(s / s.shift(20)); X['m60'] = lg(s / s.shift(60)); y = lg(s.shift(-20) / s)
    cols = ['usd_r20', 'usd_r60', 'm20', 'm60'] + (['real'] if ix == 'ewt' else [])
    d = pd.concat([y.rename('y'), X[cols]], axis=1)
    tr = d.dropna().iloc[::20]   # non-overlapping 20-day samples
    m = sm.OLS(tr.y, sm.add_constant(tr[cols])).fit()
    now = X[cols].iloc[-1]
    pred = float(m.params['const'] + (m.params[cols] * now).sum()); base = float(tr.y.mean())
    resid = float(np.std(m.resid))
    # honest reliability: expanding out-of-sample forecasts since 1400 on the same non-overlapping grid
    ok = []
    for i in range(tr.index.searchsorted(pd.Timestamp('2021-03-21')), len(tr)):
        mm = sm.OLS(tr.y.iloc[:i - 1], sm.add_constant(tr[cols].iloc[:i - 1])).fit(); pr = mm.params['const'] + (mm.params[cols] * tr[cols].iloc[i]).sum()
        ok.append((pr, tr.y.iloc[i]))
    ok = pd.DataFrame(ok, columns=['p', 'y']); hi = ok[ok.p > ok.p.quantile(0.66)]
    oos = dict(n=len(ok), hit_all=float((np.sign(ok.p) == np.sign(ok.y)).mean()), n_hi=len(hi), up_when_high=float((hi.y > 0).mean()), mean_when_high=float(np.exp(hi.y.mean()) - 1), mean_all=float(np.exp(ok.y.mean()) - 1))
    out[ix] = dict(oos=oos, name=nm, exp20=float(np.exp(pred) - 1), base20=float(np.exp(base) - 1), sd=resid)
out['usd'] = dict(last=float(M.usd.iloc[-1]), r20=float(np.exp(X.usd_r20.iloc[-1]) - 1), r60=float(np.exp(X.usd_r60.iloc[-1]) - 1),
                  pct20=float((X.usd_r20.dropna() < X.usd_r20.iloc[-1]).mean()), nima=float(M.nima.iloc[-1]), real=float(X.real.iloc[-1]) if pd.notna(X.real.iloc[-1]) else None,
                  rf_date=str(M.rfavg.last_valid_index().date()), date=str(M.index[-1].date()))
json.dump(out, open(os.path.join(B, 'live_state', 'macro_regime.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
