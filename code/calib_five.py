"""Out-of-sample calibration of the 5% model: train on 1400-1403, score 1404-1405, map raw probability -> realized hit rate."""
import pandas as pd, numpy as np, lightgbm as lgb, warnings, pickle; warnings.filterwarnings('ignore')
exec(open('five_feats.py').read())
P = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=5, verbose=-1, seed=5, num_threads=2)
out = {}
for side in ['yb', 'ys']:
    tr = F[(F.jy >= 1400) & (F.jy <= 1403) & F[side].notna()]; te = F[(F.jy >= 1404) & F[side].notna()].copy()
    m = lgb.train(P, lgb.Dataset(tr[FEAT], tr[side]), num_boost_round=250)
    te['p'] = m.predict(te[FEAT])
    q = np.unique(np.quantile(te.p, np.linspace(0, 1, 21)))
    te['bin'] = pd.cut(te.p, q, include_lowest=True)
    t = te.groupby('bin', observed=True).agg(p_lo=('p', 'min'), p_hi=('p', 'max'), p_mean=('p', 'mean'), hit=(side, 'mean'), n=(side, 'size')).reset_index(drop=True)
    t['hit_mono'] = np.maximum.accumulate(t.hit.values)   # monotone
    out[side] = dict(table=t, base=float(te[side].mean()))
    print(side, 'base', round(te[side].mean(), 3)); print(t.round(3).to_string())
pickle.dump(out, open('five_calib.pkl', 'wb'))
