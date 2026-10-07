import pandas as pd, numpy as np, lightgbm as lgb, warnings, os; warnings.filterwarnings('ignore')
os.chdir('/home/claude/multi')
exec(open('five_feats.py').read())
from sklearn.metrics import roc_auc_score
M = pd.read_pickle('/home/claude/idx/macro.pkl'); lg = np.log
Z = pd.DataFrame(index=M.index)
Z['usd_r5'] = lg(M.usd / M.usd.shift(5)); Z['usd_r20'] = lg(M.usd / M.usd.shift(20)); Z['usd_r60'] = lg(M.usd / M.usd.shift(60))
Z['nima_r20'] = lg(M.nima / M.nima.shift(20)); Z['spread'] = M.usd / M.nima - 1
Z['real'] = M.rfavg - (M.usd / M.usd.shift(240) - 1)
Z['ewt_m20'] = lg(M.ewt / M.ewt.shift(20)); Z['ted_m20'] = lg(M.tedpix / M.tedpix.shift(20))
Z = Z.reindex(pd.date_range(Z.index.min(), Z.index.max())).ffill()
G = F.join(Z, on='date')
P = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=5, verbose=-1, seed=5, num_threads=2)
MAC = list(Z.columns)
sets = {'base': FEAT, '+dollar': FEAT + ['usd_r5', 'usd_r20', 'usd_r60', 'nima_r20'], '+dollar+real+spread': FEAT + ['usd_r5', 'usd_r20', 'usd_r60', 'nima_r20', 'real', 'spread'], '+market': FEAT + ['ewt_m20', 'ted_m20'], '+all': FEAT + MAC}
for side in ['yb', 'ys']:
    tr = G[(G.jy >= 1400) & (G.jy <= 1403) & G[side].notna()]; te = G[(G.jy >= 1404) & G[side].notna()]
    for nm, fs in sets.items():
        m = lgb.train(P, lgb.Dataset(tr[fs], tr[side]), num_boost_round=250); p = m.predict(te[fs])
        top = te[side][p >= np.quantile(p, 0.9)].mean()
        # per-day top-10 hit rate (the way picks are used)
        tt = te.assign(p=p); d10 = tt.groupby('date').apply(lambda g: g.nlargest(10, 'p')[side].mean()).mean()
        print(side, f'{nm:22s}', 'AUC', round(roc_auc_score(te[side], p), 4), 'top-decile hit', round(top, 4), 'daily top10 hit', round(d10, 4))
