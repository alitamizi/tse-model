"""Append a saved state/daily/<date>.csv session to the daily panel (used when the minute archive for that day is missing)."""
import sys, numpy as np, pandas as pd, live
day = sys.argv[1]; D = pd.read_csv(f'/home/claude/tse-model/state/daily/{day}.csv'); t = pd.Timestamp(day)
X, N = live.load_daily(); by = {live.norm(k): k for k in X}; add = 0
for r in D.itertuples(index=False):
    k = by.get(live.norm(r.sym))
    if k is None: continue
    x = X[k]
    if x.index[-1] >= t: continue
    row = {c: getattr(r, c) for c in x.columns if hasattr(r, c)}
    X[k] = pd.concat([x, pd.DataFrame([row], index=[t]).reindex(columns=x.columns)]); add += 1
pd.to_pickle((X, N), 'panel.pkl'); print(day, 'appended', add)
