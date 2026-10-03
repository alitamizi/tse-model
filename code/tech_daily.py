"""Daily (after close): state of all mechanical technical methods per symbol + divergences."""
import os, numpy as np, pandas as pd, methods as M, live
from concurrent.futures import ProcessPoolExecutor
X, N = live.load_daily()
def one(item):
    s, x = item
    if len(x) < 120 or x.index[-1] < max_day - pd.Timedelta(days=10): return None
    x = x.iloc[-400:].copy()
    S = M.signals(x); c = x.c.values; n = len(c)
    st = {}
    for m, (en, ex, hold, stop) in S.items():
        en = np.nan_to_num(en).astype(bool); ex = np.nan_to_num(ex).astype(bool)
        pos = False; e = 0; since = None
        for i in range(n):
            if not pos and en[i]: pos = True; e = i
            elif pos and ((ex[i] and i > e) or (hold and i - e >= hold) or (stop and c[i] < c[e] * (1 - stop))): pos = False
        st[m] = 'buy_today' if en[-1] else ('sell_today' if ex[-1] else ('long' if pos else 'flat'))
    row = dict(sym=s, date=x.index[-1].date(), n_long=sum(v in ('long', 'buy_today') for v in st.values()),
               buy_today=','.join(k for k, v in st.items() if v == 'buy_today'), sell_today=','.join(k for k, v in st.items() if v == 'sell_today'))
    row.update({f'm_{k}': v for k, v in st.items()})
    # divergences on daily closes over the last ~60 days, using swing points (local extremes, window 5)
    r = M.rsi(c); obv = np.cumsum(np.sign(np.diff(c, prepend=c[0])) * x.vol.values.astype(float))
    net = np.nan_to_num(x.net.values.astype(float)); cnet = np.cumsum(net)
    m = pd.Series(M.ema(c, 12) - M.ema(c, 26)).values
    def swings(arr, kind, look=60, w=5):
        idx = []
        for i in range(max(w, n - look), n - 1):
            seg = arr[max(0, i - w):min(n, i + w + 1)]
            if (kind == 'low' and arr[i] == seg.min()) or (kind == 'high' and arr[i] == seg.max()): idx.append(i)
        # include the last bar as a provisional swing if it is the extreme of the last w bars
        if (kind == 'low' and c[-1] == c[-w:].min()) or (kind == 'high' and c[-1] == c[-w:].max()): idx.append(n - 1)
        return idx
    div = []
    for kind in ['low', 'high']:
        sw = swings(c, kind)
        if len(sw) >= 2 and sw[-1] >= n - 5 and sw[-1] - sw[-2] >= 5:
            a, b = sw[-2], sw[-1]
            hits = []
            for nm, ind in [('RSI', r), ('MACD', m), ('OBV', obv), ('پول حقیقی', cnet)]:
                if kind == 'low' and c[b] < c[a] * 0.98 and ind[b] > ind[a]: hits.append(nm)
                if kind == 'high' and c[b] > c[a] * 1.02 and ind[b] < ind[a]: hits.append(nm)
            # require at least two indicators, one of them RSI or MACD
            if len(hits) >= 2 and ({'RSI', 'MACD'} & set(hits)):
                div.append(('واگرایی مثبت (کف پایین‌تر قیمت): ' if kind == 'low' else 'واگرایی منفی (سقف بالاتر قیمت): ') + '، '.join(hits))
    row['divergence'] = '؛ '.join(div)
    return row
max_day = max(x.index[-1] for x in X.values())
if __name__ == '__main__':
    with ProcessPoolExecutor(2) as ex: rows = [r for r in ex.map(one, X.items(), chunksize=20) if r]
    T = pd.DataFrame(rows); T['sym'] = T.sym.map(live.norm)
    T.to_csv(os.path.join(live.STATE, 'tech_state.csv'), index=False)
    D = T[T.divergence != ''][['sym', 'date', 'divergence', 'n_long']]
    D.to_csv(os.path.join(live.STATE, 'divergence.csv'), index=False)
    print('symbols', len(T), 'date', T.date.max(), 'divergences', len(D))
    print(T[['n_long']].describe().T.round(1).to_string())
