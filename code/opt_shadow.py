"""Shadow (paper) log of the next-day signal on option underlyings, rule fixed on 20 Aban (2026-11-11).
Score at 12:00 = mean of cross-sectional pct-ranks (minus 0.5) of: level-1 book imbalance, day real buyer power, −10-min return, −30-min return.
Only fields that data/history also carries live, so the same rule can run in-session later. buy if score>=0.25, sell if <=−0.25.
python3 opt_shadow.py [DAY]   (run from /home/claude/multi; logs DAY if missing, then scores past rows)"""
import sys, os, glob, numpy as np, pandas as pd
import opt_intraday as oi
OUT = 'live_state/opt_signals.csv'; TH = 0.25; T = 12 * 3600
SIGNS = {'ob1': 1, 'bp_d': 1, 'r10': -1, 'r30': -1}
def signals(day):
    x = oi.day_rows(day)
    x = x[(x['T'] == T) & (x.ob.abs() < 1) & (x.val30 > 0)].copy()
    x['score'] = sum(s * (x[f].rank(pct=True).fillna(0.5) - 0.5) for f, s in SIGNS.items()) / len(SIGNS)
    x['side'] = np.where(x.score >= TH, 'خرید', np.where(x.score <= -TH, 'فروش', ''))
    return x[['day', 'u', 'score', 'side', 'p', 'yday', 'close_end', 'ob1', 'bp_d', 'r10', 'r30']]
L = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=['day'])
days = sorted(os.path.basename(f)[:10] for f in glob.glob(oi.AR + '20??-??-??.csv.xz'))
want = [sys.argv[1]] if len(sys.argv) > 1 else days
for d in want:
    if d in set(L.day) or d not in days: continue
    L = pd.concat([L, signals(d)], ignore_index=True)
L['side'] = L.side.fillna('')
L = L.sort_values(['day', 'score'], ascending=[True, False])
# next-day close-to-close of the underlying, from the next logged day (adjustment-safe: close/yday)
nx = dict(zip(sorted(L.day.unique())[:-1], sorted(L.day.unique())[1:]))
nr = L.assign(nr=L.close_end / L.yday).set_index(['u', 'day']).nr
L['yn'] = [(r.close_end / r.p) * nr.get((r.u, nx.get(r.day)), np.nan) - 1 for r in L.itertuples()]
L['xn'] = L.yn - L.groupby('day').yn.transform('median')
L.to_csv(OUT, index=False)
S = L[L.side != '']
day = want[-1]
t = S[S.day == day]
print('OPT_TODAY', day, ' | '.join(f'{r.side} {r.u} ({r.score:+.2f})' for r in t.itertuples()) or 'بدون سیگنال')
E = S[S.yn.notna()]
for side, ok in [('خرید', E.yn > 0), ('فروش', E.yn < 0)]:
    m = E.side == side
    if m.any(): print(f'OPT_EVAL {side}: n={m.sum()} روز={E[m].day.nunique()} درست={ok[m].mean():.0%} yn={E[m].yn.mean():+.2%} xn={E[m].xn.mean():+.2%}')
