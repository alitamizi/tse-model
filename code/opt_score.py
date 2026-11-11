"""Composite next-day score for option underlyings, weights fixed from train (before 2026-10-28)."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250)
P = pd.read_pickle('live_state/opt_panel_eval.pkl')
import sys
SIGNS = {'ob': 1, 'bp30': 1, 'r10': -1, 'r30': -1} if len(sys.argv) < 2 else {'ob1': 1, 'bp_d': 1, 'r10': -1, 'r30': -1}
def score(S):
    z = 0
    for f, s in SIGNS.items():
        z = z + s * (S.groupby(['day', 'T'])[f].rank(pct=True).fillna(0.5) - 0.5)
    return z / len(SIGNS)
P['score'] = score(P)
P['late'] = P['T'] >= 11 * 3600 + 30 * 60
SPLIT = '2026-10-28'
TOP = ['اهرم', 'وبملت', 'شستا', 'وتجارت', 'خودرو', 'شپنا', 'ذوب', 'فزر', 'فملی', 'خساپا', 'تاصیکو', 'اخابر', 'دزاگرس']
def rep(S, lab):
    S = S[S.yn.notna()]
    q = S.groupby(['day', 'T']).score.transform(lambda v: pd.qcut(v.rank(method='first'), 5, labels=False))
    g = S.assign(q=q).groupby('q').agg(n=('yn', 'size'), yn=('yn', 'mean'), xn=('xn', 'mean'), hit=('yn', lambda v: (v > 0).mean()),
                                       xhit=('xn', lambda v: (v > 0).mean()), yc=('yc', 'mean'))
    print(f'-- {lab}  days={S.day.nunique()}'); print(g.round(4).to_string())
    # per-day spread top-bottom (independence check)
    d = S.assign(q=q).groupby(['day']).apply(lambda x: x[x.q == 4].yn.mean() - x[x.q == 0].yn.mean())
    print('top−bottom by day: mean %.4f, positive days %d/%d' % (d.mean(), (d > 0).sum(), d.notna().sum()))
for lab, S in [('train', P[P.day < SPLIT]), ('test', P[P.day >= SPLIT]), ('test late>=11:30', P[(P.day >= SPLIT) & P.late]),
               ('test late, top-13 underlyings', P[(P.day >= SPLIT) & P.late & P.u.isin(TOP)])]:
    rep(S, lab)
# absolute threshold rules on test late
T = P[(P.day >= SPLIT) & P.late & P.yn.notna()]
for th in [0.2, 0.25, 0.3]:
    b, s = T[T.score >= th], T[T.score <= -th]
    print(f'th {th}: buy n={len(b)} ({b.groupby(["u","day"]).ngroups} sym-days) yn {b.yn.mean():+.4f} hit {(b.yn>0).mean():.2f} xn {b.xn.mean():+.4f} | '
          f'sell n={len(s)} ({s.groupby(["u","day"]).ngroups}) yn {s.yn.mean():+.4f} hit(down) {(s.yn<0).mean():.2f} xn {s.xn.mean():+.4f}')
P.to_pickle('live_state/opt_panel_scored%s.pkl' % ('' if len(sys.argv) < 2 else '_hist'))
