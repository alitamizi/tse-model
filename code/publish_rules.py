"""Build rules/rules.json, rules/context.csv and verdicts/verdicts.json in the tse-model repo."""
import os, json, datetime as dt, numpy as np, pandas as pd
import live
REPO = '/home/claude/tse-model'
X, N = live.load_daily()
C = live.context(X)
snap = pd.read_csv(os.path.join(live.REPO, 'data', 'latest.csv'))
snap['k'] = snap['نماد'].map(live.norm)
ref = snap.drop_duplicates('k').set_index('k')[['نماد', 'insCode', 'نوع']]
C = C.join(ref, how='inner')
st = C[C['نوع'] == 'سهم']
top = set(st.sort_values('mcap_d', ascending=False).head(100).index)
C['top100'] = [1 if k in top else 0 for k in C.index]
cols = ['نماد', 'insCode', 'v20', 'pr3', 'pr5', 'pr10', 'pr20', 'v3_20', 'v5_20', 'dn_days5', 'up_days5', 'nrp3', 'nrp5', 'bp3', 'top100']
os.makedirs(os.path.join(REPO, 'rules'), exist_ok=True); os.makedirs(os.path.join(REPO, 'verdicts'), exist_ok=True)
C[cols].to_csv(os.path.join(REPO, 'rules', 'context.csv'), index=False, float_format='%.6g')
name = lambda k: ref['نماد'].get(live.norm(k))
now = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
rules = [
 dict(id='R2', name='برگشت نماد بزرگ بعد از افت', side='buy', universe='top100', symbols=[],
      when=[['pr20', '<=', -0.12], ['cpct', '>=', 0.02], ['locked_up', '==', False], ['bp', '>=', 3], ['nrp', '>=', 0.10]],
      note='تاریخی حدود ۶۰٪ به +۱۵٪ در ۴۰ روز (پیش از −۸٪)', cooldown_min=1440),
 dict(id='R3', name='برگشت از کف دامنه بعد از افت', side='buy', universe='stocks', symbols=[],
      when=[['pr10', '<=', -0.08], ['dn_to_up', '==', True], ['pace', '>=', 1.5]],
      note='تاریخی حدود ۶۰٪ به +۱۵٪ در ۴۰ روز', cooldown_min=1440),
 dict(id='WQ', name='تشکیل صف خرید در نماد واچ‌لیست', side='buy', universe='all',
      symbols=[s for s in map(name, live.WATCH_QFORM) if s], when=[['at_up', '==', True], ['locked_up', '==', False]],
      note='تاریخی ۷۰ تا ۸۰٪ به +۱۰٪ در ۲۰ روز', cooldown_min=1440),
 dict(id='LF', name='قدرت خریدار در صندوق اهرمی', side='buy', universe='all',
      symbols=[s for s in map(name, live.FUNDS) if s], when=[['bp', '>=', 2]],
      note='تاریخی حدود ۶۶٪ به +۵٪ در ۱۰ روز', cooldown_min=1440),
 dict(id='ACC', name='انباشت چندروزه و چرخش به بالا', side='buy', universe='stocks', symbols=[],
      when=[['v5_20', '>=', 1.5], ['pr5', '<=', -0.03], ['dn_days5', '>=', 3], ['lpct', '>=', 0.02], ['bp', '>=', 1.5], ['locked_up', '==', False]],
      note='در حال ارزیابی', cooldown_min=1440),
 dict(id='DIST', name='توزیع چندروزه و چرخش به پایین', side='sell', universe='stocks', symbols=[],
      when=[['v5_20', '>=', 1.5], ['pr5', '>=', 0.08], ['up_days5', '>=', 3], ['lpct', '<=', -0.02], ['bp', '<=', 0.7]],
      note='در حال ارزیابی', cooldown_min=1440),
 dict(id='QB', name='صف خرید تازه در نماد بزرگ', side='buy', universe='top100', symbols=[],
      when=[['q_new_buy', '==', True], ['locked_up', '==', False], ['pace', '>=', 1.0]],
      note='فقط برای جمع‌آوری داده؛ دقت هنوز سنجیده نشده', cooldown_min=1440),
]
watch = []
if os.path.exists(live.SIG):
    S = pd.read_csv(live.SIG)
    S = S[S.kind == 'eod-model']
    for r in S.itertuples():
        nm = name(r.sym)
        if not nm or not r.close > 0: continue
        buy = r.strength == 'خرید'
        d0 = pd.Timestamp(str(r.time_utc)[:10])
        watch.append(dict(sym=nm, side='buy' if buy else 'sell', entry=float(r.close),
                          target=round(r.close * (1.05 if buy else 0.95)), stop=round(r.close * (0.95 if buy else 1.05)),
                          until=(d0 + pd.Timedelta(days=16)).strftime('%Y-%m-%d'), note=f'{r.rule} رتبه {int(r.rank)}'))
json.dump(dict(version=now[:10], updated_at_utc=now, rules=rules, watch=watch), open(os.path.join(REPO, 'rules', 'rules.json'), 'w'), ensure_ascii=False, indent=1)
vp = os.path.join(REPO, 'verdicts', 'verdicts.json')
if not os.path.exists(vp):
    json.dump(dict(updated_at_utc=now, verdicts=[]), open(vp, 'w'), ensure_ascii=False, indent=1)
print('context rows', len(C), 'top100', len(top), 'rules', len(rules), 'watch', len(watch))
print('WQ', rules[2]['symbols'], 'LF', rules[3]['symbols'])
