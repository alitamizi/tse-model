"""Hourly walk-forward replay of one session from the archive.
python3 replay_hourly.py DAY HH:MM   -> state and decisions using only data <= HH:MM
python3 replay_hourly.py DAY eval    -> score saved decisions against what happened later"""
import sys, json, os, numpy as np, pandas as pd, datetime as dt, live
pd.set_option('display.width', 250)
DAY, ARG = sys.argv[1], sys.argv[2]
OUT = f'live_state/replay_hourly_{DAY}.json'
AR = 'live_state/tse-data/archive/'
def arc(name):
    for ext in ('.csv.xz', '.csv.gz'):
        if os.path.exists(AR + name + ext):
            d = pd.read_csv(AR + name + ext, low_memory=False)
            if 'زمان دریافت' in d:   # repeated header lines from appended chunks
                d = d[d['زمان دریافت'] != 'زمان دریافت']
                for c in d.columns:
                    if c not in ('زمان دریافت', 'نماد', 'insCode', 'زمان آخرین معامله', 'بازار', 'شاخص'): d[c] = pd.to_numeric(d[c], errors='coerce')
            return d
    return None
A = arc(DAY)
HIST = False
def from_history():
    # intraday archive is published only after the close; rebuild today's path from data/history/<sym>.json (every ~3 min)
    import glob
    L = pd.read_csv('live_state/tse-data/data/latest.csv', low_memory=False)
    L['sym'] = L['نماد'].map(live.norm); L = L.drop_duplicates('sym').set_index('sym')
    rows = []
    for f in glob.glob('live_state/tse-data/data/history/*.json'):
        s = live.norm(os.path.basename(f)[:-5])
        if s not in L.index: continue
        try: h = pd.DataFrame(json.load(open(f)))
        except Exception: continue
        if not len(h) or 'زمان' not in h: continue
        h['sym'] = s; rows.append(h)
    H = pd.concat(rows, ignore_index=True)
    H = H[H['زمان'].astype(str).str.startswith(DAY)]
    H['زمان دریافت'] = H['زمان'].astype(str).str[11:19]
    for c in ['insCode', 'نماد', 'قیمت دیروز', 'اولین قیمت', 'سقف مجاز', 'کف مجاز', 'نوع', 'گروه', 'تعداد سهام']:
        H[c] = H.sym.map(L[c]) if c in L else np.nan
    H = H.sort_values(['sym', 'زمان دریافت'])
    H['کمترین'] = H.groupby('sym')['آخرین معامله'].cummin(); H['بیشترین'] = H.groupby('sym')['آخرین معامله'].cummax()
    H['تعداد معاملات'] = np.where(H['ارزش معاملات'] > 0, 1, 0)
    return H
now_sec = (dt.datetime.now(live.TEH) - dt.datetime.now(live.TEH).replace(hour=0, minute=0, second=0)).total_seconds()
if A is None or (DAY == dt.datetime.now(live.TEH).date().isoformat() and pd.to_timedelta(A['زمان دریافت'].iloc[-1]).total_seconds() < min(now_sec - 900, 12 * 3600 + 25 * 60)):
    A = from_history(); HIST = True
    print('source: data/history (path every ~3 min; interval buyer power not available)')
st = arc(DAY + '_static')
if st is None: st = pd.read_csv('live_state/tse-data/data/latest.csv')
STC = ['insCode', 'نام', 'نوع', 'گروه', 'بازار', 'تابلو', 'تعداد سهام', 'سقف مجاز', 'کف مجاز']
st = st[[c for c in STC if c in st.columns]].drop_duplicates('insCode')
A['insCode'] = A['insCode'].astype(str); st['insCode'] = st['insCode'].astype(str)
if not HIST: A = A.merge(st, on='insCode', how='left')
A['sec'] = pd.to_timedelta(A['زمان دریافت']).dt.total_seconds()
A['sym'] = A['نماد'].map(live.norm)
if not HIST:
    I = arc(DAY + '_indices'); I['sec'] = pd.to_timedelta(I['زمان دریافت']).dt.total_seconds()
    IDX = I[(I['بازار'] == 'بورس') & (I['شاخص'] == 'شاخص کل')]
X, N = pd.read_pickle('panel.pkl')                       # daily store up to the previous session only
X = {k: x[x.index < pd.Timestamp(DAY)] for k, x in X.items()}; X = {k: x for k, x in X.items() if len(x)}   # no look-ahead
assert max(x.index[-1] for x in X.values()) < pd.Timestamp(DAY)
C = live.context(X)
SM = pd.read_csv('live_state/symbol_methods.csv') if os.path.exists('live_state/symbol_methods.csv') else None

def raw_at(T):
    return A[A.sec <= T].sort_values('sec').groupby('insCode').tail(1).copy()
def snap_at(T):
    s = raw_at(T)
    if HIST:
        s['زمان آخرین معامله'] = DAY + 'T' + s['زمان دریافت'] + '+03:30'
        s['ارزش بازار'] = s['تعداد سهام'] * s['قیمت پایانی']
        s['صف'] = np.where((s['آخرین معامله'] >= s['سقف مجاز']) & (s['عرضه1 حجم'].fillna(0) == 0), 'صف خرید',
                  np.where((s['آخرین معامله'] <= s['کف مجاز']) & (s['تقاضا1 حجم'].fillna(0) == 0), 'صف فروش', ''))
        return live.prep_snap(s)
    s['زمان آخرین معامله'] = DAY + 'T' + s['زمان آخرین معامله'].astype(str) + '+03:30'
    avg = s['ارزش معاملات'] / s['حجم معاملات']
    rbv, rbn, rsv, rsn = s['حجم خرید حقیقی'], s['تعداد خرید حقیقی'], s['حجم فروش حقیقی'], s['تعداد فروش حقیقی']
    s['قدرت خریدار حقیقی'] = (rbv / rbn) / (rsv / rsn); s['ورود پول حقیقی'] = (rbv - rsv) * avg
    s['سرانه خرید حقیقی'] = rbv * avg / rbn; s['سرانه فروش حقیقی'] = rsv * avg / rsn
    s['ارزش بازار'] = s['تعداد سهام'] * s['قیمت پایانی']
    up = s['آخرین معامله'] >= s['سقف مجاز']; dn = s['آخرین معامله'] <= s['کف مجاز']
    s['صف'] = np.where(up & (s['عرضه1 حجم'].fillna(0) == 0), 'صف خرید', np.where(dn & (s['تقاضا1 حجم'].fillna(0) == 0), 'صف فروش', ''))
    return live.prep_snap(s)
def interval(T0, T1):
    """real-investor flow between T0 and T1 per symbol (difference of cumulative counters)"""
    a, b = raw_at(T0).set_index('sym'), raw_at(T1).set_index('sym')
    a = a[~a.index.duplicated()]; b = b[~b.index.duplicated()]
    if HIST:
        dv = b['ارزش معاملات'] - a.reindex(b.index)['ارزش معاملات'].fillna(0)
        dn = b['ورود پول حقیقی'] - a.reindex(b.index)['ورود پول حقیقی'].fillna(0)
        return pd.DataFrame(dict(iv=dv, ibp=np.nan, inrp=dn / dv.replace(0, np.nan), inet=dn), index=b.index)
    d = (b[['ارزش معاملات', 'حجم معاملات', 'حجم خرید حقیقی', 'تعداد خرید حقیقی', 'حجم فروش حقیقی', 'تعداد فروش حقیقی']]
         - a.reindex(b.index)[['ارزش معاملات', 'حجم معاملات', 'حجم خرید حقیقی', 'تعداد خرید حقیقی', 'حجم فروش حقیقی', 'تعداد فروش حقیقی']].fillna(0))
    avg = d['ارزش معاملات'] / d['حجم معاملات']
    out = pd.DataFrame(index=d.index)
    out['iv'] = d['ارزش معاملات']
    out['ibp'] = (d['حجم خرید حقیقی'] / d['تعداد خرید حقیقی'].replace(0, np.nan)) / (d['حجم فروش حقیقی'] / d['تعداد فروش حقیقی'].replace(0, np.nan))
    out['inrp'] = (d['حجم خرید حقیقی'] - d['حجم فروش حقیقی']) * avg / d['ارزش معاملات'].replace(0, np.nan)
    out['inet'] = (d['حجم خرید حقیقی'] - d['حجم فروش حقیقی']) * avg
    return out
def idx_at(T):
    if HIST:   # proxy: market-cap weighted change of stocks
        s = raw_at(T); s = s[~s['گروه'].astype(str).str.contains('صندوق', na=False)]
        m = s['تعداد سهام'] * s['قیمت دیروز']; ch = s['قیمت پایانی'] / s['قیمت دیروز'] - 1
        ok = m.notna() & ch.notna() & (m > 0) & (s['آخرین معامله'] > 0) & (ch.abs() < 0.11)
        return (np.nan, float((m[ok] * ch[ok]).sum() / m[ok].sum() * 100)) if ok.any() else (np.nan, np.nan)
    r = IDX[IDX.sec <= T].tail(1)
    return (float(r['مقدار'].iloc[0]), float(r['درصد تغییر'].iloc[0])) if len(r) else (np.nan, np.nan)
def tt(T): return dt.datetime.fromisoformat(DAY).replace(hour=int(T // 3600), minute=int(T % 3600 // 60), tzinfo=live.TEH)

def checkpoint(T, label, verbose=True, save=True):
    P = print if verbose else (lambda *a, **k: None)
    s = snap_at(T); s = s[s.ntr > 0]
    s = s[~s.sym.duplicated()]
    iv, ip = idx_at(T); iv1, ip1 = idx_at(T - 3600)
    stocks = s[~s['نوع'].astype(str).str.contains('صندوق|اختیار|اوراق', na=False)] if 'نوع' in s else s
    I60 = interval(T - 3600, T); I30 = interval(T - 1800, T)
    P(f'== {DAY} {ARG} (فقط داده تا این لحظه) ==')
    P(f'شاخص کل {iv:,.0f} ({ip:+.2f}%) | یک ساعت قبل {ip1:+.2f}% | مثبت {(stocks.lpct>0).mean():.0%} | صف خرید {(s.queue=="صف خرید").sum()} صف فروش {(s.queue=="صف فروش").sum()}')
    P(f'ورود پول حقیقی کل روز {s.net.sum()/1e10:,.0f} میلیارد تومان | ساعت اخیر {I60.inet.sum()/1e10:,.0f} | ارزش ساعت اخیر {I60.iv.sum()/1e10:,.0f} میلیارد تومان')
    sk = s[~s['گروه'].astype(str).str.contains('صندوق', na=False)].sym
    P(f'فقط سهام (بدون صندوق): ورود حقیقی روز {s[s.sym.isin(sk)].net.sum()/1e10:,.0f} | ساعت اخیر {I60.reindex(sk).inet.sum()/1e10:,.0f} میلیارد تومان')
    g = s.groupby('گروه').agg(net=('net', 'sum'), m=('lpct', 'mean')); g['net'] /= 1e10
    P('گروه‌ها (ورود حقیقی، میلیارد تومان):', ' | '.join(f'{k} {v.net:+.0f}' for k, v in pd.concat([g.nsmallest(3, 'net'), g.nlargest(3, 'net')]).iterrows()))
    Al, _, _ = live.scan(s, C, tt(T))
    Ap, _, _ = live.scan(snap_at(T - 600).pipe(lambda z: z[(z.ntr > 0) & ~z.sym.duplicated()]), C, tt(T - 600))
    prev = set(zip(Ap.sym, Ap.rule)) if len(Ap) and 'sym' in Ap else set()
    Al = Al.join(I30, on='sym').join(I60[['ibp', 'inrp']].add_suffix('60'), on='sym')
    Al['persist'] = [(a, b) in prev for a, b in zip(Al.sym, Al.rule)]
    Al['queue_s'] = Al.sym.map(s.set_index('sym')['queue'])
    def decide(r):
        sell = str(r.strength).startswith('فروش')
        if r.w < 0.5: return 'رد: ارزش کم'
        if not r.persist: return 'منتظر پایداری'
        if sell:
            # 2026-10-05: on 2 days, sells rejected by the 30-min flow filter did better next day (87%) than confirmed ones (71%);
            # the filter is dropped for sells (inrp is still logged so it can be re-tested), result is judged over the next days
            return 'تأیید فروش (روز بعد)'
        if sell:
            if r.inrp <= -0.1 and (r.ibp < 1 or np.isnan(r.ibp)): return 'تأیید فروش'
            return 'رد: جریان ۳۰دقیقه فروش را تأیید نمی‌کند'
        if r.queue_s == 'صف خرید': return 'قابل خرید نیست (صف)'
        if (r.ibp if not np.isnan(r.ibp) else r.bp) >= 1.2 and r.inrp >= 0.05: return 'تأیید خرید'
        return 'رد: جریان ۳۰دقیقه خرید را تأیید نمی‌کند'
    Al['decision'] = Al.apply(decide, axis=1)
    cols = ['sym', 'rule', 'strength', 'last', 'lpct', 'bp', 'nrp', 'pace', 'w', 'persist', 'ibp', 'inrp', 'iv', 'decision']
    D = Al[cols].copy(); D['lpct'] *= 100; D['iv'] /= 1e10
    P(f'هشدارها: {len(D)}'); print(D.sort_values('decision').round(2).to_string(index=False))
    # market call for the next hour: momentum of index over last hour + direction of real flow in last 30 minutes
    _, ip30 = idx_at(T - 1800); flow30 = I30.inet.sum()
    call = 'ادامه ضعف' if (ip - ip30 < 0 and flow30 < 0) else 'بهبود' if (ip - ip30 > 0 and flow30 > 0) else 'خنثی/نوسانی'
    P(f'پیش‌بینی بازار تا ساعت بعد: {call} (تغییر شاخص ۳۰ دقیقه {ip-ip30:+.2f}٪، ورود حقیقی ۳۰ دقیقه {flow30/1e10:+,.0f} میلیارد تومان)')
    if not save: return D, call, ip
    S = json.load(open(OUT)) if os.path.exists(OUT) else {}
    S[label] = dict(T=T, idx=ip, call=call, decisions=D[D.decision.str.startswith('تأیید')][['sym', 'strength', 'last', 'decision']].to_dict('records'),
                  watch=D[D.decision == 'منتظر پایداری'][['sym', 'strength', 'last']].to_dict('records'))
    json.dump(S, open(OUT, 'w'), ensure_ascii=False, indent=1)
    return D, call, ip

if ARG == 'sweep':
    # every 10 minutes between two times: decisions as they would have been made live, appended to a log for learning
    t0, t1 = [int(x.split(':')[0]) * 3600 + int(x.split(':')[1]) * 60 for x in sys.argv[3:5]]
    LOG = 'live_state/decisions_10m.csv'
    old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=['day', 'time'])
    done = set(zip(old['day'].astype(str), old['time'].astype(str)))
    out = []
    for T in range(t0, t1 + 1, 600):
        lab = f'{T//3600:02d}:{T%3600//60:02d}'
        if (DAY, lab) in done or T > A.sec.max() + 120: continue
        D, call, ip = checkpoint(T, lab, verbose=False, save=False)
        D = D.copy(); D.insert(0, 'time', lab); D.insert(0, 'day', DAY); D['idx'] = ip; D['call'] = call
        out.append(D)
    if out:
        new = pd.concat(out); new.to_csv(LOG, mode='a', header=not os.path.exists(LOG), index=False)
        f = new.drop_duplicates(['sym', 'rule'], keep='first')
        print(f'{len(out)} checkpoints logged, {len(new)} rows'); print(f[['time', 'sym', 'rule', 'last', 'lpct', 'ibp', 'inrp', 'decision']].round(2).to_string(index=False))
        print(new.drop_duplicates('time')[['time', 'idx', 'call']].to_string(index=False))
elif ARG != 'eval':
    h, m = map(int, ARG.split(':')); checkpoint(h * 3600 + m * 60, ARG)
else:
    S = json.load(open(OUT)); L = live.prep_snap(pd.read_csv('live_state/tse-data/data/latest.csv')).drop_duplicates('sym').set_index('sym')
    fin = raw_at(99999).set_index('sym'); fin = fin[~fin.index.duplicated()]
    rows = []
    for k, v in S.items():
        T = v['T']; _, ipn = idx_at(T + 3600); _, ipe = idx_at(99999)
        print(f'{k}: پیش‌بینی «{v["call"]}» | شاخص {v["idx"]:+.2f}% → یک ساعت بعد {ipn:+.2f}% → پایان {ipe:+.2f}%')
        for d in v['decisions'] + [dict(x, decision='پایش') for x in v['watch']]:
            f = A[(A.sym == d['sym']) & (A.sec > T)]['آخرین معامله']
            e = d['last']; sell = 'فروش' in d['decision'] or str(d['strength']).startswith('فروش')
            rows.append(dict(t=k, sym=d['sym'], dec=d['decision'], entry=e, close=fin['قیمت پایانی'].get(d['sym']),
                             max_after=f.max() / e - 1 if len(f) else np.nan, min_after=f.min() / e - 1 if len(f) else np.nan,
                             r_close=fin['قیمت پایانی'].get(d['sym'], np.nan) / e - 1, r_today=L['last'].get(d['sym'], np.nan) / e - 1, sell=sell))
    R = pd.DataFrame(rows)
    if len(R):
        R['ok_close'] = np.where(R.sell, R.r_close < 0, R.r_close > 0)
        R['ok_today'] = np.where(R.sell, R.r_today < 0, R.r_today > 0)
        print(R.round(3).to_string(index=False))
        print(R.groupby('dec').agg(n=('sym', 'size'), ok_close=('ok_close', 'mean'), ok_today=('ok_today', 'mean'), r_close=('r_close', 'mean'), r_today=('r_today', 'mean')).round(3))
        R.to_csv(f'live_state/replay_hourly_{DAY}_eval.csv', index=False)
