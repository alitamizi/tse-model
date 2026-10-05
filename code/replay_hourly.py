"""Hourly walk-forward replay of one session from the archive.
python3 replay_hourly.py DAY HH:MM   -> state and decisions using only data <= HH:MM
python3 replay_hourly.py DAY eval    -> score saved decisions against what happened later"""
import sys, json, os, numpy as np, pandas as pd, datetime as dt, live
pd.set_option('display.width', 250)
DAY, ARG = sys.argv[1], sys.argv[2]
OUT = f'live_state/replay_hourly_{DAY}.json'
A = pd.read_csv(f'live_state/tse-data/archive/{DAY}.csv.xz')
st = pd.read_csv(f'live_state/tse-data/archive/{DAY}_static.csv.xz').drop(columns=['نماد']).drop_duplicates('insCode')
A = A.merge(st, on='insCode', how='left')
A['sec'] = pd.to_timedelta(A['زمان دریافت']).dt.total_seconds()
A['sym'] = A['نماد'].map(live.norm)
I = pd.read_csv(f'live_state/tse-data/archive/{DAY}_indices.csv.xz'); I['sec'] = pd.to_timedelta(I['زمان دریافت']).dt.total_seconds()
IDX = I[(I['بازار'] == 'بورس') & (I['شاخص'] == 'شاخص کل')]
X, N = pd.read_pickle('live_state/panel_prev.pkl')        # daily store up to the previous session only
assert max(x.index[-1] for x in X.values()) < pd.Timestamp(DAY)
C = live.context(X)
SM = pd.read_csv('live_state/symbol_methods.csv') if os.path.exists('live_state/symbol_methods.csv') else None

def raw_at(T):
    return A[A.sec <= T].sort_values('sec').groupby('insCode').tail(1).copy()
def snap_at(T):
    s = raw_at(T)
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
    r = IDX[IDX.sec <= T].tail(1)
    return (float(r['مقدار'].iloc[0]), float(r['درصد تغییر'].iloc[0])) if len(r) else (np.nan, np.nan)
def tt(T): return dt.datetime.fromisoformat(DAY).replace(hour=int(T // 3600), minute=int(T % 3600 // 60), tzinfo=live.TEH)

if ARG != 'eval':
    h, m = map(int, ARG.split(':')); T = h * 3600 + m * 60
    s = snap_at(T); s = s[s.ntr > 0]
    s = s[~s.sym.duplicated()]
    iv, ip = idx_at(T); iv1, ip1 = idx_at(T - 3600)
    stocks = s[~s['نوع'].astype(str).str.contains('صندوق|اختیار|اوراق', na=False)] if 'نوع' in s else s
    I60 = interval(T - 3600, T); I30 = interval(T - 1800, T)
    print(f'== {DAY} {ARG} (فقط داده تا این لحظه) ==')
    print(f'شاخص کل {iv:,.0f} ({ip:+.2f}%) | یک ساعت قبل {ip1:+.2f}% | مثبت {(stocks.lpct>0).mean():.0%} | صف خرید {(s.queue=="صف خرید").sum()} صف فروش {(s.queue=="صف فروش").sum()}')
    print(f'ورود پول حقیقی کل روز {s.net.sum()/1e10:,.0f} میلیارد تومان | ساعت اخیر {I60.inet.sum()/1e10:,.0f} | ارزش ساعت اخیر {I60.iv.sum()/1e10:,.0f} میلیارد تومان')
    sk = s[~s['گروه'].astype(str).str.contains('صندوق', na=False)].sym
    print(f'فقط سهام (بدون صندوق): ورود حقیقی روز {s[s.sym.isin(sk)].net.sum()/1e10:,.0f} | ساعت اخیر {I60.reindex(sk).inet.sum()/1e10:,.0f} میلیارد تومان')
    g = s.groupby('گروه').agg(net=('net', 'sum'), m=('lpct', 'mean')); g['net'] /= 1e10
    print('گروه‌ها (ورود حقیقی، میلیارد تومان):', ' | '.join(f'{k} {v.net:+.0f}' for k, v in pd.concat([g.nsmallest(3, 'net'), g.nlargest(3, 'net')]).iterrows()))
    Al, _, _ = live.scan(s, C, tt(T))
    Ap, _, _ = live.scan(snap_at(T - 600).pipe(lambda z: z[(z.ntr > 0) & ~z.sym.duplicated()]), C, tt(T - 600))
    prev = set(zip(Ap.sym, Ap.rule))
    Al = Al.join(I30, on='sym').join(I60[['ibp', 'inrp']].add_suffix('60'), on='sym')
    Al['persist'] = [(a, b) in prev for a, b in zip(Al.sym, Al.rule)]
    Al['queue_s'] = Al.sym.map(s.set_index('sym')['queue'])
    def decide(r):
        sell = str(r.strength).startswith('فروش')
        if r.w < 0.5: return 'رد: ارزش کم'
        if not r.persist: return 'منتظر پایداری'
        if sell:
            if r.inrp <= -0.1 and (r.ibp < 1 or np.isnan(r.ibp)): return 'تأیید فروش'
            return 'رد: جریان ۳۰دقیقه فروش را تأیید نمی‌کند'
        if r.queue_s == 'صف خرید': return 'قابل خرید نیست (صف)'
        if r.ibp >= 1.2 and r.inrp >= 0.05: return 'تأیید خرید'
        return 'رد: جریان ۳۰دقیقه خرید را تأیید نمی‌کند'
    Al['decision'] = Al.apply(decide, axis=1)
    cols = ['sym', 'rule', 'strength', 'last', 'lpct', 'bp', 'nrp', 'pace', 'w', 'persist', 'ibp', 'inrp', 'iv', 'decision']
    D = Al[cols].copy(); D['lpct'] *= 100; D['iv'] /= 1e10
    print(f'هشدارها: {len(D)}'); print(D.sort_values('decision').round(2).to_string(index=False))
    # market call for the next hour: momentum of index over last hour + direction of real flow in last 30 minutes
    _, ip30 = idx_at(T - 1800); flow30 = I30.inet.sum()
    call = 'ادامه ضعف' if (ip - ip30 < 0 and flow30 < 0) else 'بهبود' if (ip - ip30 > 0 and flow30 > 0) else 'خنثی/نوسانی'
    print(f'پیش‌بینی بازار تا ساعت بعد: {call} (تغییر شاخص ۳۰ دقیقه {ip-ip30:+.2f}٪، ورود حقیقی ۳۰ دقیقه {flow30/1e10:+,.0f} میلیارد تومان)')
    S = json.load(open(OUT)) if os.path.exists(OUT) else {}
    S[ARG] = dict(T=T, idx=ip, call=call, decisions=D[D.decision.str.startswith('تأیید')][['sym', 'strength', 'last', 'decision']].to_dict('records'),
                  watch=D[D.decision == 'منتظر پایداری'][['sym', 'strength', 'last']].to_dict('records'))
    json.dump(S, open(OUT, 'w'), ensure_ascii=False, indent=1)
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
