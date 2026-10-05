# -*- coding: utf-8 -*-
"""Live monitoring for TSE.  Usage:
   python3 live.py intraday   -> scan current snapshot, log + print new alerts
   python3 live.py close      -> append today's final row to the daily store, score next-day 5% model, evaluate old signals
   python3 live.py status     -> data freshness only
"""
import os, sys, json, subprocess, datetime as dt, numpy as np, pandas as pd

BASE = '/home/claude/multi'
STATE = os.path.join(BASE, 'live_state'); os.makedirs(STATE, exist_ok=True)
REPO = os.path.join(STATE, 'tse-data')
SIG = os.path.join(STATE, 'signals.csv')
SEEN = os.path.join(STATE, 'seen.json')
TEH = dt.timezone(dt.timedelta(hours=3, minutes=30))

def norm(s):
    return str(s).replace('‌', '').replace(' ', '').replace('ي', 'ی').replace('ك', 'ک')

def g2j(d):
    gy, gm, gd = d.year, d.month, d.day
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    gy2 = gy + 1 if gm > 2 else gy
    days = 355666 + (365 * gy) + ((gy2 + 3) // 4) - ((gy2 + 99) // 100) + ((gy2 + 399) // 400) + gd + g_d_m[gm - 1]
    jy = -1595 + (33 * (days // 12053)); days %= 12053
    jy += 4 * (days // 1461); days %= 1461
    if days > 365:
        jy += (days - 1) // 365; days = (days - 1) % 365
    jm = 1 + days // 31 if days < 186 else 7 + (days - 186) // 30
    jd = 1 + (days % 31 if days < 186 else (days - 186) % 30)
    return f'{jy:04d}-{jm:02d}-{jd:02d}'

def pull():
    if not os.path.isdir(os.path.join(REPO, '.git')):
        subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/alitamizi/tse-data.git', REPO], check=True, timeout=180)
    else:
        subprocess.run(['git', '-C', REPO, 'fetch', '-q', '--depth', '1', 'origin', 'main'], check=True, timeout=180)
        subprocess.run(['git', '-C', REPO, 'reset', '-q', '--hard', 'origin/main'], check=True, timeout=60)
    meta = json.load(open(os.path.join(REPO, 'data', 'meta.json')))
    now = dt.datetime.now(dt.timezone.utc)
    f = meta.get('fetched_at_utc')
    lag = (now - dt.datetime.fromisoformat(f.replace('Z', '+00:00'))).total_seconds() if f else None
    snap = pd.read_csv(os.path.join(REPO, 'data', 'latest.csv')) if os.path.exists(os.path.join(REPO, 'data', 'latest.csv')) else None
    idx = None
    p = os.path.join(REPO, 'data', 'indices.json')
    if os.path.exists(p):
        try: idx = json.load(open(p))
        except Exception: pass
    return meta, lag, snap, idx, now

def prep_snap(snap):
    s = snap.copy()
    s['sym'] = s['نماد'].map(norm)
    num = lambda c: pd.to_numeric(s.get(c), errors='coerce')
    s['last'] = num('آخرین معامله'); s['close'] = num('قیمت پایانی'); s['yc'] = num('قیمت دیروز'); s['open'] = num('اولین قیمت')
    s['low'] = num('کمترین'); s['high'] = num('بیشترین'); s['ntr'] = num('تعداد معاملات'); s['vol'] = num('حجم معاملات'); s['value'] = num('ارزش معاملات')
    s['up'] = num('سقف مجاز'); s['dn'] = num('کف مجاز'); s['bp'] = num('قدرت خریدار حقیقی'); s['net'] = num('ورود پول حقیقی')
    s['pcb'] = num('سرانه خرید حقیقی'); s['pcs'] = num('سرانه فروش حقیقی'); s['mcap'] = num('ارزش بازار')
    s['lpct'] = s['last'] / s['yc'] - 1; s['cpct'] = s['close'] / s['yc'] - 1; s['lo_pct'] = s['low'] / s['yc'] - 1
    s['lim_up'] = s['up'] / s['yc'] - 1; s['lim_dn'] = 1 - s['dn'] / s['yc']
    s['at_up'] = s['last'] >= s['up'] - 1e-9; s['at_dn'] = s['last'] <= s['dn'] + 1e-9
    s['touched_dn'] = s['low'] <= s['dn'] + 1e-9
    s['locked_up'] = s['at_up'] & (s['low'] >= s['up'] - 1e-9)
    s['queue'] = s.get('صف', pd.Series('', index=s.index)).fillna('')
    s['nrp'] = s['net'] / s['value']
    return s

def load_daily():
    X, N = pd.read_pickle(os.path.join(BASE, 'panel.pkl'))
    return X, N

def context(X):
    """multi-day context per symbol from the daily store (up to the last stored day)"""
    rows = {}
    for s, x in X.items():
        if len(x) < 30: continue
        c = x.c; v = x.value.astype(float); nrp = x.nrp
        rows[norm(s)] = dict(
            last_day=x.index[-1], c_last=c.iloc[-1],
            pr3=c.iloc[-1] / c.iloc[-4] - 1, pr5=c.iloc[-1] / c.iloc[-6] - 1, pr10=c.iloc[-1] / c.iloc[-11] - 1, pr20=c.iloc[-1] / c.iloc[-21] - 1,
            v20=v.iloc[-21:-1].mean(), v3_20=v.iloc[-3:].mean() / v.iloc[-23:-3].mean() if v.iloc[-23:-3].mean() > 0 else np.nan,
            v5_20=v.iloc[-5:].mean() / v.iloc[-25:-5].mean() if v.iloc[-25:-5].mean() > 0 else np.nan,
            nrp3=nrp.iloc[-3:].mean(), nrp5=nrp.iloc[-5:].mean(), bp3=x.bp.iloc[-3:].mean(),
            dn_days5=int((c.diff().iloc[-5:] < 0).sum()), up_days5=int((c.diff().iloc[-5:] > 0).sum()),
            mcap_d=x.mcap.iloc[-1])
    C = pd.DataFrame(rows).T
    for c in C.columns:
        if c != 'last_day': C[c] = pd.to_numeric(C[c], errors='coerce')
    return C

WATCH_QFORM = ['شرنگی', 'وثخوز', 'کاسپین', 'فرآور', 'قیستو']
WATCH_BS = ['کرومیت', 'خفنر', 'حخزر', 'واحیا', 'خوساز', 'کارام', 'غگلپا', 'چنوپا']
FUNDS = ['اهرم', 'موج', 'نارنجاهرم', 'دوایکس']

def elapsed_frac(now_teh):
    m = (now_teh.hour * 60 + now_teh.minute) - 9 * 60
    return min(max(m / 210, 0.05), 1.0)

def scan(s, C, now_teh, final=False):
    s = s.join(C, on='sym', how='left')
    frac = 1.0 if final else elapsed_frac(now_teh)
    s['pace'] = s['value'] / (s['v20'] * frac)
    stocks = s[(s['نوع'].isin(['سهم', 'صندوق'])) & (s['ntr'] > 0)]
    top100 = set(stocks[stocks['نوع'] == 'سهم'].sort_values('mcap', ascending=False).head(100)['sym'])
    A = []
    def add(r, rule, note, strength):
        # weight: low trade value (vs. the symbol's usual pace, or tiny in absolute terms) means a less trustworthy signal
        w = float(np.clip(r.pace, 0, 1)) if r.pace == r.pace else 0.5
        if r.value < 2e10: w *= 0.5
        A.append(dict(sym=r.sym, rule=rule, note=note, strength=strength, last=r['last'], close=r['close'], cpct=r.cpct, lpct=r.lpct,
                      bp=r.bp, nrp=r.nrp, pace=r.pace, pr10=r.pr10, pr20=r.pr20, w=round(w, 2), value=r.value))
    for _, r in stocks.iterrows():
        big = r.sym in top100
        # R2 big caps: after 20-day drop >=12%, strong day with heavy real buying (validated ~60%)
        if big and r.pr20 <= -0.12 and r.cpct >= 0.02 and not r.locked_up and r.bp >= 3 and r.nrp >= 0.10:
            add(r, 'R2-بزرگ', 'افت ۲۰روزه ≥۱۲٪ + روز +۲٪ + قدرت خریدار ≥۳ + ورود پول ≥۱۰٪ (تاریخی ~۶۰٪ به +۱۵٪ در ۴۰ روز)', 'خرید')
        # R3: after 10-day drop >=8%, from lower limit back above mid-range with heavy value
        if r.pr10 <= -0.08 and r.touched_dn and r.lpct >= 0.5 * r.lim_up and r.pace >= 1.5:
            add(r, 'R3-برگشت', 'افت ۱۰روزه ≥۸٪ + برگشت از کف دامنه + ارزش بالا (تاریخی ~۶۰٪)', 'خرید')
        # watchlist: buy queue forming intraday (not locked from the open)
        if r.sym in WATCH_QFORM and r.at_up and not r.locked_up:
            add(r, 'صف-واچ‌لیست', 'تشکیل صف خرید در طول روز در نماد واچ‌لیست (تاریخی ۷۰–۸۰٪ به +۱۰٪ در ۲۰ روز)', 'خرید')
        # leveraged funds: real buyer power >= 2
        if r.sym in FUNDS and r.bp >= 2:
            add(r, 'صندوق-اهرمی', 'قدرت خریدار ≥۲ در صندوق اهرمی (تاریخی ~۶۶٪ به +۵٪ در ۱۰ روز)', 'خرید')
        # accumulation then flip (user hypothesis; under evaluation)
        if r.v5_20 >= 1.5 and r.pr5 <= -0.03 and r.dn_days5 >= 3 and (r.at_up or r.lpct >= 0.7 * r.lim_up) and r.bp >= 1.5 and not r.locked_up:
            add(r, 'انباشت→صف-خرید', 'چند روز ارزش بالا با قیمت منفی، امروز چرخش به سقف (در حال ارزیابی)', 'خرید؟')
        if r.v5_20 >= 1.5 and r.pr5 >= 0.08 and r.up_days5 >= 3 and (r.at_dn or r.lpct <= -0.7 * r.lim_dn) and r.bp <= 0.7:
            add(r, 'توزیع→صف-فروش', 'چند روز ارزش بالا با قیمت مثبت، امروز چرخش به کف (در حال ارزیابی)', 'فروش؟')
    return pd.DataFrame(A), stocks, top100

def market_line(s, idx):
    st = s[(s['نوع'] == 'سهم') & (s['ntr'] > 0)]
    up = (st.cpct > 0).mean(); bq = (st.queue == 'صف خرید').sum(); sq = (st.queue == 'صف فروش').sum()
    inflow = st.net.sum() / 1e10
    t = f'سهم‌های مثبت {up:.0%} | صف خرید {bq} | صف فروش {sq} | ورود پول حقیقی {inflow:,.0f} میلیارد تومان'
    try:
        b = idx['بازارها'][0]; t = f"شاخص کل {b['شاخص کل']:,.0f} ({b['درصد تغییر شاخص کل']:+.2f}٪) | " + t
    except Exception: pass
    return t

SIGCOLS = ['time_utc', 'kind', 'sym', 'rule', 'strength', 'note', 'last', 'close', 'cpct', 'lpct', 'bp', 'nrp', 'pace', 'pr10', 'pr20', 'p', 'rank', 'lag_s', 'w', 'value']

def log_signals(A, now, lag, kind):
    if A is None or len(A) == 0: return A
    A = A.copy(); A['time_utc'] = now.strftime('%Y-%m-%dT%H:%M:%SZ'); A['lag_s'] = lag; A['kind'] = kind
    seen = json.load(open(SEEN)) if os.path.exists(SEEN) else {}
    day = now.astimezone(TEH).strftime('%Y-%m-%d')
    key = A.sym + '|' + A.rule + '|' + day
    new = A[~key.isin(seen.keys())].copy()
    for k in key[~key.isin(seen.keys())]: seen[k] = 1
    json.dump(seen, open(SEEN, 'w'))
    if len(new):
        new.reindex(columns=SIGCOLS).to_csv(SIG, mode='a', header=not os.path.exists(SIG), index=False)
    return new

POS = os.path.join(STATE, 'positions.csv')
try:
    import methods as _M; M_NAMES = _M.NAMES
except Exception:
    M_NAMES = {}

FLOWCOLS = ['ارزش معاملات', 'حجم خرید حقیقی', 'تعداد خرید حقیقی', 'حجم فروش حقیقی', 'تعداد فروش حقیقی', 'حجم خرید حقوقی', 'حجم فروش حقوقی']

def baseline(sym_raw, t_utc):
    """cumulative flows of a symbol as of time t (UTC iso) from the bot's minute archive, else from our saved snapshots"""
    t = pd.Timestamp(t_utc).tz_convert(TEH)
    day = t.strftime('%Y-%m-%d'); hhmm = t.strftime('%H:%M:%S')
    p = os.path.join(REPO, 'archive', f'{day}.csv.gz')
    if os.path.exists(p):
        a = pd.read_csv(p, usecols=['زمان دریافت', 'نماد'] + FLOWCOLS)
        a = a[(a['نماد'] == sym_raw) & (a['زمان دریافت'] <= hhmm)]
        if len(a): return a.sort_values('زمان دریافت').iloc[-1][FLOWCOLS].astype(float)
    return None

def interval_flow(cur_row, base):
    d = cur_row[FLOWCOLS].astype(float) - base
    bp = (d['حجم خرید حقیقی'] / d['تعداد خرید حقیقی']) / (d['حجم فروش حقیقی'] / d['تعداد فروش حقیقی']) \
        if d['تعداد خرید حقیقی'] > 0 and d['تعداد فروش حقیقی'] > 0 and d['حجم فروش حقیقی'] > 0 else np.nan
    share = d['ارزش معاملات'] / cur_row['ارزش معاملات'] if cur_row['ارزش معاملات'] > 0 else np.nan
    return bp, d['ارزش معاملات'], share, d['حجم خرید حقوقی'] - d['حجم فروش حقوقی']

def monitor_positions(s, now, meta):
    """open positions = confirmed verdicts not yet closed; warn on stop, target or a change of direction"""
    vp = os.path.join(STATE, 'verdicts.csv')
    if not os.path.exists(vp): return
    V = pd.read_csv(vp)
    V = V[V.decision.astype(str).str.startswith('confirm')]
    closed = set(pd.read_csv(POS).verdict_id) if os.path.exists(POS) else set()
    cur = s.drop_duplicates('sym').set_index('sym')
    out = []
    for v in V.itertuples():
        if v.verdict_id in closed: continue
        k = norm(v.sym)
        if k not in cur.index: continue
        r = cur.loc[k]; buy = v.decision == 'confirm_buy'; e = float(v.entry)
        ret = (r['last'] / e - 1) * (1 if buy else -1)
        why = None
        # flows SINCE the verdict (not the cumulative day): who traded the value that came after our entry
        base = baseline(r['نماد'], v.decided_at_utc)
        bpi, vali, sharei, legali = interval_flow(r, base) if base is not None else (np.nan, np.nan, np.nan, np.nan)
        print(f'  since verdict: value={vali/1e7:,.0f}m toman ({sharei:.0%} of day) real-bp={bpi:.2f} legal-net-vol={legali:,.0f}')
        if buy:
            if r['last'] <= v.stop: why = 'حد ضرر خورد'
            elif r['last'] >= v.target: why = 'به هدف رسید'
            elif r.queue == 'صف فروش' or (r.bp < 0.7 and r.nrp < -0.10 and r.lpct < 0): why = 'تغییر جهت: فروشنده‌ها غالب شدند'
            elif bpi < 0.8 and sharei >= 0.25 and r['last'] < e: why = f'تغییر جهت بعد از ورود: {sharei:.0%} ارزش روز بعد از ورود با قدرت خریدار {bpi:.2f} معامله شد و قیمت زیر ورود است'
        else:
            if r['last'] >= v.stop: why = 'حد ضرر خورد'
            elif r['last'] <= v.target: why = 'به هدف رسید'
            elif r.queue == 'صف خرید' or (r.bp > 1.5 and r.nrp > 0.10 and r.lpct > 0): why = 'تغییر جهت: خریدارها غالب شدند'
            elif bpi > 1.25 and sharei >= 0.25 and r['last'] > e: why = f'تغییر جهت بعد از ورود: خریدارها بعد از ورود غالب شدند (قدرت خریدار {bpi:.2f})'
        print(f'POSITION {v.sym} {v.decision} entry={e:.0f} last={r["last"]:.0f} ret={ret:+.1%} bp={r.bp:.2f} queue={r.queue or "-"}' + (f' EXIT: {why}' if why else ''))
        if why:
            out.append(dict(verdict_id=v.verdict_id, sym=v.sym, side=v.decision, entry=e, exit=r['last'], ret=ret, reason=why,
                            closed_at_utc=now.strftime('%Y-%m-%dT%H:%M:%SZ'), data_fetched_at_utc=meta.get('fetched_at_utc')))
    if out:
        pd.DataFrame(out).to_csv(POS, mode='a', header=not os.path.exists(POS), index=False)

def cmd_status():
    meta, lag, snap, idx, now = pull()
    print(json.dumps(dict(fetched_at_utc=meta.get('fetched_at_utc'), symbols=meta.get('symbols'), lag_s=lag, now=now.isoformat()), ensure_ascii=False))

def cmd_intraday():
    meta, lag, snap, idx, now = pull()
    nt = now.astimezone(TEH)
    if not meta.get('symbols') or snap is None:
        print(f'NO_DATA fetched_at_utc={meta.get("fetched_at_utc")} symbols={meta.get("symbols")}'); return
    s = prep_snap(snap)
    trade_today = s['زمان آخرین معامله'].astype(str).str[:10].eq(nt.strftime('%Y-%m-%d')).mean()
    X, N = load_daily(); C = context(X)
    A, stocks, top100 = scan(s, C, nt)
    # persistence: an alert counts as persistent only if the same symbol+rule was also present in the previous scan (<= 20 min ago)
    pp = os.path.join(STATE, 'prev_alerts.json')
    prev = json.load(open(pp)) if os.path.exists(pp) else {}
    cur = {}
    if A is not None and len(A):
        keys = A.sym + '|' + A.rule
        A['persist'] = [bool(k in prev and (now.timestamp() - prev[k]) <= 1200) for k in keys]
        cur = {k: now.timestamp() for k in keys}
    json.dump(cur, open(pp, 'w'))
    new = log_signals(A, now, lag, 'intraday')
    monitor_positions(s, now, meta)
    print(f'DATA fetched_at_utc={meta["fetched_at_utc"]} lag={lag:.0f}s symbols={meta["symbols"]} traded_today_share={trade_today:.2f}')
    print('MARKET', market_line(s, idx))
    print('ALERTS_TOTAL', 0 if A is None else len(A), 'NEW', 0 if new is None else len(new))
    if A is not None and len(A):
        print(A[['sym', 'rule', 'strength', 'last', 'lpct', 'bp', 'nrp', 'pace', 'w', 'persist', 'pr20']].round(3).to_string())
    print('SYNC', sync(f'Intraday scan {nt:%Y-%m-%d %H:%M}'))

def append_day(s, X, N):
    """append the session in snapshot s (final) to the daily store X; returns (date, n_added, n_adjusted)"""
    tt = pd.to_datetime(s['زمان آخرین معامله'].astype(str).str[:10], errors='coerce')
    day = tt.max()
    if pd.isna(day): return None, 0, 0
    jd = g2j(day.date()); jy = int(jd[:4])
    by = {norm(k): k for k in X}
    add = adj = 0
    for _, r in s[(tt == day) & (s['ntr'] > 0)].iterrows():
        k = by.get(r.sym)
        if k is None: continue
        x = X[k]
        if x.index[-1] >= day: continue
        if r.yc > 0 and x.c.iloc[-1] > 0:
            ratio = r.yc / x.c.iloc[-1]
            if abs(ratio - 1) > 0.003 and (day - x.index[-1]).days <= 7:
                for c in ['o', 'h', 'l', 'c', 'last']: x[c] = x[c] * ratio
                adj += 1
        row = dict(o=r.open, h=r.high, l=r.low, c=r.close, last=r['last'], value=r.value, vol=r.vol, ntr=r.ntr, mcap=r.mcap,
                   bp=r.bp if r.bp > 0 else np.nan, net=r.net, nrp=r.nrp, pcb=r.pcb, pcs=r.pcs, jdate=jd, jy=jy, rawpct=r.cpct,
                   gap=bool(abs(np.log(r.close / x.c.iloc[-1])) > np.log(1.6)) if x.c.iloc[-1] > 0 else False)
        for c in x.columns:
            row.setdefault(c, np.nan)
        X[k] = pd.concat([x, pd.DataFrame([row], index=[day])[x.columns]])
        add += 1
    return day, add, adj

TARGETS = {'R2-بزرگ': (0.15, 0.08, 40), 'R3-برگشت': (0.15, 0.08, 40), 'صف-واچ‌لیست': (0.10, 0.10, 20), 'صندوق-اهرمی': (0.05, 0.05, 10),
           'انباشت→صف-خرید': (0.05, 0.05, 10), 'توزیع→صف-فروش': (0.05, 0.05, 10), '5%-خرید': (0.05, 0.05, 10), '5%-فروش': (0.05, 0.05, 10)}

def evaluate(X):
    if not os.path.exists(SIG): return pd.DataFrame()
    S = pd.read_csv(SIG)
    by = {norm(k): k for k in X}
    out = []
    for _, g in S.iterrows():
        k = by.get(g.sym)
        tgt, stop, H = TARGETS.get(g.rule, (0.05, 0.05, 10))
        if k is None or not g['close'] > 0: continue
        x = X[k]; t0 = pd.Timestamp(str(g.time_utc)[:10])
        fut = x.c[x.index > t0].iloc[:H]
        sell = str(g.strength).startswith('فروش')
        e = g['close']
        res = 'باز'
        for p in fut:
            up, dn = p / e - 1, 1 - p / e
            if (not sell and up >= tgt) or (sell and dn >= tgt): res = 'درست'; break
            if (not sell and dn >= stop) or (sell and up >= stop): res = 'غلط'; break
        else:
            if len(fut) >= H: res = 'بی‌نتیجه'
        out.append(dict(sym=g.sym, rule=g.rule, time=g.time_utc, entry=e, days=len(fut), result=res))
    return pd.DataFrame(out)

def cmd_close():
    import pickle, lightgbm as lgb
    meta, lag, snap, idx, now = pull()
    if not meta.get('symbols') or snap is None:
        print(f'NO_DATA fetched_at_utc={meta.get("fetched_at_utc")}'); return
    s = prep_snap(snap)
    X, N = load_daily()
    day, add, adj = append_day(s, X, N)
    print(f'DATA fetched_at_utc={meta["fetched_at_utc"]} lag={lag:.0f}s session={day.date() if day is not None else None} appended={add} adjusted={adj}')
    if add:
        os.replace(os.path.join(BASE, 'panel.pkl'), os.path.join(STATE, 'panel_prev.pkl'))
        pd.to_pickle((X, N), os.path.join(BASE, 'panel.pkl'))
    # end-of-day scan with full-day pace
    C = context(X)
    A, stocks, top100 = scan(s, C, now.astimezone(TEH), final=True)
    new = log_signals(A, now, lag, 'eod')
    # 5% model for the next 10 days
    try:
        subprocess.run(['python3', 'flow_study.py'], cwd=BASE, check=True, timeout=1200, capture_output=True)
        g = {}
        exec(open(os.path.join(BASE, 'five_feats.py')).read(), g)
        F, FEAT = g['F'], g['FEAT']
        M = pickle.load(open(os.path.join(BASE, 'five_models.pkl'), 'rb'))
        last = F[F.date == F.date.max()].copy()
        for side in ['yb', 'ys']:
            last['p_' + side] = lgb.Booster(model_str=M['models'][side]).predict(last[FEAT])
        cur = s.set_index('sym')
        rows = []
        # liquidity filter (added 2026-10-03 after review): skip symbols with tiny trade value or locked in a queue on the side we want to enter
        last['k'] = last.sym.map(norm)
        val = last.k.map(cur['value']); q = last.k.map(cur['queue']).fillna('')
        liquid = val >= 2e10
        for side, rule, st in [('yb', '5%-خرید', 'خرید'), ('ys', '5%-فروش', 'فروش')]:
            ok = liquid & (q != ('صف خرید' if side == 'yb' else 'صف فروش'))
            top = last[ok].sort_values('p_' + side, ascending=False).head(10)
            for rk, r in enumerate(top.itertuples(), 1):
                sy = norm(r.sym)
                rows.append(dict(sym=sy, rule=rule, note=f'رتبه {rk} مدل ۵٪ (۱۰ روز)', strength=st,
                                 last=cur['last'].get(sy, np.nan), close=cur['close'].get(sy, np.nan), cpct=cur['cpct'].get(sy, np.nan),
                                 lpct=np.nan, bp=cur['bp'].get(sy, np.nan), nrp=cur['nrp'].get(sy, np.nan), pace=np.nan, pr10=np.nan, pr20=np.nan,
                                 p=getattr(r, 'p_' + side), rank=rk))
        T = pd.DataFrame(rows)
        log_signals(T[T['rank'] <= 3], now, lag, 'eod-model')
        print('MODEL_DATE', last.date.max().date())
        print(T[['sym', 'rule', 'rank', 'p', 'close', 'bp']].round(3).to_string())
    except Exception as e:
        print('MODEL_ERROR', repr(e)[:300])
    print('MARKET', market_line(s, idx))
    if new is not None and len(new):
        print('EOD_ALERTS'); print(new[['sym', 'rule', 'strength', 'close', 'cpct', 'bp', 'nrp', 'pace']].round(3).to_string())
    # all mechanical technical methods + divergences on the updated daily data; log today's method signals for live scoring
    try:
        subprocess.run(['python3', 'tech_daily.py'], cwd=BASE, check=True, timeout=1800, capture_output=True)
        TS = pd.read_csv(os.path.join(STATE, 'tech_state.csv'))
        cur = s.drop_duplicates('sym').set_index('sym')
        tech = []
        for r in TS.itertuples():
            for col, st in [('buy_today', 'خرید'), ('sell_today', 'فروش')]:
                v = getattr(r, col)
                if isinstance(v, str) and v:
                    for m in v.split(','):
                        tech.append(dict(sym=r.sym, rule='تکنیکال-' + m, note=M_NAMES.get(m, m), strength=st,
                                         last=cur['last'].get(r.sym, np.nan), close=cur['close'].get(r.sym, np.nan)))
        # flow signals of today (real and legal side), logged so each symbol's flow behaviour is scored live too
        Cx = context(X)
        for sy, r in cur.iterrows():
            if not (r.ntr > 0) or r['نوع'] not in ('سهم', 'صندوق'): continue
            pace = r.value / Cx['v20'].get(sy, np.nan) if sy in Cx.index else np.nan
            pr5 = Cx['pr5'].get(sy, np.nan) if sy in Cx.index else np.nan
            legal = -r.nrp if r.nrp == r.nrp else np.nan
            qb = bool(r.at_up and not r.locked_up and r['last'] / r['close'] - 1 >= 0.01)
            qs = bool(r.at_dn and r['last'] / r['close'] - 1 <= -0.01)
            flags = [('جریان-قدرت‌خریدار-حقیقی-بالا', 'خرید', r.bp >= 2), ('جریان-ورود-پول-حقیقی', 'خرید', r.nrp >= 0.10),
                     ('جریان-خرید-حقوقی', 'خرید', legal >= 0.10), ('جریان-خرید-حقوقی-در-افت', 'خرید', legal >= 0.10 and pr5 < 0),
                     ('جریان-ارزش-بالا-روز-مثبت', 'خرید', pace >= 2 and r.cpct > 0), ('جریان-صف-خرید-درون‌روز', 'خرید', qb),
                     ('جریان-قدرت‌خریدار-حقیقی-پایین', 'فروش', r.bp <= 0.5), ('جریان-خروج-پول-حقیقی', 'فروش', r.nrp <= -0.10),
                     ('جریان-فروش-حقوقی', 'فروش', legal <= -0.10), ('جریان-فروش-حقوقی-در-رشد', 'فروش', legal <= -0.10 and pr5 > 0),
                     ('جریان-ارزش-بالا-روز-منفی', 'فروش', pace >= 2 and r.cpct < 0), ('جریان-صف-فروش-درون‌روز', 'فروش', qs)]
            for nm, st, f in flags:
                if f is True or (isinstance(f, (bool, np.bool_)) and bool(f)):
                    tech.append(dict(sym=sy, rule=nm, note='سیگنال جریان پول (حقیقی/حقوقی)', strength=st, last=r['last'], close=r['close']))
        log_signals(pd.DataFrame(tech), now, lag, 'eod-tech')
        DV = pd.read_csv(os.path.join(STATE, 'divergence.csv'))
        print('TECH signals', len(tech), '| DIVERGENCES', len(DV), '(منفی', int(DV.divergence.str.contains('منفی').sum()), '/ مثبت', int(DV.divergence.str.contains('مثبت').sum()), ')')
        print(DV.head(15).to_string())
    except Exception as e:
        print('TECH_ERROR', repr(e)[:300])
    E = evaluate(X)
    if len(E):
        E.to_csv(os.path.join(STATE, 'evaluation.csv'), index=False)
        print('EVAL'); print(E.groupby(['rule', 'result']).size().unstack(fill_value=0).to_string())
    try:
        r = subprocess.run(['python3', 'symbol_methods.py'], cwd=BASE, capture_output=True, text=True, timeout=600); print('METHODS', r.stdout.strip()[-200:])
    except Exception as e:
        print('METHODS_ERROR', repr(e)[:200])
    rows = None
    if day is not None:
        rows = pd.concat([x.loc[[day]].assign(sym=k) for k, x in X.items() if day in x.index]).rename_axis('date').reset_index() if add else None
    try:
        r = subprocess.run(['python3', 'publish_rules.py'], cwd=BASE, capture_output=True, text=True, timeout=600)
        print('RULES', r.stdout.strip()[-300:] or r.stderr[-300:])
    except Exception as e:
        print('RULES_ERROR', repr(e)[:200])
    print('SYNC', sync(f'Session {day.date() if day is not None else ""}: daily rows, signals, evaluation, rules', rows))

MODEL_REPO = '/home/claude/tse-model'

def sync(msg, day_rows=None):
    """copy state to the tse-model repo and push (quietly skip on failure)"""
    import shutil
    try:
        os.makedirs(os.path.join(MODEL_REPO, 'state', 'daily'), exist_ok=True)
        for f in ['signals.csv', 'verdicts.csv', 'evaluation.csv', 'news.csv', 'news_impact.csv', 'positions.csv', 'tech_state.csv', 'divergence.csv', 'symbol_methods.csv', 'decisions_10m.csv', 'decision_eval.csv', 'metrics_daily.csv']:
            p = os.path.join(STATE, f)
            if os.path.exists(p): shutil.copy(p, os.path.join(MODEL_REPO, 'state', f))
        if day_rows is not None and len(day_rows):
            d = str(day_rows['date'].iloc[0])[:10]
            day_rows.to_csv(os.path.join(MODEL_REPO, 'state', 'daily', f'{d}.csv'), index=False)
        g = ['git', '-C', MODEL_REPO]
        subprocess.run(g + ['add', '-A'], check=True, timeout=60)
        if subprocess.run(g + ['diff', '--cached', '--quiet']).returncode == 0: return 'nothing to sync'
        subprocess.run(g + ['-c', 'user.name=Claude', '-c', 'user.email=noreply@anthropic.com', 'commit', '-qm',
                            msg + '\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01Wwa8GopZgXfw823bVrXVBr'], check=True, timeout=60)
        subprocess.run(g + ['fetch', '-q', 'origin', 'main'], timeout=120)
        subprocess.run(g + ['rebase', '-q', 'origin/main'], timeout=120)
        r = subprocess.run(g + ['push', '-q', 'origin', 'HEAD:main'], capture_output=True, text=True, timeout=180)
        return 'pushed' if r.returncode == 0 else 'push failed: ' + r.stderr[-200:]
    except Exception as e:
        return 'sync error: ' + repr(e)[:200]

if __name__ == '__main__':
    {'status': cmd_status, 'intraday': cmd_intraday, 'close': cmd_close}.get(sys.argv[1] if len(sys.argv) > 1 else 'status')()
