"""Intraday feature panel for option underlyings (20 Aban study).
python3 opt_intraday.py build  -> live_state/opt_panel.pkl (one row per underlying x day x 10-min checkpoint)
Reads minute archive live_state/tse-data/archive/<day>.csv.xz and <day>_static.csv.xz. Run from /home/claude/multi."""
import sys, os, re, glob, numpy as np, pandas as pd
AR = 'live_state/tse-data/archive/'
NUM = ['آخرین معامله', 'قیمت پایانی', 'قیمت دیروز', 'ارزش معاملات', 'حجم معاملات', 'حجم خرید حقیقی', 'تعداد خرید حقیقی',
       'حجم فروش حقیقی', 'تعداد فروش حقیقی'] + [f'{s}{i} حجم' for s in ('تقاضا', 'عرضه') for i in range(1, 6)] + ['موقعیت‌های باز']
COLS = ['زمان دریافت', 'نماد', 'insCode'] + NUM
CK = [9 * 3600 + 20 * 60 + 600 * k for k in range(19)]          # 09:20 .. 12:20
END = 12 * 3600 + 35 * 60
def und(n):
    m = re.search(r'اختیار\s*([خف])\s*([^\-–]+?)\s*[-–]', str(n))
    return (m.group(1), m.group(2).strip()) if m else (None, None)
def norm(s): return str(s).replace('ي', 'ی').replace('ك', 'ک').strip()

def load(day):
    d = pd.read_csv(AR + day + '.csv.xz', usecols=lambda c: c in COLS, low_memory=False)
    d = d[d['زمان دریافت'] != 'زمان دریافت']
    for c in NUM:
        if c in d: d[c] = pd.to_numeric(d[c], errors='coerce')
    d['sec'] = pd.to_timedelta(d['زمان دریافت']).dt.total_seconds()
    d['sym'] = d['نماد'].map(norm); d['insCode'] = d['insCode'].astype(str)
    return d

def day_rows(day):
    st = pd.read_csv(AR + day + '_static.csv.xz', low_memory=False)
    st = st[st['insCode'].astype(str) != 'insCode']
    st['insCode'] = st['insCode'].astype(str)
    st[['kind', 'u']] = pd.DataFrame(st['نام'].map(und).tolist(), index=st.index)
    opt = st[st.u.notna()].set_index('insCode')[['kind', 'u']]
    d = load(day)
    unds = set(opt.u.map(norm))
    D = d[d.sym.isin(unds) | d.insCode.isin(opt.index)].copy()
    D['kind'] = D.insCode.map(opt.kind); D['u'] = D.insCode.map(opt.u).map(norm, na_action='ignore')
    D.loc[D.kind.isna(), 'u'] = D.sym
    D = D.sort_values('sec')
    def at(T):
        return D[D.sec <= T].groupby('insCode').tail(1).set_index('insCode')
    snaps = {T: at(T) for T in sorted(set(CK + [T - 600 for T in CK] + [T - 1800 for T in CK] + [T + 3600 for T in CK] + [END]))}
    rows = []
    for T in CK:
        a, b, c, e = snaps[T - 1800], snaps[T - 600], snaps[T], snaps[END]
        h = snaps.get(T + 3600, e)
        und_c = c[c.kind.isna()]
        for ic, r in und_c.iterrows():
            u = r.u; p = r['آخرین معامله']
            if not p or p <= 0 or r['قیمت دیروز'] <= 0: continue
            g = lambda s, col: s[col].get(ic, np.nan) if ic in s.index else np.nan
            p30, p10 = g(a, 'آخرین معامله'), g(b, 'آخرین معامله')
            dv = r['ارزش معاملات'] - (g(a, 'ارزش معاملات') if ic in a.index else 0)
            dq = r['حجم معاملات'] - (g(a, 'حجم معاملات') if ic in a.index else 0)
            avg = dv / dq if dq > 0 else p
            rb = r['حجم خرید حقیقی'] - (g(a, 'حجم خرید حقیقی') if ic in a.index else 0)
            rs = r['حجم فروش حقیقی'] - (g(a, 'حجم فروش حقیقی') if ic in a.index else 0)
            nb = r['تعداد خرید حقیقی'] - (g(a, 'تعداد خرید حقیقی') if ic in a.index else 0)
            ns = r['تعداد فروش حقیقی'] - (g(a, 'تعداد فروش حقیقی') if ic in a.index else 0)
            bidv = sum(r.get(f'تقاضا{i} حجم', 0) or 0 for i in range(1, 6)); askv = sum(r.get(f'عرضه{i} حجم', 0) or 0 for i in range(1, 6))
            o_c, o_a = c[c.u == u], a[a.u == u]
            ov = o_c['ارزش معاملات'] - o_a['ارزش معاملات'].reindex(o_c.index).fillna(0)
            cv, pv = ov[o_c.kind == 'خ'].sum(), ov[o_c.kind == 'ف'].sum()
            # option price momentum over 30 min, weighted by 30-min value
            opr = (o_c['آخرین معامله'] / o_a['آخرین معامله'].reindex(o_c.index) - 1).replace([np.inf, -np.inf], np.nan)
            w = ov.where(opr.notna(), 0)
            call_m = (opr[o_c.kind == 'خ'] * w[o_c.kind == 'خ']).sum() / w[o_c.kind == 'خ'].sum() if w[o_c.kind == 'خ'].sum() > 0 else np.nan
            put_m = (opr[o_c.kind == 'ف'] * w[o_c.kind == 'ف']).sum() / w[o_c.kind == 'ف'].sum() if w[o_c.kind == 'ف'].sum() > 0 else np.nan
            oi_c = o_c['موقعیت‌های باز'] - o_a['موقعیت‌های باز'].reindex(o_c.index)
            pe = g(e, 'آخرین معامله'); ph = g(h, 'آخرین معامله')
            rows.append(dict(day=day, T=T, u=u, p=p, yday=r['قیمت دیروز'], ret_d=p / r['قیمت دیروز'] - 1,
                r10=p / p10 - 1 if p10 else np.nan, r30=p / p30 - 1 if p30 else np.nan,
                val30=dv, flow30=(rb - rs) * avg / dv if dv > 0 else np.nan,
                bp30=(rb / nb) / (rs / ns) if nb > 0 and ns > 0 and rs > 0 else np.nan,
                flow_d=(r['حجم خرید حقیقی'] - r['حجم فروش حقیقی']) * r['ارزش معاملات'] / r['حجم معاملات'] / r['ارزش معاملات'] if r['حجم معاملات'] > 0 else np.nan,
                ob1=((r['تقاضا1 حجم'] or 0) - (r['عرضه1 حجم'] or 0)) / ((r['تقاضا1 حجم'] or 0) + (r['عرضه1 حجم'] or 0)) if (r['تقاضا1 حجم'] or 0) + (r['عرضه1 حجم'] or 0) > 0 else np.nan,
                bp_d=(r['حجم خرید حقیقی'] / r['تعداد خرید حقیقی']) / (r['حجم فروش حقیقی'] / r['تعداد فروش حقیقی']) if r['تعداد خرید حقیقی'] > 0 and r['تعداد فروش حقیقی'] > 0 and r['حجم فروش حقیقی'] > 0 else np.nan,
                ob=(bidv - askv) / (bidv + askv) if bidv + askv > 0 else np.nan,
                cp_val=(cv - pv) / (cv + pv) if cv + pv > 0 else np.nan, opt_val30=cv + pv,
                call_m=call_m, put_m=put_m, oi_call=oi_c[o_c.kind == 'خ'].sum(), oi_put=oi_c[o_c.kind == 'ف'].sum(),
                p_end=pe, close_end=g(e, 'قیمت پایانی'), p_h60=ph))
    return pd.DataFrame(rows)

if __name__ == '__main__' and sys.argv[1] == 'build':
    days = sorted(os.path.basename(f)[:10] for f in glob.glob(AR + '20??-??-??.csv.xz'))
    out = []
    for dday in days:
        if not os.path.exists(AR + dday + '_static.csv.xz'): print('no static', dday); continue
        try: x = day_rows(dday); out.append(x); print(dday, len(x), x.u.nunique(), flush=True)
        except Exception as ex: print('ERR', dday, ex, flush=True)
    P = pd.concat(out, ignore_index=True); P.to_pickle('live_state/opt_panel.pkl'); print(P.shape)
