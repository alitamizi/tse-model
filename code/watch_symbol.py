"""Hourly in-session watch of one symbol (user request 23 Aban 1405: غدام, this week).
python3 watch_symbol.py غدام   -> prints WATCH line; SIGNAL=BUY only when all buy conditions hold.
Reads the bot's public files from raw.githubusercontent (no clone needed).
Buy conditions (monitoring rule, NOT back-tested on this symbol; based on 20 Aban study where day buyer power and
book imbalance were the best next-day predictors):
  day real buyer power >= 1.3, day real net inflow >= 10% of value, last-hour real net inflow >= 15% of last-hour value,
  last-hour price change >= 0, traded value >= 0.5 bn toman."""
import sys, io, json, urllib.parse, urllib.request, datetime as dt
import pandas as pd
SYM = sys.argv[1] if len(sys.argv) > 1 else 'غدام'
RAW = 'https://raw.githubusercontent.com/alitamizi/tse-data/main/data/'
def get(p): return urllib.request.urlopen(RAW + urllib.parse.quote(p), timeout=60).read()
def norm(s): return str(s).replace('ي', 'ی').replace('ك', 'ک').strip()
meta = json.loads(get('meta.json'))
lag = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(meta['fetched_at_utc'].replace('Z', '+00:00'))).total_seconds() / 60
L = pd.read_csv(io.BytesIO(get('latest.csv')), low_memory=False); L['نماد'] = L['نماد'].map(norm)
r = L[L['نماد'] == SYM].iloc[0]
H = pd.DataFrame(json.loads(get(f'history/{SYM}.json')))
today = dt.datetime.now(dt.timezone(dt.timedelta(hours=3, minutes=30))).date().isoformat()
H = H[H['زمان'].astype(str).str.startswith(today) & (H['زمان'].astype(str).str[11:16] >= '09:00')].sort_values('زمان')  # pre-open rows carry the previous day
val = r['ارزش معاملات']; net = r['ورود پول حقیقی']; bp = r['قدرت خریدار حقیقی']
nrv = net / val if val else float('nan')
h1 = dict(dp=float('nan'), nrv=float('nan'), val=0.0)
if len(H) >= 2:
    t_last = pd.Timestamp(H['زمان'].iloc[-1]); prev = H[pd.to_datetime(H['زمان']) <= t_last - pd.Timedelta(minutes=60)]
    a = prev.iloc[-1] if len(prev) else H.iloc[0]; b = H.iloc[-1]
    dv = b['ارزش معاملات'] - a['ارزش معاملات']
    h1 = dict(dp=b['آخرین معامله'] / a['آخرین معامله'] - 1 if a['آخرین معامله'] else float('nan'),
              nrv=(b['ورود پول حقیقی'] - a['ورود پول حقیقی']) / dv if dv > 0 else float('nan'), val=dv)
ask, bid = r.get('عرضه1 حجم', 0) or 0, r.get('تقاضا1 حجم', 0) or 0
queue = 'صف خرید' if r['آخرین معامله'] >= r['سقف مجاز'] and ask == 0 else ('صف فروش' if r['آخرین معامله'] <= r['کف مجاز'] and bid == 0 else '')
conds = {'قدرت خریدار روز>=1.3': bp >= 1.3, 'ورود حقیقی روز>=10%': nrv >= 0.10, 'ورود حقیقی ساعت اخیر>=15%': h1['nrv'] >= 0.15,
         'قیمت ساعت اخیر>=0': h1['dp'] >= 0, 'ارزش>=0.5 میلیارد تومان': val >= 5e9}
sig = 'BUY' if all(conds.values()) else 'NONE'
if lag > 20: sig = 'STALE'
print(f"WATCH {SYM} SIGNAL={sig} lag={lag:.0f}min | آخرین {r['آخرین معامله']:,.0f} ({r['درصد تغییر آخرین']:+.2f}%) پایانی {r['قیمت پایانی']:,.0f} "
      f"({r['درصد تغییر پایانی']:+.2f}%) دامنه {r['کف مجاز']:,.0f}-{r['سقف مجاز']:,.0f} | ارزش {val/1e10:,.1f} میلیارد تومان | "
      f"قدرت خریدار {bp:.2f} | ورود حقیقی {net/1e10:+,.1f} میلیارد تومان ({nrv:+.0%} ارزش) | ساعت اخیر: قیمت {h1['dp']:+.2%}، "
      f"ورود حقیقی {h1['nrv']:+.0%} ارزش، ارزش {h1['val']/1e10:,.1f} | دفتر: خرید۱ {bid:,.0f} فروش۱ {ask:,.0f} {queue}")
print('CONDS', ' | '.join(f"{k}:{'✓' if v else '✗'}" for k, v in conds.items()))
