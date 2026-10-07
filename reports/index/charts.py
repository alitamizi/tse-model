import pandas as pd, numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm, matplotlib.ticker as mt
from arabic_reshaper import reshape
from bidi.algorithm import get_display
for w in ('400', '700'): fm.fontManager.addfont(f'/home/claude/idx/Vazirmatn-{w}.ttf')
plt.rcParams.update({'font.family': 'Vazirmatn', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.titlesize': 12, 'axes.titleweight': 'bold', 'figure.dpi': 110, 'savefig.dpi': 170, 'axes.titlelocation': 'right'})
P = lambda s: get_display(reshape(s))
INK, BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL = '#1f2a44', '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#94a3b8', '#7c3aed', '#0d9488'
F = pd.read_pickle('full.pkl'); c = F.v.astype(float); J = F.j; I = pd.read_pickle('ind.pkl'); S = json.load(open('stats.json'))
W = pd.read_pickle('weekly.pkl'); Z = pd.read_pickle('zz12.pkl'); Z7 = pd.read_pickle('zz7.pkl'); ICH = pd.read_pickle('ichi.pkl'); BR = pd.read_pickle('breadth.pkl'); A50 = pd.read_pickle('above50.pkl')
def jticks(ax, idx, every='year'):
    jj = J.reindex(idx).dropna()
    if every == 'year': key = jj.str[:4]; lab = lambda k: k
    else: key = jj.str[:7]; lab = lambda k: k[:4] + '/' + k[5:7]
    first = jj.groupby(key).apply(lambda s: s.index[0])
    if every != 'year' and len(first) > 14: first = first.iloc[::2]
    keep = []; 
    for k, v in first.items():
        if not keep or (v - keep[-1][1]).days > 25: keep.append((k, v))
    first = pd.Series({k: v for k, v in keep})
    ax.set_xticks(list(first.values)); ax.set_xticklabels([lab(k) for k in first.index], fontsize=8)
def fmtk(ax):
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, p: f'{v/1e6:.2f}M' if v >= 1e6 else f'{v/1e3:.0f}K'))
def save(fig, n): fig.tight_layout(); fig.savefig(f'ch/{n}.png'); plt.close(fig)
last = c.iloc[-1]
# 1 overview (log) + USD
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.4), height_ratios=[3, 2], sharex=True)
a.semilogy(c.index, c, color=INK, lw=1.2); a.set_title(P('شاخص قیمت وزنی-ارزشی (مقیاس لگاریتمی، ریالی)')); fmtk(a); a.yaxis.set_minor_formatter(mt.NullFormatter()); a.set_ylim(top=c.max() * 2.2)
for _, z in Z[Z.type != 0].iterrows():
    if abs(z.chg) > 0.25 or z.t == c.index[-1]:
        a.annotate(f'{z.v/1e3:,.0f}K\n{z.j[2:]}', (z.t, z.v), xytext=(0, 12 if z.type == 1 else -26), textcoords='offset points', ha='center', fontsize=7, color=GREEN if z.type == 1 else RED)
b.plot(F.index, F.c_usd / F.c_usd.iloc[0] * 100, color=ORANGE, lw=1.2); b.axhline(100, color=GRAY, lw=.8, ls='--')
b.set_title(P('همان شاخص به دلار آزاد (۱۰۰ = فروردین ۱۳۹۴)')); jticks(b, c.index)
save(fig, 'p01_overview')
# yearly returns rial vs usd
yr = c.groupby(J.str[:4]).last(); yu = F.c_usd.groupby(J.str[:4]).last()
ry = (yr / yr.shift() - 1).dropna(); ru = (yu / yu.shift() - 1).dropna()
fig, a = plt.subplots(figsize=(10, 3.4)); x = np.arange(len(ry))
a.bar(x - .2, ry * 100, .4, color=BLUE, label=P('ریالی')); a.bar(x + .2, ru * 100, .4, color=ORANGE, label=P('دلاری'))
for i, (r1, r2) in enumerate(zip(ry, ru)): a.text(i - .2, r1 * 100 + (4 if r1 > 0 else -12), f'{r1*100:.0f}', ha='center', fontsize=7); a.text(i + .2, r2 * 100 + (4 if r2 > 0 else -12), f'{r2*100:.0f}', ha='center', fontsize=7, color=ORANGE)
a.set_xticks(x); a.set_xticklabels(list(ry.index)); a.axhline(0, color=INK, lw=.8); a.legend(); a.set_title(P('بازده سالانه (٪) — ۱۴۰۵ تا ۱۵ مهر'))
save(fig, 'p01_years')
# 2 regression channel
rg = pd.read_pickle('reg.pkl'); t = np.arange(len(c)); fit = np.exp(rg['b0'] + rg['b1'] * t)
fig, a = plt.subplots(figsize=(10, 4.8)); a.semilogy(c.index, c, color=INK, lw=1.1, label=P('شاخص')); a.yaxis.set_minor_formatter(mt.NullFormatter())
for k, col, lab in [(0, BLUE, 'خط روند بلندمدت'), (1, GRAY, '±۱ انحراف معیار'), (2, RED, '±۲ انحراف معیار')]:
    a.semilogy(c.index, fit * np.exp(k * rg['sd']), color=col, lw=1, ls='-' if k == 0 else '--', label=P(lab))
    if k: a.semilogy(c.index, fit * np.exp(-k * rg['sd']), color=col, lw=1, ls='--')
a.set_title(P('کانال رگرسیون لگاریتمی بلندمدت (۱۳۹۴ تا امروز)')); fmtk(a); jticks(a, c.index); a.legend(loc='upper left', fontsize=8)
save(fig, 'p02_reg')
# z-score history
z = (np.log(c.values) - np.log(fit)) / rg['sd']
fig, a = plt.subplots(figsize=(10, 2.4)); a.plot(c.index, z, color=PURPLE, lw=1); a.axhline(0, color=GRAY); a.axhline(2, color=RED, ls='--', lw=.8); a.axhline(-2, color=GREEN, ls='--', lw=.8)
a.set_title(P('فاصلهٔ شاخص از خط روند بلندمدت (بر حسب انحراف معیار)')); jticks(a, c.index); save(fig, 'p02_z')
# 3 MAs (last 3 years) + distance from MA200
s0 = c.index[-720]
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1.6], sharex=True)
a.plot(c[s0:].index, c[s0:], color=INK, lw=1.2, label=P('شاخص'))
for n, col in [(50, BLUE), (100, ORANGE), (200, RED)]: a.plot(I[s0:].index, I[f'ma{n}'][s0:], color=col, lw=1, label=f'MA{n}')
a.legend(fontsize=8, loc='upper left'); a.set_title(P('میانگین‌های متحرک ۵۰، ۱۰۰ و ۲۰۰ روزه (سه سال اخیر)')); fmtk(a)
dd = (c / I.ma200 - 1) * 100; b.fill_between(dd.index, dd, 0, where=dd > 0, color=GREEN, alpha=.35); b.fill_between(dd.index, dd, 0, where=dd < 0, color=RED, alpha=.35)
b.set_xlim(s0, c.index[-1]); b.set_title(P('فاصله از MA200 (٪)')); jticks(b, c[s0:].index, 'month'); save(fig, 'p03_ma')
fig, a = plt.subplots(figsize=(10, 2.6)); a.plot(dd.index, dd, color=TEAL, lw=1); a.axhline(dd.iloc[-1], color=RED, ls='--', lw=.8); a.text(dd.index[50], dd.iloc[-1] + 6, P(f'امروز: {dd.iloc[-1]:.0f}٪'), color=RED, fontsize=8)
a.set_title(P('فاصله از MA200 در کل تاریخچه (٪)')); jticks(a, c.index); save(fig, 'p03_dist')
# 4 support/resistance & swings (2 years)
s1 = c.index[-480]; cc = c[s1:]
fig, a = plt.subplots(figsize=(10, 5)); a.plot(cc.index, cc, color=INK, lw=1.2)
zz = Z7[Z7.t >= s1]; a.plot(zz.t, zz.v, color=GRAY, lw=.8, ls=':')
for _, r in zz.iterrows():
    if r.type == 0: continue
    a.annotate(f'{r.v/1e3:,.0f}K', (r.t, r.v), xytext=(0, 8 if r.type == 1 else -14), textcoords='offset points', ha='center', fontsize=7, color=GREEN if r.type == 1 else RED)
levels = [(914721, 'سقف تیر ۱۴۰۵ (۹۱۵K) — حمایت جدید'), (787658, 'سقف دی ۱۴۰۴ (۷۸۸K)'), (1072695, 'فیبوناچی ۳۸٫۲٪ (۱٫۰۷M)'), (1176476, 'فیبوناچی ۲۳٫۶٪ (۱٫۱۸M)'), (1256420, 'سقف ۲۵ شهریور (۱٫۲۶M)')]
for v, lab in levels: a.axhline(v, color=BLUE, lw=.8, ls='--', alpha=.7); a.text(cc.index[5], v * 1.01, P(lab), fontsize=7, color=BLUE)
a.set_title(P('سطوح حمایت و مقاومت و چرخش‌های ۷٪+ (دو سال اخیر)')); fmtk(a); jticks(a, cc.index, 'month'); save(fig, 'p04_sr')
# 5 trend channel of current leg (log-linear fit since 1405-04-28 low)
t0 = Z7[Z7.type == -1].iloc[-1].t; seg = c[t0:]; tt = np.arange(len(seg)); bb, aa = np.polyfit(tt, np.log(seg.values), 1); fr = np.exp(aa + bb * tt); rs = np.log(seg.values) - np.log(fr)
up_, dn_ = rs.max(), rs.min()
fig, a = plt.subplots(figsize=(10, 4.6)); s2 = c.index[-160]; a.plot(c[s2:].index, c[s2:], color=INK, lw=1.3)
a.plot(seg.index, fr, color=BLUE, lw=1); a.plot(seg.index, fr * np.exp(up_), color=RED, ls='--', lw=1); a.plot(seg.index, fr * np.exp(dn_), color=GREEN, ls='--', lw=1)
# medium trendline: from 1404-12-03 low through 1405-04-28 low
p1 = Z[Z.type == -1].iloc[-1]; p2 = Z7[Z7.type == -1].iloc[-1]
i1, i2 = c.index.get_loc(p1.t), c.index.get_loc(p2.t); sl = (np.log(p2.v) - np.log(p1.v)) / (i2 - i1); xi = np.arange(i1, len(c)); tl = np.exp(np.log(p1.v) + sl * (xi - i1))
a.plot(c.index[xi], tl, color=ORANGE, lw=1.2, label=P('خط روند میان‌مدت (کف اسفند ۱۴۰۴ تا کف تیر ۱۴۰۵)'))
a.legend(fontsize=8, loc='upper left'); a.set_title(P('کانال صعودی موج جاری (از کف ۲۸ تیر ۱۴۰۵)')); fmtk(a); jticks(a, c[s2:].index, 'month'); save(fig, 'p05_channel')
S['channel'] = dict(slope_day=float(bb), ann=float(np.exp(bb * 240) - 1), top=float(fr[-1] * np.exp(up_)), mid=float(fr[-1]), bot=float(fr[-1] * np.exp(dn_)), trend_mid=float(tl[-1]), start=J[t0])
# 6 Fibonacci
fb = S['fib']; s3 = c.index[c.index.get_loc(pd.Timestamp(Z[Z.type == -1].iloc[-1].t)) - 40]
fig, a = plt.subplots(figsize=(10, 4.8)); a.plot(c[s3:].index, c[s3:], color=INK, lw=1.2)
for r, v in fb['retr'].items(): a.axhline(v, color=PURPLE, lw=.8, ls='--'); a.text(c[s3:].index[3], v * 1.008, f'{float(r)*100:.1f}%  {v/1e3:,.0f}K', fontsize=7, color=PURPLE)
for r, v in fb['ext'].items(): a.axhline(v, color=TEAL, lw=.8, ls=':'); a.text(c[s3:].index[3], v * 1.008, f'{float(r)*100:.1f}%  {v/1e3:,.0f}K', fontsize=7, color=TEAL)
a.axhline(fb['low'], color=GRAY, lw=.8); a.axhline(fb['high'], color=GRAY, lw=.8)
a.set_title(P('بازگشت‌ها و گسترش‌های فیبوناچی موج کف اسفند ۱۴۰۴ تا امروز')); fmtk(a); jticks(a, c[s3:].index, 'month'); save(fig, 'p06_fib')
# 7 RSI daily (1y) + weekly
s4 = c.index[-260]
fig, (a, b, e) = plt.subplots(3, 1, figsize=(10, 6.6), height_ratios=[2, 1.2, 1.2], sharex=False)
a.plot(c[s4:].index, c[s4:], color=INK, lw=1.2); a.set_title(P('شاخص و RSI روزانه (یک سال اخیر)')); fmtk(a); jticks(a, c[s4:].index, 'month')
for dv in S['divs'][-2:-1]:
    t1 = J[J == dv['p1']].index[0]; t2 = J[J == dv['p2']].index[0]
    a.plot([t1, t2], [dv['v1'], dv['v2']], color=RED, lw=1.5); b.plot([t1, t2], [dv['rsi1'], dv['rsi2']], color=RED, lw=1.5)
b.plot(I[s4:].index, I.rsi[s4:], color=PURPLE, lw=1); b.axhline(70, color=RED, ls='--', lw=.8); b.axhline(30, color=GREEN, ls='--', lw=.8); b.set_ylim(10, 100); b.set_title('RSI(14) ' + P('روزانه')); jticks(b, c[s4:].index, 'month')
ws = W.index[-260:]; e.plot(ws, W.rsi[-260:], color=TEAL, lw=1); e.axhline(70, color=RED, ls='--', lw=.8); e.axhline(30, color=GREEN, ls='--', lw=.8); e.set_ylim(10, 100); e.set_title('RSI(14) ' + P('هفتگی (پنج سال اخیر)'))
Jn = J.reindex(W.index[-260:], method='nearest'); fy = Jn.groupby(Jn.str[:4]).apply(lambda s: s.index[0]); e.set_xticks(list(fy.values)); e.set_xticklabels(list(fy.index), fontsize=8)
save(fig, 'p07_rsi')
# 8 MACD daily & weekly
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5.4))
a.plot(I[s4:].index, I.macd[s4:], color=BLUE, lw=1, label='MACD'); a.plot(I[s4:].index, I.sig[s4:], color=ORANGE, lw=1, label=P('سیگنال'))
a.bar(I[s4:].index, I['hist'][s4:], color=np.where(I['hist'][s4:] > 0, GREEN, RED), width=1.2, alpha=.5); a.legend(fontsize=8, loc='upper left'); a.set_title(P('MACD روزانه (۱۲، ۲۶، ۹) — یک سال اخیر')); jticks(a, c[s4:].index, 'month')
b.plot(ws, W.macd[-260:], color=BLUE, lw=1); b.plot(ws, W.sig[-260:], color=ORANGE, lw=1); hh = W.macd[-260:] - W.sig[-260:]; b.bar(ws, hh, color=np.where(hh > 0, GREEN, RED), width=5, alpha=.5); b.set_title(P('MACD هفتگی — پنج سال اخیر'))
save(fig, 'p08_macd')
# 9 Bollinger
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5.6), height_ratios=[3, 1.3], sharex=True)
a.plot(c[s4:].index, c[s4:], color=INK, lw=1.2); a.plot(I[s4:].index, I.bbm[s4:], color=BLUE, lw=.9); a.fill_between(I[s4:].index, I.bbl[s4:], I.bbu[s4:], color=BLUE, alpha=.12)
a.set_title(P('باندهای بولینگر (۲۰ روزه، ±۲ انحراف معیار)')); fmtk(a)
b.plot(I[s4:].index, I.bw[s4:] * 100, color=ORANGE, lw=1); b.set_title(P('پهنای باند (٪)')); jticks(b, c[s4:].index, 'month'); save(fig, 'p09_bb')
# 10 Ichimoku
s5 = c.index[-200]; ich = ICH[s5:]
fig, a = plt.subplots(figsize=(10, 4.8)); a.plot(ich.index, ich.c, color=INK, lw=1.3, label=P('شاخص'))
a.plot(ich.index, ich.ten, color=BLUE, lw=1, label=P('تنکان‌سن (۹)')); a.plot(ich.index, ich.kij, color=RED, lw=1, label=P('کیجون‌سن (۲۶)'))
a.fill_between(ich.index, ich.sa, ich.sb, where=ich.sa >= ich.sb, color=GREEN, alpha=.18); a.fill_between(ich.index, ich.sa, ich.sb, where=ich.sa < ich.sb, color=RED, alpha=.18)
# future cloud
hh_ = lambda n: c.rolling(n).max(); ll_ = lambda n: c.rolling(n).min(); ten = (hh_(9) + ll_(9)) / 2; kij = (hh_(26) + ll_(26)) / 2
fsa = ((ten + kij) / 2).iloc[-26:].values; fsb = ((hh_(52) + ll_(52)) / 2).iloc[-26:].values
fut = pd.bdate_range(c.index[-1] + pd.Timedelta(days=1), periods=26, freq='C', weekmask='Sat Sun Mon Tue Wed')
a.fill_between(fut, fsa, fsb, color=GREEN, alpha=.10, hatch='//'); a.legend(fontsize=8, loc='upper left'); a.set_title(P('ابر ایچیموکو (تقریبی؛ با قیمت پایانی)')); fmtk(a); jticks(a, ich.index, 'month'); save(fig, 'p10_ichi')
# 11 momentum ROC
fig, a = plt.subplots(figsize=(10, 3.8))
for h, col in [(20, BLUE), (60, ORANGE), (240, PURPLE)]:
    r = (c / c.shift(h) - 1) * 100; a.plot(r.index, r, color=col, lw=.9, label=P(f'بازده {h} روزه'))
a.axhline(0, color=INK, lw=.8); a.legend(fontsize=8, loc='upper left'); a.set_title(P('مومنتوم (نرخ تغییر) — کل تاریخچه، ٪')); jticks(a, c.index); a.set_ylim(-60, 300); save(fig, 'p11_mom')
# 12 volatility
fig, a = plt.subplots(figsize=(10, 3.4)); a.plot(I.index, I.vol20 * 100, color=RED, lw=.8, alpha=.7, label=P('نوسان ۲۰ روزه')); a.plot(I.index, I.vol60 * 100, color=INK, lw=1.1, label=P('نوسان ۶۰ روزه'))
a.legend(fontsize=8, loc='upper left'); a.set_title(P('نوسان سالانه‌شده (٪)')); jticks(a, c.index); save(fig, 'p12_vol')
# 13 drawdowns
fig, a = plt.subplots(figsize=(10, 3.4)); a.fill_between(I.index, I.dd * 100, 0, color=RED, alpha=.4); a.set_title(P('افت از سقف قبلی (٪) — ریالی'))
du = F.c_usd / F.c_usd.cummax() - 1; a.plot(du.index, du * 100, color=ORANGE, lw=1, label=P('افت از سقف به دلار')); a.legend(fontsize=8, loc='lower left'); jticks(a, c.index); save(fig, 'p13_dd')
# 14 distribution & autocorrelation
lr = np.log(c).diff().dropna()
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
a.hist(lr * 100, bins=80, color=BLUE, alpha=.7, density=True); xs = np.linspace(-5, 5, 200); a.plot(xs, np.exp(-(xs - lr.mean() * 100) ** 2 / (2 * (lr.std() * 100) ** 2)) / (lr.std() * 100 * np.sqrt(2 * np.pi)), color=RED, lw=1)
a.set_title(P('توزیع بازده روزانه (٪) و توزیع نرمال')); 
ac = [lr.autocorr(k) for k in range(1, 21)]; b.bar(range(1, 21), ac, color=PURPLE); b.axhline(2 / np.sqrt(len(lr)), color=GRAY, ls='--'); b.axhline(-2 / np.sqrt(len(lr)), color=GRAY, ls='--'); b.set_title(P('خودهمبستگی بازده روزانه (تأخیر ۱ تا ۲۰ روز)'))
save(fig, 'p14_dist')
# 15 seasonality
mm = pd.DataFrame(S['months']).T; names = ['فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور', 'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند']
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), width_ratios=[2.2, 1])
a.bar(range(12), mm['median'] * 100, color=[GREEN if v > 0 else RED for v in mm['median']]); pass
a.set_xticks(range(12)); a.set_xticklabels([P(n) for n in names], rotation=45, fontsize=8); a.set_title(P('میانهٔ بازده ماه‌های شمسی (٪) ۱۳۹۴–۱۴۰۵'))
dw = S['dow']; order = [5, 6, 0, 1, 2]; dn = ['شنبه', 'یکشنبه', 'دوشنبه', 'سه‌شنبه', 'چهارشنبه']
b.bar(range(5), [dw[str(k)][1] * 100 if str(k) in dw else dw[k][1] * 100 for k in order], color=TEAL); b.axhline(50, color=GRAY, ls='--'); b.set_ylim(40, 65)
b.set_xticks(range(5)); b.set_xticklabels([P(n) for n in dn], fontsize=8, rotation=45); b.set_title(P('درصد روزهای مثبت'))
save(fig, 'p15_season')
# 16 cycles: major lows/highs & bubble analog (1398-1399 vs now) aligned on start of leg
fig, a = plt.subplots(figsize=(10, 4.4))
def leg(t0, n=320):
    i = c.index.get_loc(t0); s = c.iloc[i:i + n]; return s.values / s.values[0]
L1 = leg(pd.Timestamp('2018-12-22')); L2 = leg(pd.Timestamp('2022-10-26'), 260); L3 = leg(pd.Timestamp('2026-02-22'))
a.semilogy(range(len(L1)), L1, color=GRAY, lw=1.2, label=P('از کف دی ۱۳۹۷ (حباب ۱۳۹۹)')); a.semilogy(range(len(L2)), L2, color=ORANGE, lw=1.2, label=P('از کف آبان ۱۴۰۱'))
a.semilogy(range(len(L3)), L3, color=RED, lw=2, label=P('از کف اسفند ۱۴۰۴ (امروز)')); a.legend(fontsize=8, loc='upper left'); a.set_xlabel(P('روز معاملاتی از کف')); a.set_title(P('مقایسهٔ موج‌های صعودی بزرگ (۱ = کف)'))
a.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, p: f'{v:.1f}x')); a.yaxis.set_minor_formatter(mt.NullFormatter()); a.set_yticks([1, 1.5, 2, 3, 4]); save(fig, 'p16_analog')
# 17 breadth
b0 = c.index[-480]; br = BR.reindex(c.index)[b0:]
fig, (a, b, e) = plt.subplots(3, 1, figsize=(10, 6.6), sharex=True)
cw = c[b0:] / c[b0]; ew = br.EW / br.EW.dropna().iloc[0]
a.plot(cw.index, cw, color=INK, lw=1.2, label=P('شاخص وزنی-ارزشی')); a.plot(ew.index, ew, color=ORANGE, lw=1.2, label=P('میانگین هم‌وزن سهم‌ها (محاسبهٔ من)')); a.legend(fontsize=8, loc='upper left'); a.set_title(P('وزنی در برابر هم‌وزن (۱ = دو سال پیش)'))
b.plot(A50[b0:].index, A50[b0:] * 100, color=TEAL, lw=1); b.axhline(80, color=RED, ls='--', lw=.8); b.axhline(20, color=GREEN, ls='--', lw=.8); b.set_title(P('درصد سهم‌های بالای میانگین ۵۰ روزهٔ خودشان'))
adl = (br.adv - br.dec).fillna(0).cumsum(); e.plot(adl.index, adl, color=PURPLE, lw=1); e.set_title(P('خط پیشروی-پسروی (تجمعی تعداد مثبت منهای منفی)')); jticks(e, c[b0:].index, 'month'); save(fig, 'p17_breadth')
# 18 value & real money
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True)
vv = F.val / 1e13; a.plot(vv.index, vv.rolling(20).mean(), color=BLUE, lw=1.1, label=P('ارزش معاملات، میانگین ۲۰ روزه (همت)')); a.set_yscale('log'); a.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, p: f'{v:g}')); a.yaxis.set_minor_formatter(mt.NullFormatter()); a.set_title(P('ارزش معاملات روزانهٔ بورس (هزار میلیارد تومان، لگاریتمی)'))
vu = F.val / F.usd / 1e6; a2 = a.twinx(); a2.plot(vu.index, vu.rolling(20).mean(), color=ORANGE, lw=1, alpha=.8); a2.set_ylabel(P('میلیون دلار'), color=ORANGE); a2.grid(False); a.legend(fontsize=8, loc='upper left')
cum = (F.rnet.fillna(0) / 1e13).cumsum(); b.plot(cum.index, cum, color=GREEN, lw=1.1); b.set_title(P('ورود تجمعی پول حقیقی به بورس (همت)')); jticks(b, c.index); save(fig, 'p18_value')
fig, a = plt.subplots(figsize=(10, 3.2)); s6 = c.index[-250]; rn = F.rnet[s6:] / 1e13
a.bar(rn.index, rn, color=np.where(rn > 0, GREEN, RED), width=1.2); a2 = a.twinx(); a2.plot(c[s6:].index, c[s6:], color=INK, lw=1); a2.grid(False); fmtk(a2)
a.set_title(P('ورود روزانهٔ پول حقیقی (همت، ستون‌ها) و شاخص (خط) — یک سال اخیر')); jticks(a, c[s6:].index, 'month'); save(fig, 'p18_flow')
# 19 waves (zigzag 12% full + annotate)
fig, a = plt.subplots(figsize=(10, 4.6)); s7 = c.index[-900]; a.semilogy(c[s7:].index, c[s7:], color=GRAY, lw=.9)
zz = Z[Z.t >= s7]; a.semilogy(zz.t, zz.v, color=INK, lw=1.4, marker='o', ms=3)
for i, (_, r) in enumerate(zz.iterrows()):
    if r.type == 0: continue
    a.annotate(f'{r.v/1e3:,.0f}K\n{r.j[2:]}', (r.t, r.v), xytext=(0, 10 if r.type == 1 else -26), textcoords='offset points', ha='center', fontsize=7)
a.set_ylim(bottom=c[s7:].min() * 0.78); a.set_title(P('ساختار موجی (زیگزاگ ۱۲٪) — سه و نیم سال اخیر')); fmtk(a); a.yaxis.set_minor_formatter(mt.NullFormatter()); jticks(a, c[s7:].index); save(fig, 'p19_waves')
# 20 backtests
EQ = pd.read_pickle('bt_eq.pkl'); fig, a = plt.subplots(figsize=(10, 4))
cols = [INK, RED, ORANGE, BLUE, PURPLE, TEAL, GRAY]
for (k, v), col in zip(EQ.items(), cols): a.semilogy(v.index, v, color=col, lw=1.1 if k != 'خرید و نگهداری' else 1.8, label=P(k))
a.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, p: f'{v:g}x')); a.yaxis.set_minor_formatter(mt.NullFormatter()); a.legend(fontsize=7, loc='upper left'); a.set_title(P('آزمون تاریخی قواعد روندی روی خود شاخص (۱ = فروردین ۱۳۹۴، بدون کارمزد)')); jticks(a, c.index); save(fig, 'p20_bt')
# 21 USD index zoom
fig, a = plt.subplots(figsize=(10, 3.6)); a.plot(F.index, F.mcap_usd / 1e9, color=ORANGE, lw=1.2); a.set_title(P('ارزش کل بازار بورس به دلار آزاد (میلیارد دلار)')); jticks(a, c.index)
a2 = a.twinx(); a2.plot(F.index, F.usd / 10, color=GRAY, lw=.8, alpha=.7); a2.set_ylabel(P('دلار آزاد (تومان)'), color=GRAY); a2.grid(False); a2.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, p: f'{v/1e3:.0f}K'))
save(fig, 'p21_usd')
json.dump(S, open('stats.json', 'w'), ensure_ascii=False, indent=1, default=str)
print('ok', S['channel'])
