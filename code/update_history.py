"""Append the newest sessions to history/<symbol>.zip (one row per trading day, same 13 columns, adjusted prices).

Usage (run inside /home/claude/multi after `live.py close`, so panel.pkl holds the latest adjusted daily panel):
    python3 update_history.py [panel.pkl] [--dry]

How it works
- The daily panel X (panel.pkl) is already adjusted for capital increases and cash dividends (live.append_day rescales the
  older rows when the exchange's reference price differs from the last close).
- For each zip: find the last date L in the file. If L is in the panel, ratio = panel close(L) / file close(L). A ratio
  different from 1 by more than 0.3% means a corporate action happened after the file was written, so the older rows of the
  file (o, h, l, c, last) are multiplied by that ratio. Rows after L are then appended from the panel.
- A ratio outside 0.3..1.1 is not a normal adjustment (e.g. a re-listing with a new reference price). That symbol is skipped
  and reported, because rescaling would corrupt its history.
- Old rows are kept byte-for-byte unless a rescale is needed.
"""
import os, sys, io, glob, zipfile
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(REPO, 'history')
COLS = ['تاریخ میلادی', 'تاریخ شمسی', 'اولین', 'بیشترین', 'کمترین', 'پایانی', 'آخرین', 'حجم', 'ارزش (ریال)',
        'تعداد معاملات', 'قدرت خریدار حقیقی', 'ورود پول حقیقی (ریال)', 'ارزش بازار (ریال)']
SRC = {'اولین': 'o', 'بیشترین': 'h', 'کمترین': 'l', 'پایانی': 'c', 'آخرین': 'last', 'حجم': 'vol', 'ارزش (ریال)': 'value',
       'تعداد معاملات': 'ntr', 'قدرت خریدار حقیقی': 'bp', 'ورود پول حقیقی (ریال)': 'net', 'ارزش بازار (ریال)': 'mcap'}
PRICE_IDX = [2, 3, 4, 5, 6]          # positions of o, h, l, c, last in COLS
ROUND3 = {'اولین', 'بیشترین', 'کمترین', 'پایانی', 'آخرین', 'قدرت خریدار حقیقی', 'ورود پول حقیقی (ریال)'}
TOL, LO, HI = 0.003, 0.3, 1.1
MONTHS = ['فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور', 'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند']


def norm(s):
    return str(s).replace('‌', '').replace(' ', '').replace('ي', 'ی').replace('ك', 'ک')


def num(v, r3=False):
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return ''
    v = float(v)
    return repr(round(v, 3) if r3 else v)


def fmt_row(d, r):
    out = [d.strftime('%Y-%m-%d'), str(r['jdate'])]
    for c in COLS[2:]:
        out.append(num(r[SRC[c]], c in ROUND3))
    return ','.join(out)


def jalali_line(jdate):
    y, m, d = (int(p) for p in jdate.split('-'))
    return str(d).translate(str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')) + ' ' + MONTHS[m - 1] + ' ' + \
        str(y).translate(str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹'))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    dry = '--dry' in sys.argv
    X, _ = pd.read_pickle(args[0] if args else 'panel.pkl')
    by = {norm(k): k for k in X}
    latest = max(x.index[-1] for x in X.values())
    stats = dict(files=0, appended=0, rows=0, rescaled=0, skipped=[], current=0)
    table = []
    for path in sorted(glob.glob(os.path.join(HIST, '*.zip'))):
        sym = os.path.basename(path)[:-4]
        z = zipfile.ZipFile(path)
        name = z.namelist()[0]
        text = z.read(name).decode('utf-8-sig')
        z.close()
        lines = text.split('\n')
        header, body = lines[0], [l for l in lines[1:] if l.strip()]
        last = pd.Timestamp(body[-1].split(',')[0])
        k = by.get(norm(sym))
        new_lines, ratio = [], 1.0
        if k is not None:
            x = X[k]
            newer = x[x.index > last]
            if len(newer):
                if last not in x.index:
                    stats['skipped'].append((sym, 'نقطهٔ اتصال در پنل نیست'))
                    newer = newer.iloc[0:0]
                else:
                    old_c = float(body[-1].split(',')[5])
                    ratio = float(x.c.loc[last]) / old_c if old_c > 0 else 1.0
                    if not (LO <= ratio <= HI):
                        stats['skipped'].append((sym, f'نسبت تعدیل غیرعادی {ratio:.4f}'))
                        newer = newer.iloc[0:0]
                        ratio = 1.0
                for d, r in newer.iterrows():
                    if float(r['ntr']) > 0:
                        new_lines.append(fmt_row(d, r))
            if abs(ratio - 1) > TOL:
                rescaled = []
                for l in body:
                    p = l.split(',')
                    for i in PRICE_IDX:
                        if p[i] != '':
                            p[i] = repr(round(float(p[i]) * ratio, 3))
                    rescaled.append(','.join(p))
                body = rescaled
                stats['rescaled'] += 1
        if new_lines or abs(ratio - 1) > TOL:
            body = body + new_lines
            stats['appended'] += 1 if new_lines else 0
            stats['rows'] += len(new_lines)
            if not dry:
                data = ('﻿' + header + '\n' + '\n'.join(body) + '\n').encode('utf-8')
                tmp = path + '.tmp'
                with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zo:
                    zo.writestr(name, data[3:] if False else data)
                os.replace(tmp, path)
        else:
            stats['current'] += 1
        stats['files'] += 1
        table.append((sym, body[0].split(',')[1], body[-1].split(',')[1], len(body)))
    readme = os.path.join(HIST, 'README.md')
    if not dry and os.path.exists(readme):
        from urllib.parse import quote
        txt = open(readme, encoding='utf-8').read().split('\n')
        head = [l for l in txt if not l.startswith('| ')]
        i = next(n for n, l in enumerate(txt) if l.startswith('| نماد'))
        pre = [l for l in txt[:i]]
        jd = x.jdate.iloc[-1] if False else None
        last_j = max(t[2] for t in table)
        pre = [('به‌روزرسانی: ' + jalali_line(last_j) + '.') if l.startswith('به‌روزرسانی:') else l for l in pre]
        rows = [f'| [{s}]({quote(s)}.zip) | {a} | {b} | {n:,} |' for s, a, b, n in table]
        open(readme, 'w', encoding='utf-8').write('\n'.join(pre + ['| نماد | از | تا | تعداد روز |', '|---|---|---|---|'] + rows) + '\n')
    print(f"HISTORY latest_panel_date={latest.date()} files={stats['files']} updated_with_new_rows={stats['appended']} "
          f"new_rows={stats['rows']} rescaled={stats['rescaled']} up_to_date={stats['current']} skipped={len(stats['skipped'])}")
    for s in stats['skipped']:
        print('HISTORY_SKIPPED', *s)


if __name__ == '__main__':
    main()
