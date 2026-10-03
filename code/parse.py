import openpyxl, pandas as pd, numpy as np, glob, os
from concurrent.futures import ProcessPoolExecutor
COLS=['gdate','jdate','open','high','low','last','adj','close','chg','pct','mcap','ntr','vol','value','float','pe','pb','ps','bp','nrp','net','pcb','pcs']
def num(x):
    try: return float(x)
    except: return np.nan
HMAP={'تاریخ میلادی':'gdate','تاریخ شمسی':'jdate','بازگشایی':'open','بالاترین':'high','پایین\u200cترین':'low','آخرین معامله':'last','قیمت تعدیلی':'adj','پایانی':'close',
 'میزان تغییر':'chg','درصد تغییر':'pct','ارزش بازار':'mcap','تعداد معاملات':'ntr','حجم':'vol','ارزش معاملات':'value','سهام شناور':'float','P/E-ttm':'pe','P/B':'pb','P/S':'ps',
 'قدرت خرید حقیقی':'bp','درصد خالص حقیقی':'nrp','خالص حقیقی':'net','سرانه خرید حقیقی':'pcb','سرانه فروش حقیقی':'pcs','NAV صدور':'nav_i','NAV ابطال':'nav_r','P/NAV':'pnav','NAV':'nav'}
def parse(fn):
    wb=openpyxl.load_workbook(fn,read_only=True); ws=wb.worksheets[0]
    rows=[];hdr=None;name=None
    for i,r in enumerate(ws.iter_rows(values_only=True)):
        if i==4: name=r[1]
        if hdr is None:
            if r[1]=='تاریخ میلادی': hdr=[HMAP.get(h,h) for h in r[1:] if h is not None]
            continue
        if r[1] is None: continue
        rows.append(r[1:1+len(hdr)])
    d=pd.DataFrame(rows,columns=hdr if hdr else COLS)
    for c in COLS:
        if c not in d: d[c]=np.nan
    d['gdate']=pd.to_datetime(d.gdate); d=d.sort_values('gdate').set_index('gdate')
    for c in d.columns:
        if c!='jdate': d[c]=d[c].map(num)
    sym=os.path.basename(fn)[:-5]
    if sym.startswith('price-history') or '-price-history' in sym: sym=str(name).split('-')[0].strip()
    return sym,name,d
if __name__=='__main__':
    fs=sorted(glob.glob('/home/claude/multi/raw/**/*.xlsx',recursive=True)+glob.glob('/home/claude/multi/raw2/**/*.xlsx',recursive=True)+glob.glob('/home/claude/multi/raw3/*.xlsx'))
    with ProcessPoolExecutor(8) as ex: res=list(ex.map(parse,fs))
    P={};N={}
    for s,n,d in res:
        if s in P and len(P[s]) and (len(d)==0 or d.index[-1]<P[s].index[-1] or (d.index[-1]==P[s].index[-1] and len(d)<=len(P[s]))): continue
        P[s]=d; N[s]=n
    pd.to_pickle((P,N),'/home/claude/multi/panel_raw.pkl')
    for s,n,d in res:
        if len(d)==0: print('EMPTY',s); continue
        print(s,'|',n,'|',d.index[0].date(),d.jdate.iloc[0],'->',d.index[-1].date(),len(d),'| flow from',d.bp.first_valid_index().date() if d.bp.notna().any() else None)
