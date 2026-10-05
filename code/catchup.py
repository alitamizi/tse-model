import sys, os, numpy as np, pandas as pd
sys.argv=[sys.argv[0]]+sys.argv[1:]
day=sys.argv[1]
import live
a=pd.read_csv(f'live_state/tse-data/archive/{day}.csv.xz')
a=a.sort_values('زمان دریافت').groupby('insCode').last().reset_index()
st=pd.read_csv(f'live_state/tse-data/archive/{day}_static.csv.xz').drop(columns=['نماد'])
s=a.merge(st,on='insCode',how='left')
s['زمان آخرین معامله']=day+'T'+s['زمان آخرین معامله'].astype(str)+'+03:30'
v=lambda c: pd.to_numeric(s[c],errors='coerce')
avg=v('ارزش معاملات')/v('حجم معاملات')
s['قدرت خریدار حقیقی']=(v('حجم خرید حقیقی')/v('تعداد خرید حقیقی'))/(v('حجم فروش حقیقی')/v('تعداد فروش حقیقی'))
s['ورود پول حقیقی']=(v('حجم خرید حقیقی')-v('حجم فروش حقیقی'))*avg
s['سرانه خرید حقیقی']=v('حجم خرید حقیقی')*avg/v('تعداد خرید حقیقی')
s['سرانه فروش حقیقی']=v('حجم فروش حقیقی')*avg/v('تعداد فروش حقیقی')
s['ارزش بازار']=v('تعداد سهام')*v('قیمت پایانی')
s=live.prep_snap(s)
X,N=live.load_daily()
d,add,adj=live.append_day(s,X,N)
print(d,add,adj)
if add:
    os.replace('panel.pkl','live_state/panel_prev.pkl'); pd.to_pickle((X,N),'panel.pkl')
