#!/bin/bash
# One-command daily after-market run (so a scheduled session needs a single approval).
# Usage: bash /home/claude/tse-model/code/daily.sh [YYYY-MM-DD]
set -u
REPO=/home/claude/tse-model; RUN=/home/claude/multi
DAY=${1:-$(TZ=Asia/Tehran date +%F)}
LOG=$REPO/state/run_logs/$DAY.log; mkdir -p $REPO/state/run_logs
exec > >(tee "$LOG") 2>&1
step(){ echo "== STEP $1"; }
fail(){ echo "!! FAIL $1"; }
step restore; cd $REPO && python3 code/bootstrap.py restore || { fail restore; exit 1; }
cd $RUN
# fill a missing session from state/daily if the archive skipped a day (see NOTES)
step close; timeout 1800 python3 live.py close | grep -E "^DATA|MODEL_ERROR|MARKET|DIVERG|SYNC|Traceback|Error" || fail close
step sweep
if ls live_state/tse-data/archive/$DAY.csv.* >/dev/null 2>&1; then
  cp live_state/decisions_10m.csv live_state/decisions_10m.bak
  python3 -c "import pandas as pd;d=pd.read_csv('live_state/decisions_10m.csv');d[d.day!='$DAY'].to_csv('live_state/decisions_10m.csv',index=False)"
  START=$(python3 -c "import pandas as pd;a=pd.read_csv('live_state/tse-data/archive/$DAY.csv.xz' if __import__('os').path.exists('live_state/tse-data/archive/$DAY.csv.xz') else 'live_state/tse-data/archive/$DAY.csv.gz',usecols=['زمان دریافت']);t=sorted(x for x in a['زمان دریافت'].astype(str) if x[:2] in ('09','10'))[0];h,m=int(t[:2]),int(t[3:5]);m=(m//10+1)*10;h+=m//60;m%=60;print(f'{h:02d}:{m:02d}')" 2>/dev/null || echo 09:20)
  python3 replay_hourly.py $DAY sweep $START 12:30 | grep -E "checkpoints|Error" || fail sweep
else fail "sweep (no minute archive for $DAY yet)"; fi
step eval; python3 eval_decisions.py > /dev/null || fail eval_decisions; python3 excel_eval.py > /dev/null || fail excel_eval
step macro; python3 macro_regime.py > /dev/null || fail macro_regime
step sheet; timeout 1500 python3 daily_sheet.py | grep -E "MARKET_TILT|xlsx|^خرید|^فروش|^خنثی" || fail daily_sheet
if [ "$(TZ=Asia/Tehran date +%u)" = "3" ]; then step retrain; python3 five_final.py > /dev/null && python3 calib_five.py > /dev/null || fail retrain; fi
step save; cd $REPO && python3 code/bootstrap.py save "after-market $DAY" | tail -1 || fail save
echo "== DONE $DAY  report=$RUN/live_state/reports/signals_$DAY.xlsx"
