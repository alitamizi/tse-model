"""Run the daily after-market pipeline from a fresh session.
  python3 code/bootstrap.py restore   -> rebuild /home/claude/multi (code, models, state, daily panel) from this repo + tse-data archive
  python3 code/bootstrap.py save      -> copy state, models and reports back into this repo and push
The daily panel is stored as a frozen base (state/panel_base) plus every later session rebuilt from the tse-data minute archive."""
import os, sys, glob, shutil, subprocess, importlib
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = '/home/claude/multi'; ST = os.path.join(RUN, 'live_state')
STATE_FILES = ['signals.csv', 'verdicts.csv', 'positions.csv', 'evaluation.csv', 'symbol_methods.csv', 'symbol_flow_hist.csv', 'tech_state.csv',
               'divergence.csv', 'decisions_10m.csv', 'decision_eval.csv', 'metrics_daily.csv', 'five_scores.csv', 'excel_eval.csv',
               'macro_regime.json', 'market_tilt.json', 'seen.json', 'prev_alerts.json', 'news.csv', 'news_impact.csv']
def sh(*a, **k): return subprocess.run(list(a), check=True, **k)
def restore():
    for pkg in ('lightgbm', 'statsmodels', 'openpyxl', 'sklearn'):
        try: importlib.import_module(pkg)
        except ImportError: sh(sys.executable, '-m', 'pip', 'install', '-q', '--break-system-packages', 'scikit-learn' if pkg == 'sklearn' else pkg)
    os.makedirs(ST, exist_ok=True); os.makedirs(os.path.join(RUN, 'macro'), exist_ok=True); os.makedirs(os.path.join(ST, 'reports'), exist_ok=True)
    for f in glob.glob(os.path.join(REPO, 'code', '*.py')):
        if not f.endswith('bootstrap.py'): shutil.copy(f, RUN)
    for f in ('five_models.pkl', 'five_calib.pkl', 'bt.pkl'): shutil.copy(os.path.join(REPO, 'models', f), RUN)
    shutil.copy(os.path.join(REPO, 'state', 'macro_hist.pkl'), os.path.join(RUN, 'macro', 'macro_hist.pkl'))
    for f in STATE_FILES + [os.path.basename(p) for p in glob.glob(os.path.join(REPO, 'state', 'queue_override_*.csv'))]:
        p = os.path.join(REPO, 'state', f)
        if os.path.exists(p): shutil.copy(p, ST)
    for p in glob.glob(os.path.join(REPO, 'reports', 'signals_*.xlsx')): shutil.copy(p, os.path.join(ST, 'reports'))
    # tse-data (shallow)
    td = os.path.join(ST, 'tse-data')
    if not os.path.isdir(os.path.join(td, '.git')): sh('git', 'clone', '-q', '--depth', '1', 'https://github.com/alitamizi/tse-data.git', td, timeout=600)
    else: sh('git', '-C', td, 'fetch', '-q', '--depth', '1', 'origin', 'main'); sh('git', '-C', td, 'reset', '-q', '--hard', 'origin/main')
    # daily panel = base + sessions after base rebuilt from the minute archive (same logic as catchup.py)
    import pandas as pd
    if not os.path.exists(os.path.join(RUN, 'panel.pkl')):
        X = {}
        for p in sorted(glob.glob(os.path.join(REPO, 'state', 'panel_base', 'part*.pkl.xz'))): X.update(pd.read_pickle(p))
        N = pd.read_pickle(os.path.join(REPO, 'state', 'panel_base', 'N.pkl.xz'))
        pd.to_pickle((X, N), os.path.join(RUN, 'panel.pkl'))
    base = open(os.path.join(REPO, 'state', 'panel_base', 'base_date.txt')).read().strip()
    days = sorted({os.path.basename(p)[:10] for p in glob.glob(os.path.join(td, 'archive', '20??-??-??.csv.xz'))})
    for d in [d for d in days if d > base]:
        r = subprocess.run([sys.executable, 'catchup.py', d], cwd=RUN, capture_output=True, text=True)
        print('catchup', d, (r.stdout.strip().splitlines() or [''])[-1], r.stderr.strip()[-200:] if r.returncode else '')
    print('RESTORED', RUN)
def save():
    os.makedirs(os.path.join(REPO, 'state'), exist_ok=True); os.makedirs(os.path.join(REPO, 'reports'), exist_ok=True)
    for f in STATE_FILES + [os.path.basename(p) for p in glob.glob(os.path.join(ST, 'queue_override_*.csv'))]:
        p = os.path.join(ST, f)
        if os.path.exists(p): shutil.copy(p, os.path.join(REPO, 'state'))
    shutil.copy(os.path.join(RUN, 'macro', 'macro_hist.pkl'), os.path.join(REPO, 'state', 'macro_hist.pkl'))
    for f in ('five_models.pkl', 'five_calib.pkl'): shutil.copy(os.path.join(RUN, f), os.path.join(REPO, 'models'))
    for p in glob.glob(os.path.join(ST, 'reports', 'signals_*.xlsx')): shutil.copy(p, os.path.join(REPO, 'reports'))
    g = ['git', '-C', REPO]
    subprocess.run(g + ['add', '-A'], check=True)
    msg = sys.argv[2] if len(sys.argv) > 2 else 'daily after-market state'
    r = subprocess.run(g + ['commit', '-q', '-m', msg + '\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-200:])
    subprocess.run(g + ['pull', '-q', '--rebase', 'origin', 'main'], check=False)
    print('PUSH', subprocess.run(g + ['push', '-q', 'origin', 'HEAD:main'], capture_output=True, text=True).returncode)
if __name__ == '__main__':
    {'restore': restore, 'save': save}[sys.argv[1]]()
