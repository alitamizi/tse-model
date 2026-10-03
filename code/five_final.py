import pandas as pd,numpy as np,lightgbm as lgb,warnings,pickle;warnings.filterwarnings('ignore')
exec(open('five_feats.py').read())
P=dict(objective='binary',learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.7,bagging_fraction=0.5,bagging_freq=1,lambda_l2=5,verbose=-1,seed=5,num_threads=2)
M={}
for side in ['yb','ys']:
    tr=F[(F.jy>=1400)&F[side].notna()]
    M[side]=lgb.train(P,lgb.Dataset(tr[FEAT],tr[side]),num_boost_round=250)
    print(side,len(tr),flush=True)
pickle.dump(dict(models={k:v.model_to_string() for k,v in M.items()},FEAT=FEAT),open('five_models.pkl','wb'))
print('saved')
