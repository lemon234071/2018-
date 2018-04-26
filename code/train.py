import numpy as np
import pandas as pd
import scipy as sp
import lightgbm as lgb
import gc
import datetime
import random
import scipy.special as special
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
rawpath='../data/origin/'
temppath='../data/temp/'

t_start = datetime.datetime.now()
X_loc_train = pd.read_csv(temppath + '2_smooth.csv')
print('load train over...')
X_loc_test = pd.read_csv(temppath + '2_test_smooth.csv')
print('load test over...')


drop = ['label']
y_loc_train = X_loc_train.loc[:, 'label']
X_loc_train.drop(drop, axis=1, inplace=True)

# y_loc_val = X_loc_val.loc[:, 'label']
# X_loc_val.drop(drop, axis=1, inplace=True)

res = X_loc_test[['aid','uid']]
X_loc_test.drop(drop, axis=1, inplace=True)

gc.collect()
print('preprocess over...', X_loc_train.shape)

##########################################################比赛只用了lightGBM单模型
X_loc_train = X_loc_train.values
y_loc_train = y_loc_train.values
# X_loc_val=X_loc_val.values
# y_loc_val=y_loc_val.values
X_loc_test = X_loc_test.values

##########################################################交叉预测，实际上是stacking第一层做的操作
# 利用不同折数加参数，特征，样本（随机数种子）扰动，再加权平均得到最终成绩
model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=29, max_depth=-1, learning_rate=0.1, n_estimators=10000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=1, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, nthread=-1, silent=True)

skf = list(StratifiedKFold(y_loc_train, n_folds=10, shuffle=True, random_state=1024))
for i, (train, test) in enumerate(skf):
    print("Fold", i)
    model.fit(X_loc_train[train], y_loc_train[train], eval_metric='logloss',
              eval_set=[(X_loc_train[train], y_loc_train[train]), (X_loc_train[test], y_loc_train[test])],
              early_stopping_rounds=100)
    preds = model.predict_proba(X_loc_test, num_iteration=-1)[:, 1]
    print('mean:', preds.mean())
    res['prob_%s' % str(i)] = preds

model._Booster.save_model('../code/lgb.model')
# model = xgb.Booster(model_file='xgb.model')

# 平均或者加权的方式有很多种，台大三傻的比赛分享里有一个利用sigmoid反函数来平均的方法效果不错
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
print(now)
res.to_csv(rawpath + "%s.csv" % now, index=False)

t_end = datetime.datetime.now()
print('training time: %s' % ((t_end - t_start).seconds / 60))
