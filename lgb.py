# -*- coding: utf8 -*-
import pandas as pd
import lightgbm as lgb
import gc
import datetime
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import sys
from scipy import sparse
from sklearn.model_selection import train_test_split
from hpsklearn import HyperoptEstimator
from hyperopt import tpe
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
from sklearn.cross_validation import cross_val_score

t_start = datetime.datetime.now()
print(t_start)
sys.stdout.flush()
mytype ={'uid':np.uint16,'LBS': np.uint16,'adCategoryId':np.uint16,'advertiserId':np.uint16,'age':np.uint16,'aid':np.uint16,'campaignId':np.uint16,
'carrier':np.uint16,'consumptionAbility':np.uint16,'creativeId':np.uint16,'creativeSize':np.uint16,'ct':object,'education':np.uint16,'gender':np.uint16,
         'house':np.uint16,'interest1':object,'interest2':object,'interest3':object,'interest4':object,'interest5':object,'kw1':object,'kw2':object,'kw3':object,
         'marriageStatus':object,'os':object ,'productId':np.uint16,'productType':np.uint16,'topic1':object,'topic2':object,'topic3':object,'label':np.uint16,
         'appIdAction':object,'appIdInstall':object,'appIdAction_score':np.uint16,'appIdInstall_score':np.uint16,'productId_to_aid':np.uint16,
         'aid_age_count':np.uint16,'aid_gender_count':np.uint16,'campaignId_active_aid':np.uint16,'LBS_marriageStatus':np.uint16,
         'adCategoryId_LBS':np.uint16,'adCategoryId_age':np.uint16,'adCategoryId_ct':np.uint16,'adCategoryId_productId':np.uint16,
         'advertiserId_LBS':np.uint16,'campaignId_productId':np.uint16,'campaignId_adCategoryId':np.uint16,'campaignId_age':np.uint16,
         'campaignId_ct':np.uint16,'campaignId_marriageStatus':np.uint16,'consumptionAbility_ct':np.uint16,'education_ct':np.uint16,'gender_os':np.uint16,
         'productId_LBS':np.uint16,'productType_marriageStatus':np.uint16,'productType_ct':np.uint16,
         'adCategoryId_age_stas':np.uint16,'adCategoryId_marriageStatus_stas':np.uint16,
         'age_carrier_stas':np.uint16,'age_consumptionAbility_stas':np.uint16,'age_gender_stas':np.uint16,
         'age_house_stas':np.uint16,'age_os_stas':np.uint16,'campaignId_LBS_stas':np.uint16,'campaignId_marriageStatus_stas':np.uint16,'consumptionAbility_gender_stas':np.uint16,'creativeSize_age_stas':np.uint16,'creativeSize_marriageStatus_stas':np.uint16,'education_gender_stas':np.uint16,'productId_marriageStatus_stas':np.uint16,'productType_age_stas':np.uint16,'productType_consumptionAbility_stas':np.uint16}

vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']

one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']


data = pd.read_csv('/data/spa/datas/mergedAll.csv', dtype=mytype, header=0, iterator=True)
data = data.get_chunk(10000000)
print(data.head())
# test = data.get_chunk(2000000)
# print(test.head())
data[data.label == -1] = 0
data[data.label == -2] = 0
data.fillna(0, inplace=True)
data_y = data.pop('label')
train, test, train_y, test_y = train_test_split(data, data_y, test_size=0.2, random_state=2018, stratify=data_y)
# X_loc_train = data[data.label!=0]
# X_loc_test = pd.read_csv('./datas/test2m.csv')
# X_loc_test['label']=0
# data = pd.concat([X_loc_train,X_loc_test])
# data.fillna(0,inplace=True)
# from sklearn.preprocessing import LabelEncoder
# for feature in data.columns:
#     if feature not in vector_feature:
#         try:
#             data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
#         except:
#             data[feature] = LabelEncoder().fit_transform(data[feature])
#
# data.to_csv('./datas/mergedAll.csv', index=False)
##########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
res = test[['uid','aid']]
train_y = train_y.values
# train_x=train[['creativeSize']]
# test_x=test[['creativeSize']]
#
# from sklearn.preprocessing import OneHotEncoder,LabelEncoder
# enc = OneHotEncoder()
# for feature in one_hot_feature:
#     enc.fit(data[feature].values.reshape(-1, 1))
#     train_a=enc.transform(train[feature].values.reshape(-1, 1))
#     test_a = enc.transform(test[feature].values.reshape(-1, 1))
#     train_x= sparse.hstack((train_x, train_a))
#     test_x = sparse.hstack((test_x, test_a))
# print('one-hot prepared !')
# from sklearn.feature_extraction.text import CountVectorizer
# cv=CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=5)
# for feature in vector_feature:
#     cv.fit(data[feature])
#     train_a = cv.transform(train[feature])
#     test_a = cv.transform(test[feature])
#     train_x = sparse.hstack((train_x, train_a))
#     test_x = sparse.hstack((test_x, test_a))
# print('cv prepared !')
# sparse.save_npz('./datas/train.npz', train_x, compressed=True)
# sparse.save_npz('./datas/test.npz', test_x, compressed=True)
print('load npz')
train_x = sparse.load_npz('./datas/train.npz')
test_x = sparse.load_npz('./datas/test.npz')
print('loaded npz')
cols = [col for col in train.columns if col not in one_hot_feature and col not in vector_feature]
train, test = train[cols],test[cols]
del train['creativeSize'],test['creativeSize']
train_x = sparse.hstack((train, train_x))
test_x = sparse.hstack((test, test_x))
###################################################################################

def LGB_test(train_x,train_y,test_x,test_y):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=300, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50,random_state=2018,n_jobs=-1
    )
    # clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],eval_metric='auc',early_stopping_rounds=100)
    # # print(clf.feature_importances_)
    # return clf,clf.best_score_[ 'valid_1']['auc']

    estim = HyperoptEstimator(classifier=clf,
                              algo=tpe.suggest,
                              max_evals=100,
                              trial_timeout=120)

    # Search the hyperparameter space based on the data
    estim.fit(train_x, train_y, random_state=2018)
    y_pred_te = estim.predict(test_x)

    print(roc_auc_score(test_y, y_pred_te))
    #print(estim.score(test_x, test_y))
    print(estim.best_model())

#LGB_test(train_x,train_y,test_x,test_y)

def LGB(argsDict):
    print("LGB test")
    #max_depth = argsDict["max_depth"] + 5
    n_estimators = argsDict['n_estimators'] * 1000 + 1000
    learning_rate = argsDict["learning_rate"] * 0.01 + 0.005
    subsample = argsDict["subsample"] * 0.1 + 0.7
    min_child_weight = argsDict["min_child_weight"]*10+5
    num_leaves = argsDict["num_leaves"]*10 + 21
    #print( "max_depth:" + str(max_depth))
    print("n_estimator:" + str(n_estimators))
    print("learning_rate:" + str(learning_rate))
    print("subsample:" + str(subsample))
    print("min_child_weight:" + str(min_child_weight))
    print("num_leaves:" + str(num_leaves))
    global train_x,train_y
    global test_x,test_y
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=num_leaves, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=n_estimators, objective='binary',
        subsample=subsample, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=learning_rate, min_child_weight=min_child_weight, random_state=2018, n_jobs=-1
    )
    #metric = cross_val_score(clf, train_x, train_y, cv=5, scoring="roc_auc").mean()
    clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],eval_metric='auc',early_stopping_rounds=100)
    # print(clf.feature_importances_)
    return -clf.best_score_[ 'valid_1']['auc']
    # y_pred_te = clf
    # metric = roc_auc_score(test_y, y_pred_te)
    #print(metric)
    #return -metric


def HY():
    space = {#"max_depth": hp.randint("max_depth", 15),
             "n_estimators": hp.randint("n_estimators", 10),  # [0,1,2,3,4,5] -> [50,]
             "learning_rate": hp.randint("learning_rate", 10),  # [0,1,2,3,4,5] -> 0.05,0.06
             "subsample": hp.randint("subsample", 4),  # [0,1,2,3] -> [0.7,0.8,0.9,1.0]
             "min_child_weight": hp.randint("min_child_weight", 10),
            "num_leaves": hp.randint("num_leaves", 2)
             }
    algo = partial(tpe.suggest, n_startup_jobs=1)
    best = fmin(LGB, space, algo=algo, max_evals=4)
    print(best)
    print(LGB(best))

HY()


t_end = datetime.datetime.now()
print('training time: %s' % ((t_end - t_start).seconds / 60))
sys.stdout.flush()
