import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import os
import gc
from do_feature_v2 import run

rawpath='../data/origin/'
temppath='../data/temp/'
one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                   'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                   'adCategoryId', 'productId', 'productType']
vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                  'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

def get_data():
    tf_train = pd.read_csv(rawpath + 'train.csv')
    tf_adfeat = pd.read_csv(rawpath + 'adfeature.csv')
    tf_userfeat = pd.read_csv(rawpath + 'userfeature.csv')
    tf_userfeat.drop(['interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                  'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3'], axis=1, inplace=True)
    tf_train = pd.merge(tf_train, tf_adfeat, on='aid', how='left')
    tf_train = pd.merge(tf_train, tf_userfeat, on='uid', how='inner')
    del tf_adfeat
    del tf_userfeat
    gc.collect()
    #neet to wash
    tf_train.fillna(0,inplace=True)
    tf_train['LBS'].fillna(1, inplace=True)
    tf_train['carrier'].fillna(1, inplace=True)
    tf_train.loc[tf_train['LBS'] == 0, 'LBS'] = 1
    tf_train.loc[tf_train['carrier'] == 0, 'carrier'] = 1
    tf_train.to_csv(temppath + 'merge_all.csv')

    for feature in one_hot_feature:
        try:
            tf_train[feature] = LabelEncoder().fit_transform(tf_train[feature].apply(int))
        except:
            tf_train[feature] = LabelEncoder().fit_transform(tf_train[feature])
    tf_train.to_csv(temppath + 'LE_all.csv')
    return tf_train

def do_feat1(tf_train):

    for feat_1 in ['uid','LBS','advertiserId']:#'creativeId',
        gc.collect()
        res = pd.DataFrame()
        temp = tf_train[[feat_1,'label']]
        count_par = temp.groupby([feat_1]).apply(lambda x: x['label'][(x['label'] == 1).values].count()).reset_index(
            name=feat_1 + '_par')
        count_all = temp.groupby([feat_1]).apply(lambda x: x['label'].count()).reset_index(
            name=feat_1 + '_all')
        count_all[feat_1 + '_par'] = count_par[feat_1 + '_par']
        count_all.fillna(value=0, inplace=True)
        res = res.append(count_all,ignore_index=True)
        print(feat_1,'over')
        res.to_csv(temppath + '%s.csv' %feat_1, index=False)

    for feat_1,feat_2 in [('creativeId','consumptionAbility'),('consumptionAbility','productType'),('education','productType'),
                          ('education','productId'),('uid','LBS'),('LBS','advertiserId'),('LBS','ct'),('LBS','creativeId'),
                          ('LBS','productId'),('os', 'LSB'),('advertiserId','ct'),('appIdInstall','productType'),
                          ('productType','appIdAction'),('uid','productId'),('uid','ct'),('ct','productType')]:
        gc.collect()
        res = pd.DataFrame()
        temp = tf_train[[feat_1, feat_2, 'label']]
        count_all = temp.groupby([feat_1, feat_2]).apply(
            lambda x: x['label'].count()).reset_index(name=feat_1 + '_' + feat_2 + '_all')
        count_par = temp.groupby([feat_1, feat_2]).apply(
            lambda x: x['label'][(x['label'] == 1).values].count()).reset_index(name=feat_1 + '_' + feat_2 + '_par')
        count_all[feat_1 + '_' + feat_2 + '_par'] = count_par[feat_1 + '_' + feat_2 + '_par']
        count_all.fillna(value=0, inplace=True)
        res = res.append(count_all, ignore_index=True)
        print(feat_1, feat_2, ' over')
        res.to_csv(temppath + '%s.csv' % (feat_1 + '_' + feat_2), index=False)

    for feat_1,feat_2,feat_3 in[('productId','ct','LBS'),('productId','house','gender'),('age','creativeId','productId')]:
        gc.collect()
        res=pd.DataFrame()
        temp = tf_train[[feat_1,feat_2,feat_3,'label']]
        count=temp.groupby([feat_1,feat_2,feat_3]).apply(lambda x: x['label'].count()).reset_index(name=feat_1+'_'+feat_2+'_'+feat_3+'_all')
        count1=temp.groupby([feat_1,feat_2,feat_3]).apply(lambda x: x['label'][(x['label'] == 1).values].sum()).reset_index(name=feat_1+'_'+feat_2+'_'+feat_3+'_par')
        count[feat_1+'_'+feat_2+'_'+feat_3+'_par']=count1[feat_1+'_'+feat_2+'_'+feat_3+'_par']
        count.fillna(value=0, inplace=True)
        res=res.append(count,ignore_index=True)
        print(feat_1,feat_2,feat_3,' over')
        res.to_csv(temppath+'%s.csv' % (feat_1+'_'+feat_2+'_'+feat_3), index=False)



#tf_train = get_data()
tf_train = pd.read_csv(temppath + 'LE_all.csv')
do_feat1(tf_train)
run()
