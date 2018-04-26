import pandas as pd
import os
import gc
#v1
user_one_feature = ['carrier', 'LBS', 'age', 'consumptionAbility', 'education', 'gender', 'house']
adfeature = ['advertiserId','campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType']
def step1():
    xi = 1
    df_train_adfeature = pd.DataFrame()
    while os.path.exists('../data/merge_feature/train_%s.csv' %xi):
        df_train = pd.read_csv('../data/merge_feature/train_%s.csv' %xi)
        df_train_adfeature = df_train_adfeature.append(df_train, ignore_index=True)
        xi += 1
    df_ctr = pd.DataFrame()
    for feature in user_one_feature:#reget the one_feature
        df_one_userfeature = pd.read_csv('../data/merge_feature/' + feature + '.csv')##need product
        df_train_ad_oneuser = pd.merge(df_train_adfeature, df_one_userfeature, on=['uid','aid'], how='left')
        del df_one_userfeature
        gc.collect()
        df_train_ad_oneuser = df_train_ad_oneuser.fillna('0')
        #计算这条用户特征在各广告条目下的转化率
        slipt_feature = 待生成的分片特征名
        for one_slipt in slipt_feature:
            df_temp = cul_ctr(df_train_ad_oneuser, feature, one_slipt)
            df_train_ad_oneuser.pop(feature)#or drop
            df_ctr = df_ctr.append(df_temp,ignore_index=True)#bug:需要按横轴append每个用户特征,need test

def cul_ctr(df_train_ad_oneuser, col, userfeature):
    ctr_list =[]
    for feature in slipt_adfeature:
        ctr_labelok = df_train_ad_oneuser.loc[df_train_ad_oneuser['label']==1,['label', feature, col]]
        ctr_value = ctr_labelok.loc[ctr_labelok[col] == userfeature, 'label'].count()
        ctr_list.append(ctr_value)
    df_ctr = pd.DataFrame(ctr_list, columns=[userfeature])
    return df_ctr