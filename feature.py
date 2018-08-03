import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import gc
import random
import numpy as np
import scipy.special as special
from tqdm import tqdm
from collections import *

cross_feat = ['LBS_marriageStatus','adCategoryId_LBS','adCategoryId_age','adCategoryId_ct','adCategoryId_productId','advertiserId_LBS','campaignId_productId','campaignId_adCategoryId','campaignId_age','campaignId_ct','campaignId_marriageStatus','consumptionAbility_ct','education_ct','gender_os','productId_LBS','productType_marriageStatus','productType_ct']

stas_feat = ['adCategoryId_age','adCategoryId_marriageStatus','age_carrier','age_consumptionAbility','age_gender','age_house','age_os','campaignId_LBS','campaignId_marriageStatus','consumptionAbility_gender','creativeSize_age','creativeSize_marriageStatus','education_gender','productId_marriageStatus','productType_age','productType_consumptionAbility']
#多值特征
def feat_mulval(data):
    for i in data.columns:
        i_types = data[i].dtypes
        if i_types != 'object':
            print('single', i)
            # train_and_test_and_adFeature[i] = train_and_test_and_adFeature[i].astype('category', copy=False)
        else:
            if str(i).startswith('marriageStatus'):
                tmp = data[i].apply(lambda x: len(str(x).split(' ')))
                i_length = min(5, tmp.max())
                for sub_i in range(i_length):
                    data[i + '_%d' % (sub_i)] = data[i].apply(
                        lambda x: str(x).split(' ')[sub_i] if sub_i < len(str(x).split(' ')) else -1
                    )
                    data[i + '_length'] = data[i].apply(
                        lambda x: len(list(set(str(x).split(' '))))
                    )
    return data
#过滤低值
def remove_lowcase(se,times):
    count = dict(se.value_counts())
    se = se.map(lambda x : -1 if count[x]<times else x)
    return se
#交叉组合特征
def feat_cross(data):
    for feat in cross_feat+stas_feat:
        afeat,ufeat = feat.split('_')
        if afeat+'_'+ufeat in data.columns or ufeat+'_'+afeat in data.columns:
            continue

        concat_feat = afeat + '_' + ufeat
        data[concat_feat] = data[afeat].astype('str') + '_' + data[ufeat].astype('str')
        data[concat_feat] = remove_lowcase(data[concat_feat], 100)  ####################change the num
        try:
            data[concat_feat] = LabelEncoder().fit_transform(data[concat_feat].apply(int))
        except:
            data[concat_feat] = LabelEncoder().fit_transform(data[concat_feat].astype(str))
        print(concat_feat + ' ok')
    return data

#转化率特征
def feat_sta(data):
    df_tr = data[['uid', 'aid', 'label']]
    for feat in stas_feat:
        concat_feat = feat
        if feat not in data.columns:
            afeat,ufeat = feat.split('_')
            concat_feat = afeat + '_' + ufeat

        df_test = df_tr[data.label == 0]
        df_train = df_tr[data.label != 0]

        from sklearn.model_selection import KFold
        df_stas_feat = None
        kf = KFold(n_splits=5, random_state=2018, shuffle=True)
        for train_index, val_index in kf.split(df_train):
            X_train = df_train.loc[train_index, :]
            X_val = df_train.loc[val_index, :]
            X_val = static_feat(X_train, X_val, concat_feat)
            df_stas_feat = pd.concat([df_stas_feat, X_val], axis=0)

        df_test = static_feat(df_train, df_test, concat_feat)
        df_stas_feat = pd.concat([df_stas_feat, df_test], axis=0)
        del df_stas_feat['label'], df_stas_feat[concat_feat],df_train,df_test
        gc.collect()
        data= pd.merge(data,df_stas_feat,on=['aid','uid'],how='left')
        print(concat_feat+'ok')
    return data
def static_feat(df,df_val, feature):

    df['label'] = df['label'].replace(-1,0)
    df = df.groupby(feature)['label'].agg(['sum','count']).reset_index()

    new_feat_name = feature + '_stas'
    ###########################
    bs = BayesianSmoothing(1, 1)
    bs.update(df['count'].values, df['sum'].values, 1000, 0.001)
    df.loc[:, new_feat_name] = (df['sum'] + bs.alpha) / (df['count'] + bs.alpha + bs.beta)
    ###############################
    #df.loc[:,new_feat_name] = 100 * (df['sum'] + 1) / (df['count'] + np.sum(df['sum']))

    # print(df.head())
    df_stas = df[[feature,new_feat_name]]

    df_val = pd.merge(df_val, df_stas, how='left', on=feature)
    ####
    df_val.fillna(value=bs.alpha / (bs.alpha + bs.beta), inplace=True)
    df_val[new_feat_name]=round(100*df_val[new_feat_name],8)
    del df_val[feature]
    return df_val
#贝叶斯平滑
class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)



#多值转化率特征
def nlp_feature_score(feature,data):
    predict = data[data['label'] == 0 ]
    train = data[data['label'] != 0 ]
    del data
    gc.collect()
    feature_count = {}
    feature_label_count = {}
    feature_list = train[feature].astype(str).values.tolist()
    label_list = train['label'].tolist()
    for i in range(train.shape[0]):
        for item in feature_list[i].split(" "):
            if item in feature_count.keys():
                feature_count[item] += 1
            else:
                feature_count[item] = 1
            if (item not in feature_label_count.keys()) and (label_list[i]==1):
                feature_label_count[item] = 1
            elif (item in feature_label_count.keys()) and (label_list[i]==1):
                feature_label_count[item] += 1
    dianji=[]
    zhuanhua = []
    for item in feature_label_count.keys():
        dianji.append(feature_count[item])
        zhuanhua.append(feature_label_count[item])
    c = pd.DataFrame({'dianji':dianji,'zhuanhua':zhuanhua})
    hyper = HyperParam(1, 1)
    hyper.update_from_data_by_FPI(c['dianji'], c['zhuanhua'], 1000, 0.00000001)
    train_feature_score = []
    for i in range(train.shape[0]):
        score = 1
        for item in feature_list[i].split(" "):
            if item not in feature_label_count.keys():
                score += 0
            else:
                score  = score *(1- (feature_label_count[item]+hyper.alpha)/(feature_count[item]+hyper.alpha+hyper.beta))
        train_feature_score.append(score)
    test_feature_score = []
    test_feature_list = predict[feature].values.tolist()
    for i in range(predict.shape[0]):
        score = 1
        for item in test_feature_list[i].split(" "):
            if item not in feature_label_count.keys():
                score += 0
            else:
                score  = score *(1- (feature_label_count[item]+hyper.alpha)/(feature_count[item]+hyper.alpha+hyper.beta))
        test_feature_score.append(score)
    train[feature + "_score"] = train_feature_score
    train[feature + "_score"][train[feature]=="-1"]=-1.0
    predict[feature + "_score"] = test_feature_score
    predict[feature + "_score"][predict[feature]=="-1"]=-1.0

    predict['label']=0
    data = pd.concat([train,predict])
    return data
    #data.to_csv(config.feat_folder + '/part/'+feature +'.csv',index=False)

class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        # 产生样例数据
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            # imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return pd.Series(I), pd.Series(C)

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        # 更新策略
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        # 迭代函数
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        sumfenzialpha = (special.digamma(success + alpha) - special.digamma(alpha)).sum()
        sumfenzibeta = (special.digamma(tries - success + beta) - special.digamma(beta)).sum()
        sumfenmu = (special.digamma(tries + alpha + beta) - special.digamma(alpha + beta)).sum()

        return alpha * (sumfenzialpha / sumfenmu), beta * (sumfenzibeta / sumfenmu)
#计数特征
def feat_size(data):
    aid_age_count = data.groupby(['aid', 'age']).size().reset_index().rename(columns={0: 'aid_age_count'})
    data = pd.merge(data, aid_age_count, 'left', on=['aid', 'age'])
    aid_gender_count = data.groupby(['aid', 'gender']).size().reset_index().rename(columns={0: 'aid_gender_count'})
    data = pd.merge(data, aid_gender_count, 'left', on=['aid', 'gender'])
    return data

#nunique特征
def feat_active(data):
    add = pd.DataFrame(data.groupby(["campaignId"]).aid.nunique()).reset_index()
    add.columns = ["campaignId", "campaignId_active_aid"]
    data = data.merge(add, on=["campaignId"], how="left")
    return data

from sklearn.preprocessing import StandardScaler
def get_nunique(data,pr_key,col,name):
    tmp = data.groupby([pr_key])[col].nunique().reset_index()
    #print(tmp.head())

    tmp.columns = [pr_key,name]
    sc = StandardScaler()
    data = pd.merge(data,tmp,on=[pr_key],how='left',copy=False)
    data[name] = sc.fit_transform(data[name].values.reshape(-1,1))
    del tmp
    return data

def pro(data):
    data = get_nunique(data,'uid','aid','uid_to_aid')
    # [515]	valid_0's auc: 0.666197
    data = get_nunique(data,'uid','campaignId','uid_to_campaignId')
    # [509]	valid_0's auc: 0.666677
    data = get_nunique(data,'uid','adCategoryId','uid_to_adCategoryId')
    # [405]	valid_0's auc: 0.667196
    data = get_nunique(data,'uid','productId','uid_to_productId')
    # [515]	valid_0's auc: 0.667237
    data = get_nunique(data,'uid','productType','uid_to_productType')
    # [784]	valid_0's auc: 0.66769

    # {515]	valid_0's auc: 0.670179
    data = get_nunique(data,'productId','aid','productId_to_aid')
    return data

#计数特征
def feat_count():
    for feat in config.vector:
    df_LE[feat+'_count_byuid'] = df_LE.groupby('uid')[feat].apply(lambda x: str(x).split(' ').count())
#####################################################
#重复值特征
def feat_duplicate():
    subset = []
    data['maybe'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 3

    features_trans = ['maybe']
    data = pd.get_dummies(data, columns=features_trans)
    data['maybe_0'] = data['maybe_0'].astype(np.int8)
    data['maybe_1'] = data['maybe_1'].astype(np.int8)
    data['maybe_2'] = data['maybe_2'].astype(np.int8)
    data['maybe_3'] = data['maybe_3'].astype(np.int8)
    # 重复次数是否大于2
    temp = data.groupby(subset)['label'].count().reset_index()
    temp.columns = ['creativeID', 'positionID', 'adID', 'appID', 'userID', 'large2']
    temp['large2'] = 1 * (temp['large2'] > 2)
    data = pd.merge(data, temp, how='left', on=subset)
    del temp
    gc.collect()
    return data


#均值特征
def feat_mean():
    res = data[['index']]

    grouped = data.groupby('appID')['age'].mean().reset_index()
    grouped.columns = ['appID', 'app_mean_age']
    data = data.merge(grouped, how='left', on='appID')
    grouped = data.groupby('positionID')['age'].mean().reset_index()
    grouped.columns = ['positionID', 'position_mean_age']
    data = data.merge(grouped, how='left', on='positionID')
    grouped = data.groupby('appCategory')['age'].mean().reset_index()
    grouped.columns = ['appCategory', 'appCategory_mean_age']
    data = data.merge(grouped, how='left', on='appCategory')

    X_train = data.loc[data['label'] != -1, :]
    X_test = data.loc[data['label'] == -1, :]
    X_test.loc[:, 'instanceID'] = res.values
    del data, grouped
    gc.collect()
    return X_train, X_test

#排序特征
def get_rank_feat(data, col, name):
    data[name] = data.groupby(col).cumcount() + 1
    return data

#
def gen_pos_neg_aid_fea():
    train_data = pd.read_csv('input/train.csv')
    test2_data = pd.read_csv('input/test2.csv')

    train_user = train_data.uid.unique()

    # user-aid dict
    uid_dict = defaultdict(list)
    for row in tqdm(train_data.itertuples(), total=len(train_data)):
        uid_dict[row[2]].append([row[1], row[3]])

    # user convert
    uid_convert = {}
    for uid in tqdm(train_user):
        pos_aid, neg_aid = [], []
        for data in uid_dict[uid]:
            if data[1] > 0:
                pos_aid.append(data[0])
            else:
                neg_aid.append(data[0])
        uid_convert[uid] = [pos_aid, neg_aid]

    test2_neg_pos_aid = {}
    for row in tqdm(test2_data.itertuples(), total=len(test2_data)):
        aid = row[1]
        uid = row[2]
        if uid_convert.get(uid, []) == []:
            test2_neg_pos_aid[row[0]] = ['', '', -1]
        else:
            pos_aid, neg_aid = uid_convert[uid][0].copy(), uid_convert[uid][1].copy()
            convert = len(pos_aid) / (len(pos_aid) + len(neg_aid)) if (len(pos_aid) + len(neg_aid)) > 0 else -1
            test2_neg_pos_aid[row[0]] = [' '.join(map(str, pos_aid)), ' '.join(map(str, neg_aid)), convert]
    df_test2 = pd.DataFrame.from_dict(data=test2_neg_pos_aid, orient='index')
    df_test2.columns = ['pos_aid', 'neg_aid', 'uid_convert']

    train_neg_pos_aid = {}
    for row in tqdm(train_data.itertuples(), total=len(train_data)):
        aid = row[1]
        uid = row[2]
        pos_aid, neg_aid = uid_convert[uid][0].copy(), uid_convert[uid][1].copy()
        if aid in pos_aid:
            pos_aid.remove(aid)
        if aid in neg_aid:
            neg_aid.remove(aid)
        convert = len(pos_aid) / (len(pos_aid) + len(neg_aid)) if (len(pos_aid) + len(neg_aid)) > 0 else -1
        train_neg_pos_aid[row[0]] = [' '.join(map(str, pos_aid)), ' '.join(map(str, neg_aid)), convert]

    df_train = pd.DataFrame.from_dict(data=train_neg_pos_aid, orient='index')
    df_train.columns = ['pos_aid', 'neg_aid', 'uid_convert']

    df_train.to_csv("dataset/train_neg_pos_aid.csv", index=False)
    df_test2.to_csv("dataset/test2_neg_pos_aid.csv", index=False)

