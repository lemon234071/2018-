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

#zhada**平滑转化率
# def count_feature(train_data, test1_data, test2_data, labels, k, test_only= False):
#     nums = len(train_data)
#     interval = nums // k
#     split_points = []
#     for i in range(k):
#         split_points.append(i * interval)
#     split_points.append(nums)
#
#     s = set()
#     for d in train_data:
#         xs = d.split(' ')
#         for x in xs:
#             s.add(x)
#     b = nums // len(s)
#     a = b*1.0 / 20
#
#     train_res = []
#     if not test_only:
#         for i in range(k):
#             tmp = []
#             total_dict, pos_dict = gen_count_dict(train_data, labels, split_points[i],split_points[i+1])
#             for j in range(split_points[i],split_points[i+1]):
#                 xs = train_data[j].split(' ')
#                 t = []
#                 for x in xs:
#                     if not total_dict.has_key(x):
#                         t.append(0.05)
#                         continue
#                     t.append((a + pos_dict[x]) / (b + total_dict[x]))
#                 tmp.append(max(t))
#             train_res.extend(tmp)
#
#     total_dict, pos_dict = gen_count_dict(train_data, labels, 1, 0)
#     test1_res = []
#     for d in test1_data:
#         xs = d.split(' ')
#         t = []
#         for x in xs:
#             if not total_dict.has_key(x):
#                 t.append(0.05)
#                 continue
#             t.append((a + pos_dict[x]) / (b + total_dict[x]))
#         test1_res.append(max(t))
#
#     test2_res = []
#     for d in test2_data:
#         xs = d.split(' ')
#         t = []
#         for x in xs:
#             if not total_dict.has_key(x):
#                 t.append(0.05)
#                 continue
#             t.append((a + pos_dict[x]) / (b + total_dict[x]))
#         test2_res.append(max(t))
#
#     return train_res, test1_res, test2_res

#dnn feature################################################################################################################################
def mutil_ids(train_df, dev_df, test_df, word2index):
    features_mutil = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                      'topic2', 'topic3', 'appIdAction', 'appIdInstall', 'marriageStatus', 'ct', 'os']
    for s in features_mutil:
        cont = {}
        with open('ffm_data/train/' + str(s), 'w') as f:
            for lines in list(train_df[s].values):
                f.write(str(lines) + '\n')
                for line in lines.split():
                    if str(line) not in cont:
                        cont[str(line)] = 0
                    cont[str(line)] += 1

        with open('ffm_data/dev/' + str(s), 'w') as f:
            for line in list(dev_df[s].values):
                f.write(str(line) + '\n')

        with open('ffm_data/test/' + str(s), 'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line) + '\n')
        index = []
        for k in cont:
            if cont[k] >= threshold:
                index.append(k)
        word2index[s] = {}
        for idx, val in enumerate(index):
            word2index[s][val] = idx + 2
        print(s + ' done!')


def len_features(train_df, dev_df, test_df, word2index):
    len_features = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                    'topic2', 'topic3']
    for s in len_features:
        dev_df[s + '_len'] = dev_df[s].apply(lambda x: len(x.split()) if x != '-1' else 0)
        test_df[s + '_len'] = test_df[s].apply(lambda x: len(x.split()) if x != '-1' else 0)
        train_df[s + '_len'] = train_df[s].apply(lambda x: len(x.split()) if x != '-1' else 0)
        s = s + '_len'
        cont = {}
        with open('ffm_data/train/' + str(s), 'w') as f:
            for line in list(train_df[s].values):
                f.write(str(line) + '\n')
                if str(line) not in cont:
                    cont[str(line)] = 0
                cont[str(line)] += 1

        with open('ffm_data/dev/' + str(s), 'w') as f:
            for line in list(dev_df[s].values):
                f.write(str(line) + '\n')

        with open('ffm_data/test/' + str(s), 'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line) + '\n')
        index = []
        for k in cont:
            if cont[k] >= threshold:
                index.append(k)
        word2index[s] = {}
        for idx, val in enumerate(index):
            word2index[s][val] = idx + 2
        del train_df[s]
        del dev_df[s]
        del test_df[s]
        gc.collect()
        print(s + ' done!')


def count_features(train_df, dev_df, test_df, word2index):
    count_feature = ['uid']
    data = train_df.append(dev_df)
    data = data.append(test_df)
    for s in count_feature:
        g = dict(data.groupby(s).size())
        s_ = s
        s = s + '_count'
        cont = {}

        with open('ffm_data/train/' + str(s), 'w') as f:
            for line in list(train_df[s_].values):
                line = g[line]
                if str(line) not in cont:
                    cont[str(line)] = 0
                cont[str(line)] += 1
                f.write(str(line) + '\n')

        with open('ffm_data/dev/' + str(s), 'w') as f:
            for line in list(dev_df[s_].values):
                line = g[line]
                f.write(str(line) + '\n')

        with open('ffm_data/test/' + str(s), 'w') as f:
            for line in list(test_df[s_].values):
                line = g[line]
                f.write(str(line) + '\n')
        index = []
        for k in cont:
            if cont[k] >= threshold:
                index.append(k)
        word2index[s] = {}
        for idx, val in enumerate(index):
            word2index[s][val] = idx + 2
        print(s + ' done!')


def kfold_features(train_df, dev_df, test_df, word2index):
    features_mutil = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                      'topic2', 'topic3', 'appIdAction', 'appIdInstall', 'marriageStatus', 'ct', 'os']
    for f in features_mutil:
        del train_df[f]
        del dev_df[f]
        del test_df[f]
        gc.collect()
    count_feature = ['uid']
    feature = ['advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType']
    for f in feature:
        train_df[f + '_uid'] = train_df[f] + train_df['uid'] * 10000000
        dev_df[f + '_uid'] = dev_df[f] + dev_df['uid'] * 10000000
        test_df[f + '_uid'] = test_df[f] + test_df['uid'] * 10000000
        count_feature.append(f + '_uid')

    for s in count_feature:
        temp = s
        kfold_static(train_df, dev_df, test_df, s)
        s = temp + '_positive_num'
        cont = {}
        with open('ffm_data/train/' + str(s), 'w') as f:
            for line in list(train_df[s].values):
                if str(line) not in cont:
                    cont[str(line)] = 0
                cont[str(line)] += 1
                f.write(str(line) + '\n')

        with open('ffm_data/dev/' + str(s), 'w') as f:
            for line in list(dev_df[s].values):
                f.write(str(line) + '\n')

        with open('ffm_data/test/' + str(s), 'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line) + '\n')
        index = []
        for k in cont:
            if cont[k] >= threshold:
                index.append(k)
        word2index[s] = {}
        for idx, val in enumerate(index):
            word2index[s][val] = idx + 2
        pkl.dump(word2index, open('ffm_data/dic.pkl', 'wb'))
        print(s + ' done!')
        del train_df[s]
        del dev_df[s]
        del test_df[s]

        s = temp + '_negative_num'
        cont = {}
        with open('ffm_data/train/' + str(s), 'w') as f:
            for line in list(train_df[s].values):
                if str(line) not in cont:
                    cont[str(line)] = 0
                cont[str(line)] += 1
                f.write(str(line) + '\n')

        with open('ffm_data/dev/' + str(s), 'w') as f:
            for line in list(dev_df[s].values):
                f.write(str(line) + '\n')

        with open('ffm_data/test/' + str(s), 'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line) + '\n')
        index = []
        for k in cont:
            if cont[k] >= threshold:
                index.append(k)
        word2index[s] = {}
        for idx, val in enumerate(index):
            word2index[s][val] = idx + 2
        pkl.dump(word2index, open('ffm_data/dic.pkl', 'wb'))
        del train_df[s]
        del dev_df[s]
        del test_df[s]
        gc.collect()
        print(s + ' done!')
    for f in feature:
        del train_df[f + '_uid']
        del test_df[f + '_uid']
        del dev_df[f + '_uid']
    gc.collect()


def kfold_static(train_df, dev_df, test_df, f):
    print("K-fold static:", f)
    # K-fold positive and negative num
    index = set(range(train_df.shape[0]))
    K_fold = []
    for i in range(5):
        if i == 4:
            tmp = index
        else:
            tmp = random.sample(index, int(0.2 * train_df.shape[0]))
        index = index - set(tmp)
        print("Number:", len(tmp))
        K_fold.append(tmp)
    positive = [-1 for i in range(train_df.shape[0])]
    negative = [-1 for i in range(train_df.shape[0])]
    for i in range(5):
        print('fold', i)
        pivot_index = K_fold[i]
        sample_idnex = []
        for j in range(5):
            if j != i:
                sample_idnex += K_fold[j]
        dic = {}
        for item in train_df.iloc[sample_idnex][[f, 'label']].values:
            if item[0] not in dic:
                dic[item[0]] = [0, 0]
            dic[item[0]][item[1]] += 1
        uid = train_df[f].values
        for k in pivot_index:
            if uid[k] in dic:
                positive[k] = dic[uid[k]][1]
                negative[k] = dic[uid[k]][0]
    train_df[f + '_positive_num'] = positive
    train_df[f + '_negative_num'] = negative

    # for dev and test
    dic = {}
    for item in train_df[[f, 'label']].values:
        if item[0] not in dic:
            dic[item[0]] = [0, 0]
        dic[item[0]][item[1]] += 1
    positive = []
    negative = []
    for uid in dev_df[f].values:
        if uid in dic:
            positive.append(dic[uid][1])
            negative.append(dic[uid][0])
        else:
            positive.append(-1)
            negative.append(-1)
    dev_df[f + '_positive_num'] = positive
    dev_df[f + '_negative_num'] = negative
    print('dev', 'done')

    positive = []
    negative = []
    for uid in test_df[f].values:
        if uid in dic:
            positive.append(dic[uid][1])
            negative.append(dic[uid][0])
        else:
            positive.append(-1)
            negative.append(-1)
    test_df[f + '_positive_num'] = positive
    test_df[f + '_negative_num'] = negative
    print('test', 'done')
    print('avg of positive num', np.mean(train_df[f + '_positive_num']), np.mean(dev_df[f + '_positive_num']),
          np.mean(test_df[f + '_positive_num']))
    print('avg of negative num', np.mean(train_df[f + '_negative_num']), np.mean(dev_df[f + '_negative_num']),
          np.mean(test_df[f + '_negative_num']))



def uid_seq_feature(train_data, test1_data, test2_data, label):
    count_dict = {}#存 该id： 追加这次出现的label
    seq_dict = {}#存序列字典 ：该种序列出现次数
    seq_emb_dict = {}#存该序列的key
    train_seq = []#存每个学列的index
    ind = 0
    for i, d in enumerate(train_data):
        if not count_dict.__contains__(d):
            count_dict[d] = []
        seq_key = ' '.join(count_dict[d][-4:])
        if not seq_dict.__contains__(seq_key):
            seq_dict[seq_key] = 0
            seq_emb_dict[seq_key] = ind
            ind += 1
        seq_dict[seq_key] += 1
        train_seq.append(seq_emb_dict[seq_key])
        count_dict[d].append(label[i])
    test1_seq = []
    for d in test1_data:
        if not count_dict.__contains__(d):
            seq_key = ''
        else:
            seq_key = ' '.join(count_dict[d][-4:])
        if seq_emb_dict.__contains__(seq_key):
            key = seq_emb_dict[seq_key]
        else:
            key = 0
        test1_seq.append(key)
    test2_seq = []
    for d in test2_data:
        if not count_dict.__contains__(d):
            seq_key = ''
        else:
            seq_key = ' '.join(count_dict[d][-4:])
        if seq_emb_dict.__contains__(seq_key):
            key = seq_emb_dict[seq_key]
        else:
            key = 0
        test2_seq.append(key)

def gen_uid_aid_fea():
    '''
    载入数据，　提取aid, uid的全局统计特征
    '''
    train_data = pd.read_csv('input/train.csv')
    test1_data = pd.read_csv('input/test1.csv')
    test2_data = pd.read_csv('input/test2.csv')

    ad_Feature = pd.read_csv('input/adFeature.csv')

    train_len = len(train_data)  # 45539700
    test1_len = len(test1_data)
    test2_len = len(test2_data)  # 11727304

    ad_Feature = pd.merge(ad_Feature, ad_Feature.groupby(['campaignId']).aid.nunique().reset_index(
    ).rename(columns={'aid': 'campaignId_aid_nunique'}), how='left', on='campaignId')

    df = pd.concat([train_data, test1_data, test2_data], axis=0)
    df = pd.merge(df, df.groupby(['uid'])['aid'].nunique().reset_index().rename(
        columns={'aid': 'uid_aid_nunique'}), how='left', on='uid')

    df = pd.merge(df, df.groupby(['aid'])['uid'].nunique().reset_index().rename(
        columns={'uid': 'aid_uid_nunique'}), how='left', on='aid')

    df['uid_count'] = df.groupby('uid')['aid'].transform('count')
    df = pd.merge(df, ad_Feature[['aid', 'campaignId_aid_nunique']], how='left', on='aid')

    fea_columns = ['campaignId_aid_nunique', 'uid_aid_nunique', 'aid_uid_nunique', 'uid_count', ]

    df[fea_columns].iloc[:train_len].to_csv('dataset/train_uid_aid.csv', index=False)
    df[fea_columns].iloc[train_len: train_len+test1_len].to_csv('dataset/test1_uid_aid.csv', index=False)
    df[fea_columns].iloc[-test2_len:].to_csv('dataset/test2_uid_aid.csv', index=False)


def digitize():
    uid_aid_train = pd.read_csv('dataset/train_uid_aid.csv')
    uid_aid_test1 = pd.read_csv('dataset/test1_uid_aid.csv')
    uid_aid_test2 = pd.read_csv('dataset/test2_uid_aid.csv')
    uid_aid_df = pd.concat([uid_aid_train, uid_aid_test1, uid_aid_test2], axis=0)
    for col in range(3):
        bins = []
        for percent in [0, 20, 35, 50, 65, 85, 100]:
            bins.append(np.percentile(uid_aid_df.iloc[:, col], percent))
        uid_aid_train.iloc[:, col] = np.digitize(uid_aid_train.iloc[:, col], bins, right=True)
        uid_aid_test1.iloc[:, col] = np.digitize(uid_aid_test1.iloc[:, col], bins, right=True)
        uid_aid_test2.iloc[:, col] = np.digitize(uid_aid_test2.iloc[:, col], bins, right=True)

    count_bins = [1, 2, 4, 6, 8, 10, 16, 27, 50]
    uid_aid_train.iloc[:, 3] = np.digitize(uid_aid_train.iloc[:, 3], count_bins, right=True)
    uid_aid_test1.iloc[:, 3] = np.digitize(uid_aid_test1.iloc[:, 3], count_bins, right=True)
    uid_aid_test2.iloc[:, 3] = np.digitize(uid_aid_test2.iloc[:, 3], count_bins, right=True)

    uid_convert_train = pd.read_csv("dataset/train_neg_pos_aid.csv", usecols=['uid_convert'])
    uid_convert_test2 = pd.read_csv("dataset/test2_neg_pos_aid.csv", usecols=['uid_convert'])

    convert_bins = [-1, 0, 0.1, 0.3, 0.5, 0.7, 1]
    uid_convert_train.iloc[:, 0] = np.digitize(uid_convert_train.iloc[:, 0], convert_bins, right=True)
    uid_convert_test2.iloc[:, 0] = np.digitize(uid_convert_test2.iloc[:, 0], convert_bins, right=True)

    uid_aid_train = pd.concat([uid_aid_train, uid_convert_train], axis=1)
    uid_aid_test2 = pd.concat([uid_aid_test2, uid_convert_test2], axis=1)

    uid_aid_train.to_csv('dataset/train_uid_aid_bin.csv', index=False)
    uid_aid_test2.to_csv('dataset/test2_uid_aid_bin.csv', index=False)


#################
def feature_count(full, features=[]):
    new_feature = 'new_count'
    for i in features:
        get_series(i)
        new_feature += '_' + i
    log(new_feature)
    try:
        del full[new_feature]
    except:
        pass
    temp = full.groupby(features).size().reset_index().rename(columns={0: new_feature})
    full = full.merge(temp, 'left', on=features)
    # save(full, new_feature)
    return full

def Information_entropy():
    for i in ['aid']:
        for j in ['age', 'gender', 'education', 'consumptionAbility', 'LBS', 'carrier', 'house',
                  'marriageStatus', 'ct', 'os']:
            t = time.time()
            full = feature_count(full, [i, j])
            full['new_inf_' + i + '_' + j] = np.log1p(
                full['new_count_' + j] * full['new_count_' + i] / full['new_count_' + i + '_' + j] / len_full)

            min_v = full['new_inf_' + i + '_' + j].min()
            full['new_inf_' + i + '_' + j] = full['new_inf_' + i + '_' + j].apply(
                lambda x: int(float('%.1f' % min(x - min_v, 1.5)) * 10))
            print(full['new_inf_' + i + '_' + j].value_counts())
            save(full, 'new_inf_' + i + '_' + j, 'cate', max=15)

    # 多值特征的信息熵
    def get_inf(cond, keyword):
        get_series(cond)
        get_series(keyword)

        # 背景字典 每个ID出现的次数
        back_dict = {}
        # 不同条件下 每个id 出现的次数condi_dict
        condi_dict = {}
        # 不同条件下aid 每个id 出现的次数

        # 预先生成字典。省的判断慢
        for i in full[cond].unique():
            condi_dict[i] = {}

        for i, row in full[[cond, keyword]].iterrows():
            word_list = row[keyword].split()
            for word in word_list:
                # 对背景字典加1
                try:
                    back_dict[word] = back_dict[word] + 1
                except:
                    # 没有该词则设为0
                    back_dict[word] = 1
                try:
                    # 该条件下的该词的出现次数加1
                    condi_dict[row[cond]][word] = condi_dict[row[cond]][word] + 1
                except:
                    condi_dict[row[cond]][word] = 1

        # 先获取平均熵
        max_inf_list = []
        mean_inf_list = []
        condi_count = full.groupby(cond)[cond].count().to_dict()
        for i, row in full[[cond, keyword]].iterrows():
            word_list = row[keyword].split()

            count = len(word_list)
            prob = 1
            prob_list = []
            for word in word_list:
                temp_prob = condi_count[row[cond]] * back_dict[word] / condi_dict[row[cond]][word] / len_full
                prob = prob * temp_prob
                prob_list.append(temp_prob)
            mean_inf_list.append(np.log1p(prob) / count)
            max_inf_list.append(
                np.log1p(np.min(prob)))
        return max_inf_list, mean_inf_list

    # 计算maxpool 和meanpool
    for i in ['aid']:
        for j in ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                  'topic2', 'topic3', 'appIdAction', 'appIdInstall']:
            log('new_inf_' + i + '_' + j)
            full['new_inf_' + i + '_' + j + '_max'], full['new_inf_' + i + '_' + j + '_mean'] = get_inf(i, j)

            min_v = full['new_inf_' + i + '_' + j + '_max'].min()
            full['new_inf_' + i + '_' + j + '_max'] = full['new_inf_' + i + '_' + j + '_max'].apply(
                lambda x: int(float('%.1f' % min(x - min_v, 1.5)) * 10))
            save(full, 'new_inf_' + i + '_' + j + '_max', 'cate', 16)

            min_v = full['new_inf_' + i + '_' + j + '_mean'].min()
            full['new_inf_' + i + '_' + j + '_mean'] = full['new_inf_' + i + '_' + j + '_mean'].apply(
                lambda x: int(float('%.1f' % min(x - min_v, 1.5)) * 10))
            save(full, 'new_inf_' + i + '_' + j + '_mean', 'cate', 16)