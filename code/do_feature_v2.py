import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import os
import gc
from sklearn.preprocessing import LabelEncoder
import random
import scipy.special as special

rawpath='../data/origin/'
temppath='../data/temp/'
one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                   'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                   'adCategoryId', 'productId', 'productType']
vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                  'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

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
            imp = imp_upperbound
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


def get_data():
    tf_test = pd.read_csv(rawpath + 'test1.csv')
    tf_adfeat = pd.read_csv(rawpath + 'adfeature.csv')
    tf_userfeat = pd.read_csv(rawpath + 'userfeature.csv')
    tf_userfeat.drop(['interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                  'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3'], axis=1, inplace=True)
    tf_test = pd.merge(tf_test, tf_adfeat, on='aid', how='left')
    tf_test = pd.merge(tf_test, tf_userfeat, on='uid', how='inner')
    del tf_adfeat
    del tf_userfeat
    gc.collect()
    #neet to wash
    tf_test.fillna(0,inplace=True)
    tf_test['LBS'].fillna(1, inplace=True)
    tf_test['carrier'].fillna(1, inplace=True)
    tf_test.loc[tf_test['LBS'] == 0, 'LBS'] = 1
    tf_test.loc[tf_test['carrier'] == 0, 'carrier'] = 1
    tf_test.to_csv(rawpath + 'test_merge_all.csv')

    for feature in one_hot_feature:
        try:
            tf_test[feature] = LabelEncoder().fit_transform(tf_test[feature].apply(int))
        except:
            tf_test[feature] = LabelEncoder().fit_transform(tf_test[feature])
    tf_test.to_csv(rawpath + 'test_LE_all.csv')
    tf_train = pd.read_csv(rawpath + 'LE_all.csv')
    return tf_train, tf_test

def bs_smooth(X_loc_train, X_loc_test):

    for feat_1 in ['creativeId','uid','LBS','advertiserId']:
        temp = pd.read_csv(temppath+'%s.csv' %feat_1)
        bs = BayesianSmoothing(1, 1)
        bs.update(temp[feat_1 + '_all'].values, temp[feat_1 + '_par'].values, 1000, 0.001)
        temp[feat_1 + '_smooth'] = (temp[feat_1 + '_par'] + bs.alpha) / (temp[feat_1 + '_all'] + bs.alpha + bs.beta)
        if feat_1 in ['creativeId']:
            temp[feat_1 + '_rate'] = temp[feat_1 + '_par'] / temp[feat_1 + '_all']
        temp.drop([feat_1 + '_par',feat_1 + '_all'],axis=1,inplace=True)
        X_loc_train = pd.merge(X_loc_train, temp, how='left', on=[feat_1])
        X_loc_test = pd.merge(X_loc_test, temp, how='left', on=[feat_1])
        del temp
        gc.collect()
        print(feat_1 + ' over...')
        X_loc_train.fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)
        X_loc_test.fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)
    #类别少，不用平滑
    for feat_1 in ['sitesetID']:
        temp = pd.read_csv(temppath+'%s.csv' %feat_1)
        temp[feat_1 + '_rate'] = temp[feat_1 + '_par'] / temp[feat_1 + '_all']
        X_loc_train = pd.merge(X_loc_train, temp, how='left', on=[feat_1])
        X_loc_test = pd.merge(X_loc_test, temp, how='left', on=[feat_1])
        del temp
        gc.collect()
        print(feat_1 + ' over...')
        X_loc_train.fillna(value=0, inplace=True)
        X_loc_test.fillna(value=0, inplace=True)

    #三特征组合从周冠军分享的下载行为和网络条件限制，以及用户属性对app需求挖掘出
    for feat_1,feat_2,feat_3 in[('productId','ct','LBS'),('productId','house','gender'),('age','creativeId','productId')]:
        temp = pd.read_csv(temppath+'%s.csv' % (feat_1+'_'+feat_2+'_'+feat_3))
        bs = BayesianSmoothing(1, 1)
        bs.update(temp[feat_1+'_'+feat_2+'_'+feat_3 + '_all'].values, temp[feat_1+'_'+feat_2+'_'+feat_3 + '_par'].values, 1000, 0.001)
        temp[feat_1+'_'+feat_2+'_'+feat_3 + '_smooth'] = (temp[feat_1+'_'+feat_2+'_'+feat_3 + '_par'] + bs.alpha) / (temp[feat_1+'_'+feat_2+'_'+feat_3 + '_all'] + bs.alpha + bs.beta)
        temp.drop([feat_1+'_'+feat_2+'_'+feat_3+ '_par',feat_1+'_'+feat_2+'_'+feat_3 + '_all'],axis=1,inplace=True)
        X_loc_train = pd.merge(X_loc_train, temp, how='left', on=[feat_1,feat_2,feat_3])
        X_loc_test = pd.merge(X_loc_test, temp, how='left', on=[feat_1,feat_2,feat_3])
        del temp
        gc.collect()
        print(feat_1 + '_' + feat_2+'_'+feat_3+ ' over...')
        X_loc_train.fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)
        X_loc_test.fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)

    #userID和positionID的点击次数重要性排名靠前，所有统计特征只加了这一个点击次数
    for feat_1,feat_2 in[('creativeId','consumptionAbility'),('consumptionAbility','productType'),('education','productType'),
                              ('education','productId'),('uid','LBS'),('LBS','advertiserId'),('LBS','ct'),('LBS','creativeId'),
                              ('LBS','productId'),('os', 'LSB'),('advertiserId','ct'),('appIdInstall','productType'),
                              ('productType','appIdAction'),('uid','productId'),('uid','ct'),('ct','productType')]:
        temp = pd.read_csv(temppath+'%s.csv' % (feat_1+'_'+feat_2))
        bs = BayesianSmoothing(1, 1)
        bs.update(temp[feat_1+'_'+feat_2 + '_all'].values, temp[feat_1+'_'+feat_2 + '_par'].values, 1000, 0.001)
        temp[feat_1+'_'+feat_2 + '_smooth'] = (temp[feat_1+'_'+feat_2 + '_par'] + bs.alpha) / (temp[feat_1+'_'+feat_2 + '_all'] + bs.alpha + bs.beta)
        if (feat_1,feat_2) in [('userID','positionID')]:
            temp.drop([feat_1 + '_' + feat_2 + '_par'], axis=1, inplace=True)
        else:
            temp.drop([feat_1+'_'+feat_2 + '_par',feat_1+'_'+feat_2 + '_all'],axis=1,inplace=True)
        X_loc_train = pd.merge(X_loc_train, temp, how='left', on=[feat_1,feat_2])
        X_loc_test = pd.merge(X_loc_test, temp, how='left', on=[feat_1,feat_2])
        del temp
        gc.collect()
        print(feat_1 + '_' + feat_2 + ' over...')
        X_loc_train.fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)
        X_loc_test.fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)


    ##########################################################丢掉重要性低，缺失值多的原始特征
    drop = ['LBS', 'house', 'os', 'uid','ct','advertiserId', 'ct', 'marriageStatus', 'positionType','creativeId',
            'adCategoryId', 'productId', 'productType','gender', 'education', 'campaignId', 'maybe_0',
            'carrier','appIdAction', 'appIdInstall']
    X_loc_train.drop(drop, axis=1, inplace=True)
    X_loc_train.fillna(value=0, inplace=True)
    X_loc_test.drop(drop, axis=1, inplace=True)
    X_loc_test.fillna(value=0, inplace=True)
    print('over')
    print(X_loc_train.shape)
    print(X_loc_train.columns)
    X_loc_train.to_csv(temppath+'statistic2_smooth.csv',index=False)
    X_loc_test.to_csv(temppath+'statistic2_smooth2_test_smooth.csv',index=False)


import collections
def select_topk(data):
    """
    选择频率最高的 k 个word 此处为前20%
    :param data:
    :return:
    """
    word_list = []
    for words in data:
        word_list += words
    result = collections.Counter(word_list)
    size = len(result)
    result = result.most_common(int(size * 0.2))

    word_dict = {}
    for re in result:
        word_dict[re[0]] = 1
    print('word_vec: ', size, len(result))
    return word_dict

from gensim.models.word2vec import Word2Vec
def do_feat_W2V(data):
    df_W2V = data[['aid','uid','label']]
    for feature in vector_feature:
        print("this is feature:", feature)

        data[feature] = data[feature].apply(lambda x: str(x).split(' '))
        word_dict = select_topk(data[feature])#need to make sure!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!here is right
        data[feature] = data[feature].apply(lambda x: ' '.join(
            [word for word in x if word_dict.__contains__(word)]))

        model = Word2Vec(data[feature], size=10, min_count=1, iter=5, window=2)
        data_vec = []
        for row in data[feature]:
            data_vec.append(base_word2vec(row, model, size=10))
        column_names = []
        for i in range(10):
            column_names.append(feature + str(i))
        data_vec = pd.DataFrame(data_vec, columns=column_names)
        df_W2V = pd.concat([df_W2V, data_vec], axis=1)#need to make sure!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!here is right
        del data[feature]
    return df_W2V

def do_feat2( ):
    tf_test = pd.read_csv(temppath+'statistic2_smooth2_test_smooth.csv')
    tf_train = pd.read_csv(temppath + 'statistic2_smooth2_smooth.csv')
    tf_test['label'] = None#need to make sure!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!here is right
    data = pd.concat([tf_train, tf_test])
    data = do_feat_W2V(data)
    train = data[data.label.notnull()]
    test = data[data.label.isnull()]
    train.to_csv(temppath+'2_smooth.csv',index=False)
    test.to_csv(temppath+'2_smooth2_test_smooth.csv',index=False)

#X_loc_train,X_loc_test= get_data()
#bs_smooth(X_loc_train,X_loc_test)
do_feat2()