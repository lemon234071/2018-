import multiprocessing #导入多进程模块，实现多进程
import os #验证，打印进程的id
import numpy as np #这个模块是为了实现矩阵乘法

import pandas as pd

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
# data  = pd.read_csv('./datas/trainMerged.csv')
# for concat_feat in data.columns:
#     if data[concat_feat].dtype=='float':
#         try:
#             print(concat_feat)
#             data[concat_feat] = LabelEncoder().fit_transform((10000000*data[concat_feat]).apply(float))
#         except:
#             print('str???????????????????????')
#             data[concat_feat] = LabelEncoder().fit_transform(data[concat_feat].astype(str))
# data.to_csv('./datas/trainMerged.csv',index=False)
#
#
# print('ok')
#
# temp = OrderedDict(
#     sorted(
#         count['uid'].items(),  # return the tuple of dict
#         reverse=1, key=lambda i: i[1],
#     )
# )
#
# df = pd.DataFrame(temp)
# print(df.describe())

#
# df  = pd.read_csv('./submission.csv')
# test  = pd.read_csv('./test1.csv')
# test = pd.merge(test,df,on=['uid','aid'],how='left')
# test.fillna(0,inplace=True)
# test.to_csv('./submission1.csv',index=False, float_format='%.8f')

def read_raw_data(path, max_num):
    f = open(path, 'r')
    features = f.readline()
    features = features.strip().split(',')
    dict = {}
    num = 0
    for line in f:
        if num >= max_num:
            break
        datas = line.strip().split(',')
        for i, d in enumerate(datas):
            if not dict.__contains__(features[i]):
                dict[features[i]] = []
            dict[features[i]].append(d)
        num += 1
    f.close()
    return dict, num

nums = 1000
train_dict, train_num = read_raw_data('C:\\Users\\Administrator\\Downloads\\TencentSPA02-PreA-master\\TencentSPA02-PreA-master\\datas\\trainMerged1.csv', nums)

def count_combine_feature_times(train_data_1, train_data_2, test1_data_1, test1_data_2, test2_data_1, test2_data_2):
    total_dict = {}
    count_dict = {}
    for i, d in enumerate(train_data_1):
        xs1 = d.split(' ')
        xs2 = train_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke  = x1+'|'+x2
                if not total_dict.__contains__(ke):
                    total_dict[ke] = 0
            total_dict[ke] += 1
    for i, d in enumerate(test1_data_1):
        xs1 = d.split(' ')
        xs2 = test1_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke  = x1+'|'+x2
                if not total_dict.__contains__(ke):
                    total_dict[ke] = 0
            total_dict[ke] += 1
    for i, d in enumerate(test2_data_1):
        xs1 = d.split(' ')
        xs2 = test2_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke  = x1+'|'+x2
                if not total_dict.__contains__(ke):
                    total_dict[ke] = 0
            total_dict[ke] += 1
    for key in total_dict:
        if not count_dict.__contains__(total_dict[key]):
            count_dict[total_dict[key]] = 0
        count_dict[total_dict[key]] += 1

    train_res = []
    for i, d in enumerate(train_data_1):
        t = []
        xs1 = d.split(' ')
        xs2 = train_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke = x1 + '|' + x2
                t.append(total_dict[ke])
        train_res.append(max(t))
    test1_res = []
    for i, d in enumerate(test1_data_1):
        t = []
        xs1 = d.split(' ')
        xs2 = test1_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke = x1 + '|' + x2
                t.append(total_dict[ke])
        test1_res.append(max(t))
    test2_res = []
    for i, d in enumerate(test2_data_1):
        t = []
        xs1 = d.split(' ')
        xs2 = test2_data_2[i].split(',')
        for x1 in xs1:
            for x2 in xs2:
                ke = x1 + '|' + x2
                t.append(total_dict[ke])
        test2_res.append(max(t))
    return np.array(train_res), np.array(test1_res), np.array(test2_res), count_dict

def gen_count_dict(data, labels, begin, end):
    total_dict = {}
    pos_dict = {}
    for i, d in enumerate(data):
        if i >= begin and i < end:
            continue
        xs = d.split(' ')
        for x in xs:
            if not total_dict.__contains__(x):
                total_dict[x] = 0
            if not pos_dict.__contains__(x):
                pos_dict[x] = 0
            total_dict[x] += 1
            if labels[i] == '1':
                pos_dict[x] += 1
    return total_dict, pos_dict

def count_pos_feature(train_data, test1_data, test2_data, labels, k, test_only= False, is_val = False):
    nums = len(train_data)
    last = nums
    if is_val:
        last = nums-4739700
        assert last > 0
    interval = last // k
    split_points = []
    for i in range(k):
        split_points.append(i * interval)
    split_points.append(last)
    count_train_data = train_data[0:last]
    count_labels = labels[0:last]

    train_res = []
    if not test_only:
        for i in range(k):
            print( i,"part counting")
            print( split_points[i], split_points[i+1])
            tmp = []
            total_dict, pos_dict = gen_count_dict(count_train_data, count_labels, split_points[i],split_points[i+1])
            for j in range(split_points[i],split_points[i+1]):
                xs = train_data[j].split(' ')
                t = []
                for x in xs:
                    if not pos_dict.__contains__(x):
                        t.append(0)
                        continue
                    t.append(pos_dict[x] + 1)
                tmp.append(max(t))
            train_res.extend(tmp)

    total_dict, pos_dict = gen_count_dict(count_train_data, count_labels, 1, 0)
    count_dict = {-1:0}
    for key in pos_dict:
        if not count_dict.__contains__(pos_dict[key]):
            count_dict[pos_dict[key]] = 0
        count_dict[pos_dict[key]] += 1

    if is_val:
        for i in range(last, nums):
            xs = train_data[i].split(' ')
            t = []
            for x in xs:
                if not total_dict.__contains__(x):
                    t.append(0)
                    continue
                t.append(pos_dict[x] + 1)
            train_res.append(max(t))

    test1_res = []
    for d in test1_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            if not pos_dict.__contains__(x):
                t.append(0)
                continue
            t.append(pos_dict[x] + 1)
        test1_res.append(max(t))

    test2_res = []
    for d in test2_data:
        xs = d.split(' ')
        t = []
        for x in xs:
            if not pos_dict.__contains__(x):
                t.append(0)
                continue
            t.append(pos_dict[x] + 1)
        test2_res.append(max(t))

    return np.array(train_res), np.array(test1_res), np.array(test2_res), count_dict

def combine_to_one(data1, data2):
    assert  len(data1) == len(data2)
    new_res = []
    for i, d in enumerate(data1):
        x1 = data1[i]
        x2 = data2[i]
        new_x = x1 + '|' + x2
        new_res.append(new_x)
    return new_res

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

    return np.array(train_seq), np.array(test1_seq), np.array(test2_seq), seq_emb_dict


#train_res, test1_res, test2_res, f_dict = uid_seq_feature(train_dict['uid'], train_dict['uid'],
#                                                          train_dict['uid'], train_dict['label'])

# new_train_data = combine_to_one(train_dict['uid'], train_dict['aid'])
# train_res, test1_res, test2_res, f_dict = count_pos_feature(new_train_data, new_train_data,
#                                                             new_train_data, train_dict['label'], 5,
#                                                             is_val=False)

from collections import *
from tqdm import tqdm
def gen_pos_neg_aid_fea():
    train_data = pd.read_csv('C:\\Users\\Administrator\\Downloads\\TencentSPA02-PreA-master\\TencentSPA02-PreA-master\\datas\\trainMerged1.csv')
    test2_data = pd.read_csv('C:\\Users\\Administrator\\Downloads\\TencentSPA02-PreA-master\\TencentSPA02-PreA-master\\datas\\trainMerged1.csv')

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

def gen_uid_aid_fea():
    '''
    载入数据，　提取aid, uid的全局统计特征
    '''
    train_data = pd.read_csv('C:\\Users\\Administrator\\Downloads\\TencentSPA02-PreA-master\\TencentSPA02-PreA-master\\datas\\trainMerged1.csv')
    test1_data = pd.read_csv('C:\\Users\\Administrator\\Downloads\\TencentSPA02-PreA-master\\TencentSPA02-PreA-master\\datas\\trainMerged1.csv')
    test2_data = pd.read_csv('C:\\Users\\Administrator\\Downloads\\TencentSPA02-PreA-master\\TencentSPA02-PreA-master\\datas\\trainMerged1.csv')

    ad_Feature = pd.read_csv('C:\\Users\\Administrator\\Downloads\\TencentSPA02-PreA-master\\TencentSPA02-PreA-master\\datas\\trainMerged1.csv')

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

ad_Feature = pd.read_csv('C:\\Users\\Administrator\\Downloads\\TencentSPA02-PreA-master\\TencentSPA02-PreA-master\\datas\\trainMerged1.csv')

te = ad_Feature.groupby(['campaignId'])

te  = te.aid
te  = te.nunique().reset_index().rename(columns={'aid': 'campaignId_aid_nunique'})
print('ok')
# ad_Feature = pd.merge(ad_Feature, ad_Feature.groupby(['campaignId']).aid.nunique().reset_index(
# ).rename(columns={'aid': 'campaignId_aid_nunique'}), how='left', on='campaignId')