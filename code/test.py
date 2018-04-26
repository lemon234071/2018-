import os
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
list = [98, 88, 0, 0, 86, 33,0,0,0,0]
list1 = [ 1,2,0,0,5,6,7,8,9,0]
df = pd.DataFrame({'score':list,
                   'aid':list1

})

count = temp.groupby([feat_1, feat_2]).apply(lambda x: x['label'][(x['day'] < day).values].count()).reset_index(
    name=feat_1 + '_' + feat_2 + '_all')
count1 = temp.groupby([feat_1, feat_2]).apply(lambda x: x['label'][(x['day'] < day).values].sum()).reset_index(
    name=feat_1 + '_' + feat_2 + '_1')
# df_ctr = pd.DataFrame(list, columns=['S'])
#
# ctr_labelok = df.loc[df['score'] == 0, ['score', 'aid']]
# ctr_value = ctr_labelok.loc[ctr_labelok['aid']==0, 'score'].count()
# list1.append(10)
# print(list1)
# df['level'] = ''  # add a column
#
# name = 'score'
# #num = len(df[df.score]==0)
# num = len(df[df[name] == 33])
# num = df_feature.apply(lambda r: is_zero(r[name]), axis=1).sum()
#
#
# len df[df.x==0]
#
# 应该是这样， df[df[name] == 0].shape
# 直接(df[name]==0).sum()也行
#
# num_s= Series({'consumptionAbility':3.1})
# num = int
# df = pd.DataFrame([('falcon', 'bird',    389.0),
# #                    ('parrot', 'bird',     24.0),
# #                    ('lion',   'mammal',   80.5),
# #                    ('monkey', 'mammal', np.nan)],
# #                   columns=('name', 'class', 'max_speed'))
# #
# # dff = df.drop('class',axis=
# from sklearn.preprocessing import OneHotEncoder
#
# enc = OneHotEncoder()
#
# enc.fit([[1],[2],[3],[4]])
#
# form = enc.transform([[2],[3],[1],[4]]).toarray()
#
# print(form)
# data = pd.read_csv('../data/cut_train/train_1.csv')
#
# feature = 'aid'
# data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
# enc.fit(data[feature].values.reshape(-1, 1))
# train_a = enc.transform(data[feature].values.reshape(-1, 1))
print('end')