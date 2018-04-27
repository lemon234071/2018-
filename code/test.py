import os
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import datetime
list = [98, 88, 0, 0, 86, 33,0,0,0,0]
list1 = [ 1,2,0,0,5,6,7,8,9,0]
list2 = [ 1,0,1,0,1,0,1,0,1,0]
df = pd.DataFrame({'score':list,
                   'aid':list1,
                   'uid':list,
                   'label':list2

})


from gensim.models import word2vec

# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 引入数据集
raw_sentences = ["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep"]

# 切分词汇
sentences= [s.encode('utf-8').split() for s in raw_sentences]

# 构建模型
model = word2vec.Word2Vec(sentences, min_count=1)

# 进行相关性比较
model.similarity('dogs','you')
# df = pd.read_csv('../data/origin/train1.csv')
# t_start = datetime.datetime.now()
# count_par = df.groupby(['uid']).apply(lambda x: x.loc[x['label']==1,'label'].count()).reset_index(name='uid' + '_par')#[(x['label'] == 1).values].count()).reset_index(name='uid' + '_par')
# count_all = df.groupby(['uid']).apply(lambda x: x['label'].count()).reset_index(name='uid' + '_all')
# t_end = datetime.datetime.now()
# print('training time: %s' % ((t_end - t_start).seconds))
# print('test')
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