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


df  = pd.read_csv('./submission.csv')
test  = pd.read_csv('./test1.csv')
test = pd.merge(test,df,on=['uid','aid'],how='left')
test.fillna(0,inplace=True)
test.to_csv('./submission1.csv',index=False, float_format='%.8f')


