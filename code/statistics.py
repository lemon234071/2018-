import pandas as pd
import numpy as np
from time import sleep
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
import  os
import gc

#统计lable在不同的特征类型下的转化率（！！！！！！！！注意这里若结合不同广告的情况可能观测出良好效果

# 特征初选（！！！！！！！注意需要特征清洗
#https://blog.csdn.net/u013421629/article/details/78769949
# http://www.docin.com/p-108257148.html
# https://blog.csdn.net/sqiu_11/article/details/59487177
# https://www.cnblogs.com/hhh5460/p/5186226.html
# 构造新特征https://blog.csdn.net/shine19930820/article/details/71713680
# 特征组合，这里的组合变化远不限于把已有的特征加减乘除（比如Kernel Tricks之类）
def firsttime_cal_label(train_path,feature,name):
    yi = 0
    df_name = pd.DataFrame()
    while os.path.exists('../data/cut_userfeature/userFeature%s.csv' %yi):
        df_userfeature = pd.read_csv('../data/cut_userfeature/userFeature%s.csv' %yi)
        df_temp = df_userfeature[['uid',name]]
        df_name = df_name.append(df_temp,ignore_index = True)
        yi = yi + 400000
    #df_userfeature = pd.read_csv('../data/origin/%s.csv' %feature)
    #df_name = df_userfeature(['uid',name])
    del df_userfeature
    del df_temp
    gc.collect()
    print('feature load over')
    df_temp = df_name[name]
    df_temp.describe(include='all').to_csv('../data/feature/' + name + '_statistic.csv')
    #print('statistic finished')
    df_train = pd.read_csv(train_path)
    df_train = pd.merge(df_train, df_name,how='left',on='uid')
    #print('merge finished')
    df_temp = df_train[name]
    df_temp.describe(include='all').to_csv('../data/feature/' + 'train_' + name + '_statistic.csv')
    #print('train statistic finished')
    df_train.to_csv('../data/feature/' + name + '_label.csv',index=False)
    #print('save csv finished')
    per_name = df_train.groupby(name).apply(lambda df: np.mean(df["label"])).reset_index(name='per_'+name)
    print('cal finished')
    per_name.to_csv('../data/feature/per_' + name + '.csv',index=False)


def notfirst_cal_label(source_name,feature_name):
    df_feature = pd.read_csv('../data/feature/'+ source_name +'.csv')
    # df_temp = df_feature[feature_name]#train集统计特征
    # df_temp.describe(include='all').to_csv('../data/feature/' + feature_name + '_statistic.csv')
    per_name = df_feature.groupby(feature_name).apply(lambda df: np.mean(df["label"])).reset_index(name='per_'+feature_name)
    per_name.to_csv('../data/feature/per_' + feature_name + '.csv',index=False)

def statistic_adfeature():
    adfeature = ['advertiserId','campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType']
    yi = 1
    df_name = pd.DataFrame()
    while os.path.exists('../data/merge_feature/train%s.csv' % yi):
        df_adfeature = pd.read_csv('../data/merge_feature/train%s.csv' % yi)#此处有bug,label值-1改为了0
        df_name = df_name.append(df_adfeature, ignore_index=True)
        yi = yi + 1
    # df_userfeature = pd.read_csv('../data/origin/%s.csv' %feature)
    # df_name = df_userfeature(['uid',name])
    del df_adfeature
    gc.collect()
    print('feature load over')
    df_name.describe(include='all').to_csv('../data/feature/adfeature_statistic.csv')
    print('statistic finished')
    per_name = pd.DataFrame()
    for feature in adfeature:
        per_temp = df_name.groupby(feature).apply(lambda df: np.mean(df["label"])).reset_index(name='per_' + feature)
        per_name = per_name.append(per_temp, ignore_index=True)
    print('cal finished')
    per_name.to_csv('../data/feature/per_adfeature.csv', index=False)


def show_graghic(path, x, y):
    sns.set(style="white", color_codes=True)
    df_feature = pd.read_csv(path)
    df_feature.head()

    graghic = df_feature.plot(kind="scatter", x=x, y=y)
    fig = graghic.get_figure()
    fig.savefig('../data/picture/' + y + '.png')
    fig.show()
    sleep(5)
    #多种类变量
    # sns.FacetGrid(df_feature, hue="Species", size=5) \
    #     .map(plt.scatter, col_name, 'per_'+col_name) \
    #     .add_legend()

    # if you want more  vision turn to https://www.kaggle.com/benhamner/python-data-visualizations
    # 可以制定列，对该列各取值作统计：
    #
    # label_dis = df.label.value_counts()
    # ax = label_dis.plot(title='label distribution', kind='bar', figsize=(18, 12))
    # fig = ax.get_figure()
    # fig.savefig('label_distribution.png')

def cul_isnan(headers):

    dic_num= {}
    for name in headers:
        df_feature = pd.read_csv('../data/merge_feature/' + name + '_label.csv')
        #null_count = pd.DataFrame(df_feature.isna().sum())
        num = len(df_feature[df_feature[name] == 0])
        dic_num[name] = num
    df_all = pd.DataFrame({'sort': [i for i in dic_num], 'value': [dic_num[i] for i in dic_num]})
    df_all.to_csv('../data/feature/train_na_count.csv')

    # yi = 0
    # df_name = pd.DataFrame()
    # while os.path.exists('../data/cut_userfeature/userFeature%s.csv' %yi):
    #     df_userfeature = pd.read_csv('../data/cut_userfeature/userFeature%s.csv' %yi)
    #     for name in headers:
    #         df_temp = df_userfeature[['uid',name]]
    #         df_name = df_name.append(df_temp,ignore_index = True)
    #     yi = yi + 400000

def fill_na():
    yi = 0
    df_name = pd.DataFrame()
    while os.path.exists('../data/cut_userfeature/userFeature%s.csv' %yi):
        df_userfeature = pd.read_csv('../data/cut_userfeature/userFeature%s.csv' %yi)
        df_temp = df_userfeature[['carrier','consumptionAbility']]
        df_name = df_name.append(df_temp,ignore_index = True)
        yi = yi + 400000
    #df_name.to_csv('../data/merge_feature/need_fillna.csv',index=False)
    df_name['carrier'].fillna('0')
    df_name['consumptionAbility'].fillna('0')
    df_num = df_name.drop('carrier',axis=1)
    num_s = df_num[df_num['consumptionAbility'] != 0].mean()
    num = int(num_s['consumptionAbility'])
    df_name[df_name['consumptionAbility']==0] = num

    df_num = df_name.drop('consumptionAbility',axis=1)
    num_s = df_num[df_num['carrier'] != 0].mean()
    num = int(num_s['carrier'])
    df_name[df_name['carrier']==0] = num
    df_name.to_csv('../data/merge_feature/finish_fillna.csv', index=False)

#主函数
if __name__ == '__main__':

    #firsttime_cal_label('../data/origin/train.csv', 'usFeature','marriageStatus')#write what feature relation to label you want
    #notfirst_cal_label('LBS_label', 'LBS')
    #show_graghic('../data/feature/per_LBS.csv','LBS')

    headers = [ 'age', 'gender',  'education', 'consumptionAbility', 'LBS', 'carrier', 'house']
    vector_feature = ['ct', 'os','marriageStatus','appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    # for feature in headers:
    #     firsttime_cal_label('../data/origin/train.csv', 'usFeature', feature)
    #统计
    #statistic_adfeature()
    # 绘制
    #show_graghic('../data/feature/per_LBS.csv', 'LBS', 'per_LBS')
    #cul_isnan(headers)
    fill_na()
    print('finished')