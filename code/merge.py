import pandas as pd
import os

def merge_adfeature():
    xi = 1
    df_adfeature = pd.read_csv('../data/origin/adFeature.csv')

    while os.path.exists('../data/cut_train/train_%s.csv' %xi):
        df_train = pd.read_csv('../data/cut_train/train_%s.csv' %xi)
        #df_train.loc[df_train['label'] == -1, 'label'] = 0
        df_train = pd.merge(df_train, df_adfeature, how='left', on = 'aid')
        df_train = df_train.fillna('0')#空置填充0
        df_train.to_csv('../data/merge_feature/train%s.csv' %xi, index = False)
        print(xi)
        xi = xi + 1
    predict = pd.read_csv('../data/origin/test1.csv')
    predict['label'] = -1
    predict = pd.merge(predict, df_adfeature, how='left', on='aid')
    predict = predict.fillna('0')
    predict.to_csv('../data/merge_feature/test%s.csv' %xi, index = False)
    print('next')
# def merge_userfeature():
#     userfeature = ['age','gender','marriageStatus',' education','consumptionAbility','LBS','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','appIdInstall','appIdAction','ct','os','carrier','house',]
#     yi = 0
#     while os.path.exists('../data/cut_userfeature/userFeature%s.csv' %yi):
#         df_userfeature = pd.read_csv('../data/cut_userfeature/userFeature%s.csv' %yi)
#         xi = 1
#         while os.path.exists('../data/merge_feature/train%s.csv' %xi):
#             df_train = pd.read_csv('../data/merge_feature/train%s.csv' %xi)
#             df_train = pd.merge(df_train, df_userfeature, how='left', on = 'uid',suffixes=('_1', '_2'))
#             df_train =df_train.fillna('-1')
#             for feature in userfeature:
#                 df_train[feature].values = max(df_train[feature+'_1'].values,df_train[feature+'_2'].values)
#                 df_train = df_train.drop([feature+'_1',feature+'_2'],axis =1)#delete origin feature_1_2
#             df_train.to_csv('../data/merge_feature/train%s.csv' %xi, index = False)
#             print(xi)
#             xi = xi + 1
#         yi = yi + 400000
def merge_userfeature():
    df_userfeature = pd.read_csv('../data/origin/usFeature.csv')
    print('next')
    xi = 1
    while os.path.exists('../data/merge_feature/train%s.csv' %xi):
        df_train = pd.read_csv('../data/merge_feature/train%s.csv' %xi)
        df_train = pd.merge(df_train, df_userfeature, how='left', on = 'uid')
        print('merge finished')
        df_train = df_train.fillna('-1')
        df_train.to_csv('../data/merge_feature/train%s.csv' %xi, index = False)
        print(xi)
        xi = xi + 1

if __name__ =='__main__':
    merge_adfeature()
    #merge_userfeature()