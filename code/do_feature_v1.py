rawpath='../data/origin'
temppath='../data/origin'

##########################################################统计特征，转化率为
for feat_1 in ['creativeID','positionID','userID','sitesetID']:
    gc.collect()
    res=pd.DataFrame()
    temp=X_loc_train[[feat_1,'day','label']]
    for day in range(28,32):
        count=temp.groupby([feat_1]).apply(lambda x: x['label'][(x['day']<day).values].count()).reset_index(name=feat_1+'_all')
        count1=temp.groupby([feat_1]).apply(lambda x: x['label'][(x['day']<day).values].sum()).reset_index(name=feat_1+'_1')
        count[feat_1+'_1']=count1[feat_1+'_1']
        count.fillna(value=0, inplace=True)
        count['day']=day
        res=res.append(count,ignore_index=True)
    print(feat_1,' over')
    res.to_csv(temppath+'%s.csv' %feat_1, index=False)


for feat_1,feat_2 in[('positionID','advertiserID'),('userID','sitesetID'),('positionID','connectionType'),('userID','positionID'),
                     ('appPlatform','positionType'),('advertiserID','connectionType'),('positionID','appCategory'),('appID','age'),
                     ('userID', 'appID'),('userID','connectionType'),('appCategory','connectionType'),('appID','hour'),('hour','age')]:
    gc.collect()
    res=pd.DataFrame()
    temp=X_loc_train[[feat_1,feat_2,'day','label']]
    for day in range(28,32):
        count=temp.groupby([feat_1,feat_2]).apply(lambda x: x['label'][(x['day']<day).values].count()).reset_index(name=feat_1+'_'+feat_2+'_all')
        count1=temp.groupby([feat_1,feat_2]).apply(lambda x: x['label'][(x['day']<day).values].sum()).reset_index(name=feat_1+'_'+feat_2+'_1')
        count[feat_1+'_'+feat_2+'_1']=count1[feat_1+'_'+feat_2+'_1']
        count.fillna(value=0, inplace=True)
        count['day']=day
        res=res.append(count,ignore_index=True)
    print(feat_1,feat_2,' over')
    res.to_csv(temppath+'%s.csv' % (feat_1+'_'+feat_2), index=False)

for feat_1,feat_2,feat_3 in[('appID','connectionType','positionID'),('appID','haveBaby','gender')]:
    gc.collect()
    res=pd.DataFrame()
    temp=X_loc_train[[feat_1,feat_2,feat_3,'day','label']]
    for day in range(28,32):
        count=temp.groupby([feat_1,feat_2,feat_3]).apply(lambda x: x['label'][(x['day']<day).values].count()).reset_index(name=feat_1+'_'+feat_2+'_'+feat_3+'_all')
        count1=temp.groupby([feat_1,feat_2,feat_3]).apply(lambda x: x['label'][(x['day']<day).values].sum()).reset_index(name=feat_1+'_'+feat_2+'_'+feat_3+'_1')
        count[feat_1+'_'+feat_2+'_'+feat_3+'_1']=count1[feat_1+'_'+feat_2+'_'+feat_3+'_1']
        count.fillna(value=0, inplace=True)
        count['day']=day
        res=res.append(count,ignore_index=True)
    print(feat_1,feat_2,feat_3,' over')
    res.to_csv(temppath+'%s.csv' % (feat_1+'_'+feat_2+'_'+feat_3), index=False)

