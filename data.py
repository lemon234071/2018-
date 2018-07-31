#!/usr/bin/env python
# encoding=utf8
import gc
import os
import csv
import json
import random
import hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
from scipy.sparse import csr_matrix
from collections import defaultdict
from collections import OrderedDict

User = ['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','ct','marriageStatus','os']
Ad = ['aid']
User_Ad_idx = './datas/features/UA_idx.txt'
############################################

def mkdir(folder):
    folder = './datas/features/%s' % folder
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder
import gc
def getMerged(*keys, **kwargs):
    kind = kwargs.get('kind', 0)
    name1 = './datas/trainMerged.csv'
    name2 = './datas/test1Merged.csv'

    for name in [name1, name2]:
        if os.path.exists(name): continue
        print( "@@@@@", name)
        ad = pd.read_csv('./datas/adFeature.csv')
        if not os.path.exists(name.replace('Merged', '')):
            continue
        df = pd.read_csv(name.replace('Merged', ''))
        user = pd.read_csv('./datas/userFeature.csv')

        notUsed = ['appIdAction', 'appIdInstall']
        user.drop(notUsed, axis=1, inplace=True)

        df = df.merge(ad).merge(user)
        if 'label' not in df: df['label'] = 0

        columns = sorted(df.columns.tolist())
        columns.remove('label'); columns.append('label')
        df.to_csv(name, index=False, columns=columns)
        print(' ok ',name)
        del df,user
        gc.collect()
    print('merge ok')
    # for name in [name1, name2]:
    #     df = pd.read_csv(name)
    #     notUsed = ['appIdAction', 'appIdInstall']
    #     if notUsed in df.columns:
    #         df.drop(notUsed, axis=1, inplace=True)
    #         if 'label' not in df: df['label'] = 0
    #         columns = sorted(df.columns.tolist())
    #         columns.remove('label'); columns.append('label')
    #         df.to_csv(name, index=False, columns=columns)


    name3 = './datas/trainMerged1.csv'
    name4 = './datas/trainMerged2.csv'

    exists = os.path.exists
    if not (exists(name3) and exists(name4)):
        random.seed(2018)

        #t1n, t2n = int(7e+6), int(1e+6)

        df = pd.read_csv(name1)
        t1n, t2n = int(0.7 * len(df)), int(0.3 * len(df))
        indices = list(range(len(df)))
        #random.shuffle(indices)
        indices3 = indices[:t1n]
        indices4 = indices[-t2n:]

        columns = sorted(df.columns.tolist())
        columns.remove('label'); columns.append('label')
        df.take(indices3).to_csv(
            name3, index=False, columns=columns)
        df.take(indices4).to_csv(
            name4, index=False, columns=columns)
        del df
        gc.collect()
    print('merge ok')
    if not keys: keys = None
    if kind==1: return pd.read_csv(name1, usecols=keys)
    if kind==2: return pd.read_csv(name2, usecols=keys)
    if kind==0: return pd.concat([
        pd.read_csv(name1, usecols=keys),
        pd.read_csv(name2, usecols=keys)
    ], axis=0)
    if kind==3: return pd.read_csv(name3, usecols=keys)
    if kind==4: return pd.read_csv(name4, usecols=keys)

def getMergedA(*keys, **kwargs):
    kind = kwargs.get('kind', 0)
    name1 = './datas/trainMergedA.csv'
    name2 = './datas/test1MergedA.csv'

    for name in [name1, name2]:
        if os.path.exists(name): continue

        ad = pd.read_csv('./datas/adFeature.csv')
        df = pd.read_csv(name.replace('Merged', ''))
        user = pd.read_csv('./datas/userFeature.csv')

        df = df.merge(ad).merge(user)
        if 'label' not in df: df['label'] = 0

        columns = sorted(df.columns.tolist())
        columns.remove('label'); columns.append('label')
        df.to_csv(name, index=False, columns=columns)

    if not keys: keys = None
    if kind==1: return pd.read_csv(name1, usecols=keys)
    if kind==2: return pd.read_csv(name2, usecols=keys)
    if kind==0: return pd.concat([
        pd.read_csv(name1, usecols=keys),
        pd.read_csv(name2, usecols=keys),
    ], axis=0)

############################################

def getUF():
    name = './datas/userFeature.data'

    for f in open(name):
        user = {}
        for uf in f.strip().split('|'):
            uf = uf.split(' ')
            user[uf[0]] = list(map(int, uf[1:]))

        yield user

def userMaxLen():
    folder = mkdir('msg')
    maxLen = defaultdict(int)
    maxFile = '%s/maxLen.txt' % folder

    if not os.path.exists(maxFile):
        for i, user in enumerate(getUF()):
            for key, vs in user.items():
                maxLen[key] = max(maxLen[key], len(list(vs)))

        with open(maxFile, 'w') as f:
            for k in sorted(maxLen.keys()):
                f.write('%s %s\n' % (k, maxLen[k]))
            #print( 'Saved to %s ...' % saveName)

    for line in open(maxFile):
        k, v = line.strip().split()
        maxLen[k] = int(v)

    return maxLen

def userFeature():
    default = { k: 0 for k in userMaxLen() }
    keys = sorted(default.keys())
    keys.remove('uid'); keys = ['uid'] + keys

    saveName = './datas/userFeature.csv'
    if os.path.exists(saveName): return

    with open(saveName, 'w') as sf:
        dw = csv.DictWriter(sf, fieldnames=keys)
        dw.writeheader()

        for i, user in enumerate(getUF()):
            data = default.copy()
            for key, vs in user.items():
                data[key] = ' '.join(list(map(str,vs)))
            dw.writerow(data)

            if (i+1) % 100000 == 0:
                print( "%8d..." % (i+1))
        print( 'Saved to %s ...' % saveName)

############################################

def getFFM():
    print( 'start getFFM()')
    folder = mkdir('ffm')
    datas = [
        ['trainMerged', 'trainFFM'],
        ['test1Merged', 'testFFM'],
    ]
    print( 'start get FFM data in getFFM()')
    result, flag = [], True

    for _, save in datas:
        save = '%s/%s.txt' % (folder, save)
        result.append(save)
        if not os.path.exists(save):
            flag = False
    if flag:
        return result

    count, _  = setCount()#return the dict-->{feat:{onehot : index}} and the number of dim
    fields = sorted(count.keys())
    UA_idx, User_idx, Ad_idx = open(User_Ad_idx, 'w'), '', ''
    for k, v in enumerate(fields):
        if v in User:
            User_idx += ' '+str(k)
        elif v in Ad:
            Ad_idx += ' '+str(k)
    if not UA_idx.closed:
        UA_idx.write('%s\n' % User_idx)
        UA_idx.write('%s\n' % Ad_idx)
        UA_idx.close()

    result = []

    for name, save in datas:
        print( 'start gen data ', save)
        save = '%s/%s.txt' % (folder, save)
        result.append(save)
        if os.path.exists(save): continue

        txt = open(save, 'w')
        name = './datas/%s.csv' % name

        for i, line in enumerate(open(name)):#for the merged df
            vvs = line.strip().split(',')
            if i == 0: fNs = vvs; continue#fns feature name

            s = ''
            for k, vs in enumerate(vvs):#vvs value of this line
                if fNs[k] == 'label':
                    s = ('%d'%(vs=='1')) + s#replace the -1 label to 0
                    continue

                ii = fields.index(fNs[k])#index of the feature
                for v in vs.split(' '):#all value of one feature
                    try:
                        float(v)
                    except:
                        continue
                    v = str(int(float(v)))
                    c = count[fNs[k]][v]#index of the value in his feature
                    # try:
                    #     float(v)
                    # except:
                    #     continue
                    # if float(v) and c:
                    if int(float(v)) and c:
                        s += ' %d:%d:1' % (ii, c)#特征index 总体取值index 1131，0直空值过滤了（这不一定合理
            txt.write('%s\n' % s)
        txt.close()


        print( 'Saved to %s ...' % save)
    return result

def splitFFM():
    t1, save3 = getFFM()#get conut path(train ,test ) of file which have fomat (index of feature : index fo all dim : 1)

    folder = mkdir('ffm')
    exists = os.path.exists
    save1 = '%s/t1.txt' % folder
    save2 = '%s/t2.txt' % folder
    if exists(save1) and exists(save2):
        return save1, save2, save3

    random.seed(2018)
    lines = open(t1).readlines()#read the tranFFM
    #random.shuffle(lines)
    #t1n, t2n = int(7e+6), int(1e+6)
    t1n, t2n = int(0.7 * len(lines)), int(0.3 * len(lines))
    print( 'start gen train valid data')
    with open(save1, 'w') as txt:
        txt.writelines(lines[:t1n])
    print( 'Saved to %s ...' % save1)
    with open(save2, 'w') as txt:
        txt.writelines(lines[-t2n:])
    print( 'Saved to %s ...' % save2)

    return save1, save2, save3

def getCount():
    '''count values for every column,
    when dtype==O, expend it'''
    folder = mkdir('msg')
    saveName = '%s/counts.json' % folder
    print( 'start take get counts.json')
    if os.path.exists(saveName):
        print( 'getCount already done!!')
        return json.load(open(saveName))

    datas = OrderedDict()
    print( 'start get merged data')
    df = getMerged(kind=0)#.drop('label', 1)
    del df['label']
    gc.collect()
    for fn in df:
        gc.collect()
        print( 'start count', fn)
        if df[fn].dtype == 'object':
            # df_cnt = pd.DataFrame()
            # r = ' '.join( df[fn].dropna().tolist() )
            # df_cnt[fn] = r.split(' ')
            # datas[fn] = df_cnt[fn].value_counts().to_dict()
            df[fn] = df[fn].astype(str)
            d, ds = [], df[fn].str.split(' ', expand=False)
            for s in ds: d.extend(list(map(int, s)))
            datas[fn] = Counter(d)

        else:
            #datas[fn] = df[fn].dropna().astype('int').value_counts().to_dict()
            datas[fn] = Counter(df[fn])
        del df[fn]

    json.dump(datas, open(saveName, 'w'))
    print( 'Saved to %s ...' % saveName)

    return datas

def setCount():
    print( 'start setCount')
    count = getCount()#每个特征一个field,a dict of dict [('lbs',{value:count})]
    fields = sorted(count.keys())#list of feature['lbs',''...]

    offset = 0
    for k in fields:
        print( 'get %s fields in var range' % k)
        t = 100 if k=='uid' else 100#threshold of every feature

        temp = OrderedDict(#one col feature's dic--> value : count
            sorted(
                count[k].items(),#return the tuple of dict
                reverse=1, key=lambda i: i[1],#sort by i[1]
            )
        )
        count[k] = defaultdict( int, {#dict --> {feat:{value : index}}
            str(u): i+1+offset
            for i, u in enumerate(temp)
            if temp[u]>=t and int(float(u))!=0
        })#低于100的过滤掉了，给key对应自己index!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!all mass for this op
        offset += len(count[k])
    print('setcount ok')
    return count, offset#return the dict and the number of dim

def getFFMDim():
    print( 'start getFFMdim()')
    t1, t2, t3 = splitFFM()#get the train ,validate , test FFMfile(::)
    print( 'end splitFFM()')
    UA_idx = open(User_Ad_idx,'r')
    User, Ad = UA_idx.readline().strip().split(' '), UA_idx.readline().strip().split(' ')
    UA_idx.close()

    folder = mkdir('msg')
    saveName = '%s/ffmDim.csv' % folder
    if os.path.exists(saveName):
        df = pd.read_csv(saveName)
        return OrderedDict(
            zip(df.columns, df.values[0])), User, Ad

    datas = defaultdict(int)
    for t in [t1, t2, t3]:
        print( 'start countFFMDim in [%s]' % t)
        for line in open(t):
            temp = defaultdict(int)

            s = line.strip().split(' ')
            fs = map(int, map(
                lambda x: x.split(':')[0], s[1:]))

            for f in fs: temp['field#%03d'%f] += 1
            for key in temp:#number of value of each field
                datas[key] = max(temp[key], datas[key])#所有数据中每一行中出现该field次数最大数，也就是每个特征每行取值最大个数
    for key in datas:
        datas[key] = [datas[key]]

    pd.DataFrame.from_dict(datas).to_csv(
        saveName, index=False, columns=sorted(datas.keys()))
    print( 'Saved to %s ...' % saveName)
    
    return OrderedDict([
        (k, datas[k][0]) for k in sorted(datas.keys())]), User, Ad#return the dimlist in witch include the max val_num of the col

def getSparse():
    offsets = [0]#every field's begin index
    ffmDim, _, _ = getFFMDim()
    for k in sorted(ffmDim.keys()):
        offsets.append(offsets[-1]+ffmDim[k]) 

    print( 'start splitFFM')
    t1, t2, t3 = splitFFM()#get the train ,validate , test FFMfile(::)
    print( 'start genFFM data')
    folder = mkdir('ffm')
    save1 = '%s/t1X.npz' % folder
    save2 = '%s/t2X.npz' % folder
    save3 = '%s/testX.npz' % folder

    ydfs = [
        [t1, save1], [t2, save2], [t3, save3],
    ]

    flag = True
    for t, save in ydfs:
        if not os.path.exists(save):
            flag = False
    if flag: return save1, save2, save3
    print( 'end genFFM data')
    #data, indices, indptr = [], [], [0]#data, indices, indptr ::index of 1131 of all of each line,每个非0元素行内相对位置,这行多少个元素（起始index
    print( 'start gen sparse matrix data')
    for k, (t, save) in enumerate(ydfs):
        print( 'processing[%d], libsvm src[%s], npz dest[%s]' % (k, t, save))
        ydfs[k].append(0)
        data, indices, indptr = [], [], [0]  # data, indices, indptr ::index of 1131 of all of each line,每个非0元素行内相对位置,这行多少个元素（起始index
        for line_number, line in enumerate(open(t)):
            ydfs[k][-1] += 1#line num of this file
            
            s = line.strip().split(' ') # label field:key:value
                                        # value is always 1
            i = list(map(int, s[:1] + list(map(
                lambda x: x.split(':')[1], s[1:])))) #get index
            fs = list(map(int, list(map(
                lambda x: x.split(':')[0], s[1:])))) #get field
            
            data.extend(i) #VALUE IN EVERY ROW
            indices.append(0)
            ios = [0] * (len(offsets)) # offset存每个field的特征数目
            #print 'start process line[%s]' % line_number
            #print 'len(ios) is:', len(ios)
            #print ios[29]
            #print 'max(fs) is:', np.max(fs)
            #print 'fs is:', fs
            #fs = list(set(fs))
            for f in fs:
                ios[f] += 1#value num of this line
                #print('error:',len(offsets),len(ios),f)
                indices.append(offsets[f] + ios[f]) #每个非0元素行内相对位置
            #print 'ios is:', ios
            #print 'indices is:', indices
            #data.extend(i)
            indptr.append(indptr[-1] + len(i)) #这一行到哪个元素
        rows, cols = len(indptr) - 1, np.max(indices) + 1
        csr = csr_matrix((data, indices, indptr), shape=(rows, cols))  #
        t, save, l = ydfs[k]
        sparse.save_npz(save, csr[:-1], compressed=True)
    # rows, cols = len(indptr)-1, np.max(indices)+1
    # '''
    # csr_matrix((data, indices, indptr), [shape=(M, N)])
    # is the standard CSR representation where the column indices for row i are stored in indices[indptr[i]:indptr[i+1]] and their         corresponding values are stored in data[indptr[i]:indptr[i+1]]. If the shape parameter is not supplied, the matrix dimensions are     inferred from the index arrays.
    # indptr表示某一行在data中的range，indices表示每条数据在第几列
    # '''
    # csr = csr_matrix(
    #     (data, indices, indptr), shape=(rows, cols))#data, indices, indptr ::index of 1131 of all of each line,每个非0元素行内相对位置,这行多少个元素（起始index
    #
    # start, end = 0, 0
    # for t, save, l in ydfs:
    #     start, end = end, end + l
    #     sparse.save_npz(
    #         save, csr[start:end], compressed=True)
    print( 'end gen sparse matrix data')
    return save1, save2, save3

def loadSparse(kind):
    return sparse.load_npz(getSparse()[kind])

############################################
from sklearn.preprocessing import LabelEncoder
if __name__ == '__main__':
    #userFeature

    mytype = {'uid': np.uint16, 'LBS': np.uint16, 'adCategoryId': np.uint16, 'advertiserId': np.uint16,
              'age': np.uint16, 'aid': np.uint16, 'campaignId': np.uint16,
              'carrier': np.uint16, 'consumptionAbility': np.uint16, 'creativeId': np.uint16, 'creativeSize': np.uint16,
              'ct': object, 'education': np.uint16, 'gender': np.uint16, 'house': np.uint16, 'interest1': object,
              'interest2': object, 'interest3': object, 'interest4': object, 'interest5': object, 'kw1': object,
              'kw2': object, 'kw3': object, 'marriageStatus': object, 'os': object, 'productId': np.uint16,
              'productType': np.uint16, 'topic1': object, 'topic2': object, 'topic3': object, 'label': np.int8,
              'appIdAction': object, 'appIdInstall': object}
    # df = pd.read_csv('./datas/8g/8g.csv', header=0, iterator=True,dtype=mytype, usecols=mytype.keys())
    # data = df.get_chunk(80000)
    # data.to_csv('./datas/trainMerged.csv',index=False)
    # data = df.get_chunk(20000)
    # data.to_csv('./datas/test1Merged.csv', index=False)
    # print('data ok')
    # del data
    # gc.collect()
    print(getFFM())
    print(setCount()[1])
    loadSparse(0)
    #os.system('python3 nn.py 1 0 test 0')
