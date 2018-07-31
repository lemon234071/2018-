#!/usr/bin/env python

import sys
import time
import glob
import zipfile
import numpy as np
import pandas as pd


def combineVal():
    keys = ['aid', 'uid']
    name = '../datas/trainMerged2.csv'

    dfAll = pd.read_csv(name, usecols=keys+['label'])
    dfIds = dfAll[keys]

    files = sorted(glob.glob('./model*/val.csv'))
    for i, f in enumerate(files):
        print f

        df = pd.read_csv(f)
        d = f.split('_')[1].split('/')[0]
        dfAll['score#%s'%d] = dfIds.merge(df)['score']

    sTime = time.strftime('%m%d-%H%M', time.localtime(time.time()))
    dfAll.to_csv('valAll_%s.csv' % sTime, index=False, float_format='%.6f')


if __name__ == '__main__':
    combineVal()

