#!/usr/bin/env python

import sys
import time
import glob
import zipfile
import numpy as np
import pandas as pd


def rankAll():
    factor = 1e+4
    keys = ['aid', 'uid']
    name = '../datas/test1Merged.csv'

    dfAll = pd.read_csv(name, usecols=keys)
    dfIds = dfAll[keys]; dfAll['score'] = 0

    files = sorted(glob.glob('./model*/submission.csv'))
    for i, f in enumerate(files):
        print f

        df = pd.read_csv(f)
        score = dfIds.merge(df)['score']
        dfAll['score'] += (factor / score.rank(ascending=False))

    sTime = time.strftime('%m%d-%H%M', time.localtime(time.time()))
    dfAll.to_csv('submission.csv', index=False, float_format='%.6f')
    zipName = './submission_%s.zip' % sTime
    with zipfile.ZipFile(zipName, 'w') as f:
        f.write('submission.csv', compress_type=zipfile.ZIP_DEFLATED)


if __name__ == '__main__':
    rankAll()

