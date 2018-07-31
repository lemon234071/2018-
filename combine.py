#!/usr/bin/env python

import sys
import time
import glob
import zipfile
import numpy as np
import pandas as pd


def combineAll(t=1):
    dfID, dfAll = None, None
    print 'Combining with t = %.3f ...' % t
    files = sorted(glob.glob('./model*/submission.zip'))
    for i, f in enumerate(files):
        print f
        d = f.split('_')[1].split('/')[0]
        # zf = zipfile.ZipFile(f, 'r', zipfile.ZIP_DEFLATED)
        # zf.extract('submission.csv', '/dev/shm/submission_%s' % d)

        df = pd.read_csv(f.replace('zip', 'csv'))
        if dfID is None:
            dfID, dfAll = df[['aid', 'uid']], df
            continue

        dfAll['score'] += dfID.merge(df)['score']
    dfAll['score'] /= len(files)
    if t != 1: dfAll.loc[dfAll['score']<t, 'score'] = 0

    sTime = time.strftime('%m%d-%H%M', time.localtime(time.time()))
    dfAll.to_csv('submission.csv', index=False, float_format='%.6f')
    zipName = './submission_%s.zip' % sTime
    with zipfile.ZipFile(zipName, 'w') as f:
        f.write('submission.csv', compress_type=zipfile.ZIP_DEFLATED)


if __name__ == '__main__':
    combineAll(
        1 if len(sys.argv)<=1
        else float(sys.argv[1])
    )

