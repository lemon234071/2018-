#!/usr/bin/env python

import os
import sys
import csv
import time
import glob
import zipfile
import numpy as np
import xlearn as xl

from data import *

d = pd.read_csv('',header=0,iterator=True)
d = d.get_chunk()
def train():
    ffm = xl.create_ffm()
    train, test, _ = splitFFM()

    ffm.setTrain(train)
    ffm.setValidate(test)

    model = './modelFFM'
    sTime = time.strftime(
        '%m%d-%H%M', time.localtime(time.time()))
    if not os.path.exists(model): os.mkdir(model)
    model = '%s/xlModel_%s.txt' % (model, sTime)

    params = {
        'epoch'         : 100,
        'metric'        : 'auc',
        'task'          : 'binary',

        'k'             : 4,
        'lr'            : 0.02,
        'lambda'        : 1e-6,
        'stop_window'   : 3,
    }
    ffm.fit(params, model)

def predict():
    ffm = xl.create_ffm()
    _, _, test = splitFFM()

    ffm.setTest(test); ffm.setSigmoid()

    folder = './modelFFM'
    model = sorted(
        glob.glob('./modelFFM/xlModel_*.txt'))[-1]
    output = model.replace('Model', 'Output')
    ffm.predict(model, output)

    df = getMerged('aid', 'uid', kind=2)
    df['score'] = np.loadtxt(output)
    df.to_csv('submission.csv', index=False)

    zipName = '%s/submission.zip' % folder
    with zipfile.ZipFile(zipName, 'w') as f:
        f.write('submission.csv', compress_type=zipfile.ZIP_DEFLATED)


if __name__ == '__main__':
    assert len(sys.argv)==2, 'Failed ...'
    kind = int(sys.argv[1])

    if kind == 1:
        train()
    elif kind == 0:
        predict()

