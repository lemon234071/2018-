# -*- coding: utf8 -*-

import sys
import csv
import glob
import random
import zipfile
import numpy as np
import mxnet as mx
from mxnet.gluon import nn

from data import *
from nnHelper import *

logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y%m%d-%H:%M:%S',
    format='%(asctime)s:  %(message)s',
)
###################################################

class MyData(object):
    def __init__(self, ctx, isTrain=True):
        self.ctx = ctx
        self.isTrain = isTrain

        global name
        bs = getBS(name)
        self.batchSize = bs

        self.loadDatas()

    def loadDatas(self):
        logging.info('Loading datas ...')

        self.datas = loadSparse(
            0 if self.isTrain else 1)
        self.len = self.datas.shape[0]
        self.indices = list(range(self.len))
        self.iters = self.len / self.batchSize

        logging.info('Loading datas done ...')

    def __next__(self):
        index, batchSize = self.index, self.batchSize
        indices = self.indices[index:index+batchSize]

        self.index += batchSize
        if self.index > self.len: raise StopIteration

        sData = self.datas[indices].toarray()
        datas = mx.nd.array(
            sData[:, 1:], self.ctx, dtype='int32')
        labels = mx.nd.array(sData[:, 0], self.ctx)

        return datas, labels

    def reset(self):
        self.index = 0
        #if self.isTrain:
        #    #self.indices = random.sample(self.indices,len(self.indices))
        #    random.#shuffle(self.indices)

    def __iter__(self):
        return self

class MyDataVal(object):
    def     __init__(self, ctx):
        self.ctx = ctx

        global name
        bs = getBS(name)
        self.batchSize = bs

        self.loadDatas()

    def loadDatas(self):
        logging.info('Loading datas ...')

        keys = ['aid', 'uid']
        self.ids = getMerged(*keys, kind=4).values

        self.datas = loadSparse(1)
        self.len = self.datas.shape[0]

        logging.info('Loading datas done ...')

    def __next__(self):
        index, batchSize = self.index, self.batchSize
        if index >= self.len: raise StopIteration
        end = min(index + batchSize, self.len)

        ids = self.ids[index:end]
        ids = mx.nd.array(
            ids, self.ctx, dtype='int32')

        sData = self.datas[index:end].toarray()
        datas = mx.nd.array(
            sData[:, 1:], self.ctx, dtype='int32')
        self.index += self.batchSize

        return ids, datas

    def reset(self):
        self.index = 0

    def __iter__(self):
        return self

class MyDataTest(object):
    def __init__(self, ctx):
        self.ctx = ctx

        global name
        bs = getBS(name)
        self.batchSize = bs

        self.loadDatas()

    def loadDatas(self):
        logging.info('Loading datas ...')

        keys = ['aid', 'uid']
        self.ids = getMerged(*keys, kind=2).values

        self.datas = loadSparse(2)
        self.len = self.datas.shape[0]

        logging.info('Loading datas done ...')

    def __next__(self):
        index, batchSize = self.index, self.batchSize
        if index >= self.len: raise StopIteration
        end = min(index + batchSize, self.len)

        ids = self.ids[index:end]
        ids = mx.nd.array(
            ids, self.ctx, dtype='int32')

        sData = self.datas[index:end].toarray()
        datas = mx.nd.array(
            sData[:, 1:], self.ctx, dtype='int32')
        self.index += self.batchSize

        return ids, datas

    def reset(self):
        self.index = 0

    def __iter__(self):
        return self

###################################################

class MyNet(nn.HybridSequential):
    def __init__(self, ctx):
        super(MyNet, self).__init__()
        global name

        self.init(ctx)
        if getSorN(name):
            self.hybridize()
        #self.hybridize()
        self.collect_params().initialize(
            #mx.init.MSRAPrelu(slope=0),
            mx.init.Xavier(),
            ctx
        )

    def init(self, ctx):
        with self.name_scope():
            global name
            global lossKind

            n = lossKind + 1
            self.add(getStart(name))
            self.add(MyIBA(128, 'relu'))#全连接最后一层，另外正则可选+在loss
            self.add(MyIBA(128, 'relu'))
            self.add(nn.Dense(n))
            #self.add(MyIBA(n, 'self'))

class MyNN(MyModel):
    def getNet(self, ctx):
        global lossKind

        return MyNet(ctx), getLoss(lossKind)

    def getModel(self):
        global model; return model

    def getMetric(self):
        global lossKind

        return getMetric(lossKind), 'aucMetric'

    def getData(self, ctx):
        return MyData(ctx, True), MyData(ctx, False)

    def forDebug(self, out):
        pass

    def getTrainer(self, params, iters):
        opt = getOpt(iters)

        return mx.gluon.Trainer(params, opt)
    #return mx.gluon.Trainer(params, 'adam',{ 'clip_gradient': 2})

class MyNNV(MyPredict):
    def __init__(self, gpu=0):
        super(MyNNV, self).__init__(gpuID)
        model = self.getModel()
        self.name = '%s/val.csv' % model

        self.f = open(self.name, 'w')
        self.csv = csv.writer(self.f)
        self.csv.writerow(['aid', 'uid', 'score'])

    def getModel(self):
        global model; return model

    def onDone(self):
        self.f.close()

    def getData(self, ctx):
        return MyDataVal(ctx)

    def getNet(self, ctx):
        model = self.getModel()
        name = sorted(glob.glob(
            '%s/*.params' % model))[-1]

        net = MyNet(ctx)
        net.load_params(name, ctx)
        logging.info('Load %s ...' % name)

        return net

    def preProcess(self, data):
        return data[1]

    def postProcess(self, data, pData, output):
        ids = data[0].asnumpy()
        if lossKind == 0:
            output = mx.nd.sigmoid(output)
            output = output.asnumpy()[:,0]
        if lossKind == 1:
            output = mx.nd.softmax(output)
            output = output.asnumpy()[:,1]
        if lossKind == 2:
            output1 = output[:, 0:2]
            output2 = output[:, 2:3]

            output1 = mx.nd.softmax(output1)
            output1 = output1.asnumpy()[:,1]

            output2 = mx.nd.sigmoid(output2)
            output2 = output2.asnumpy()[:,0]

            output = 0.5*(output1 + output2)

        for id, out in zip(ids, output):
            self.csv.writerow([id[0], id[1], '%.6f'%out])

class MyNNP(MyPredict):
    def __init__(self, gpu=0):
        super(MyNNP, self).__init__(gpuID)
        model = self.getModel()
        self.name = '%s/submission.csv' % model

        self.f = open(self.name, 'w')
        self.csv = csv.writer(self.f)
        self.csv.writerow(['aid', 'uid', 'score'])

    def getModel(self):
        global model; return model

    def onDone(self):
        self.f.close()

        model = self.getModel()
        zipName = '%s/submission.zip' % model
        with zipfile.ZipFile(zipName, 'w') as f:
            f.write(
                self.name, 'submission.csv',
                compress_type=zipfile.ZIP_DEFLATED
            )

    def getData(self, ctx):
        return MyDataTest(ctx)

    def getNet(self, ctx):
        model = self.getModel()
        name = sorted(glob.glob(
            '%s/*.params' % model))[-1]

        net = MyNet(ctx)
        net.load_params(name, ctx)
        print( 'Load %s ...' % name)

        return net

    def preProcess(self, data):
        return data[1]

    def postProcess(self, data, pData, output):
        ids = data[0].asnumpy()
        if lossKind == 0:
            output = mx.nd.sigmoid(output)
            output = output.asnumpy()[:,0]
        if lossKind == 1:
            output = mx.nd.softmax(output)
            output = output.asnumpy()[:,1]
        if lossKind == 2:
            output1 = output[:, 0:2]
            output2 = output[:, 2:3]

            output1 = mx.nd.softmax(output1)
            output1 = output1.asnumpy()[:,1]

            output2 = mx.nd.sigmoid(output2)
            output2 = output2.asnumpy()[:,0]

            output = 0.5*(output1 + output2)

        for id, out in zip(ids, output):
            self.csv.writerow([id[0], id[1], '%.6f'%out])

###################################################
if __name__ == '__main__':
    mx.random.seed(2018)
    random.seed(2018)
    logging.info('All start...')


    assert len(sys.argv)==5, 'Failed ...'
    kind, gpuID = list(map(int, sys.argv[1:3]))

    name = sys.argv[3]
    model = './modelNN_%s' % name

    lossKind = int(sys.argv[4])
    if lossKind == 1:
        model += '#Softmax'
    if lossKind == 2:
        model += '#S2SLoss'

    if kind==1:
        m = MyNN(gpuID); m.train()
    if kind==2:
        m = MyNNV(gpuID); m.predict()
    if kind==0:
        m = MyNNP(gpuID); m.predict()


    logging.info('All done ...')

