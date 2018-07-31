# -*- coding: utf8 -*-

import os
import random
import logging
import mxnet as mx
import numpy as np
from mxnet.gluon import nn
from mxnet import autograd
from mxnet.gluon import loss
import os.path
import time
from data import *
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y%m%d-%H:%M:%S',
    format='%(asctime)s:  %(message)s',
)

###################################################

def getBS(name):
    # bSs = {
    #     'dffm31'    : 200,
    #     'dfcm12'    : 500,
    #     'dfcm13'    : 500,
    # }
    #
    # if name in bSs:
    #     return bSs[name]
    return 2048

def getOpt(iters):
    # opt = mx.optimizer.AdaGrad(
    #     wd=1e-3,
    #     learning_rate=2e-4,
    # )
    opt = mx.optimizer.Adam(wd=1e-3,
    learning_rate=1e-2,)

    return opt

def getSorN(name):

    models = (
        'dfr',
    )

    return name not in models

def getStart(name):
    dims, User, Ad= getFFMDim()
    dims = dims.values()
    inC = 4989
    models = {
        'test': lambda: Chen_DFM(dims, inC, 8, 128, User, Ad),  # 2048
        'test1': lambda: Guo_DFFM(dims, inC, 8, 256),  # 4096
        'testdin': lambda: MyDIN(dims, inC, 8, 128, User, Ad),  # XXXX, 7325, XXXX

        'dcn'   : lambda: MyDCN(dims, inC, 8, 64, 3),
        'dfcm'  : lambda: MyDFCM(dims, inC, 8, 64),       # 7370, 7406, XXXX
        'dfcm11': lambda: MyDFCM(dims, inC, 16, 128),     # XXXX, 7431, XXXX
        'dfcm12': lambda: MyDFCM(dims, inC, 32, 128),     # XXXX, 7482, XXXX
        'dfcm13': lambda: MyDFCM(dims, inC, 64, 256),     # XXXX, 7485, XXXX

        'dfr'   : lambda: MyDFR(dims, inC, 8, 64),
        'dfu'   : lambda: MyDFU(dims, inC, 8, 64),        # 7269, XXXX, XXXX
        'dfm'   : lambda: MyDFM(dims, inC, 8, 64),        # 7256, 7280, XXXX

        'dfm2'  : lambda: MyDFM2(dims, inC, 8, 64),       # XXXX, 7369, XXXX
        'dfm21' : lambda: MyDFM2(dims, inC, 16, 128),     # XXXX, 7410, XXXX
        'dfm22' : lambda: MyDFM2(dims, inC, 32, 128),     # XXXX, 7445, XXXX
        'dfm23' : lambda: MyDFM2(dims, inC, 32, 256),     # XXXX, 7450, XXXX
        'dfm24' : lambda: MyDFM2(dims, inC, 64, 256),     # XXXX, 7468, XXXX
        'dfm25' : lambda: MyDFM2(dims, inC, 128, 256),    # XXXX, 7468, XXXX
        'dfm26' : lambda: MyDFM2(dims, inC, 256, 256),    # XXXX, 7475, XXXX

        'dfz'   : lambda: MyDIN(dims, inC, 8, 64),        # XXXX, 7325, XXXX
        'dfz11' : lambda: MyDFZ(dims, inC, 16, 128),      # XXXX, 7402, XXXX
        'dfz12' : lambda: MyDFZ(dims, inC, 32, 128),      # XXXX, 7443, XXXX
        'dfz13' : lambda: MyDFZ(dims, inC, 32, 256),      # XXXX, 7439, XXXX
        'dfz14' : lambda: MyDFZ(dims, inC, 64, 256),      # XXXX, 7456, XXXX
        'dfz15' : lambda: MyDFZ(dims, inC, 128, 256),     # XXXX, 7463, XXXX
        'dfz16' : lambda: MyDFZ(dims, inC, 256, 256),     # XXXX, 7469, XXXX

        'din'   : lambda: MyDIN(dims, inC, 8, 64),        # XXXX, 7325, XXXX
        'dfin'  : lambda: MyDFIN(dims, inC, 8, 64),       # 7399, 7415, XXXX
        'dfcn'  : lambda: MyDFCN(dims, inC, 8, 64),       # 7415, XXXX, XXXX

        'dffm'  : lambda: MyDFFM(dims, inC, 8, 64),       # 7395, 7439, XXXX
        'dffm2' : lambda: MyDFFM2(dims, inC, 8, 64),      # 7396, 7444, XXXX
        'dffm3' : lambda: MyDFFM3(dims, inC, 8, 256),      # 2048
        'dffm31': lambda: MyDFFM3(dims, inC, 16, 64),     # XXXX, 7462, XXXX
    }

    return models[name]()

def getLoss(lossKind):
    losses = {
        0: MyLoss,
        1: MyLoss2,
        2: MyLoss3,
    }
    return losses[lossKind]()

def getMetric(lossKind):
    metrics = {
        0: MyMetric,
        1: MyMetric2,
        2: MyMetric3,
    }
    return metrics[lossKind]()

def randomRange(start, end):
    r1 = r2 = 0
    while(r1 == r2):
        r1 = random.randint(start, end)
        r2 = random.randint(start, end)
    t1, t2 = min(r1,r2), max(r1,r2)
    return t1, t2

###################################################

class MyBA(nn.HybridSequential):
    def __init__(self, act='relu'):
        super(MyBA, self).__init__()

        with self.name_scope():
            self.add(nn.BatchNorm())
            self.add(MyAct(act))

class MyAct(nn.HybridBlock):
    def __init__(self, act='relu'):
        super(MyAct, self).__init__()
        self.act = act

    def hybrid_forward(self, F, x):
        if self.act == 'self': return x
        return F.Activation(x, self.act)

class MyIBA(nn.HybridSequential):
    def __init__(self, c, act='relu'):
        super(MyIBA, self).__init__()

        with self.name_scope():
            self.add(nn.Dense(c))#全连接+bias
            self.add(MyBA(act))#可选

class MyCBA(nn.HybridSequential):
    def __init__(self, c, act='relu'):
        super(MyCBA, self).__init__()

        with self.name_scope():
            self.add(nn.Conv1D(*c))
            self.add(MyBA(act))

class MyRes(nn.HybridBlock):
    def __init__(self, c1, c2):
        super(MyRes, self).__init__()

        with self.name_scope():
            self.opr = MyAct('relu')
            self.op1 = MyIBA(c1, 'self')
            self.op2 = MyIBA(c2, 'relu')

    def hybrid_forward(self, F, x):
        return self.op2(self.opr(self.op1(x)+x))

class MyRes2(nn.HybridSequential):
    def __init__(self, c1, c2, num):
        super(MyRes2, self).__init__()

        with self.name_scope():
            for i in range(num):
                self.add(MyRes(c1, c1))
            self.add(MyRes(c1, c2))

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
class MyDPO(nn.HybridSequential):
    def __init__(self, rate=.5):
        super(MyDPO, self).__init__()
        self.rate = rate

        with self.name_scope():
            self.add(nn.Dropout(.5))

class MyIDP(nn.HybridSequential):
    def __init__(self, c, c1):
        super(MyIDP, self).__init__()

        with self.name_scope():
            self.add(nn.Dense(c, activation='sigmoid', flatten=False))#全连接+bias
            self.add(nn.Dense(c1, activation='sigmoid', flatten=False))  # 全连接+bias
            self.add(nn.Dense(1, flatten=False))  # 全连接+bias


#####################################################
class MyC(nn.HybridBlock):
    def __init__(self, shapew,shapeb):
        super(MyC, self).__init__()

        with self.name_scope():
            self.b = self.params.get('b', shape=shapeb)
            self.w = self.params.get('w', shape=shapew)

    def hybrid_forward(self, F, x0, x1, b, w):
        y = F.broadcast_add(F.dot(
            F.batch_dot(x0, x1, False, True), w), b)
        #F.FullyConnected()
        return y

class MyE(nn.HybridBlock):
    def __init__(self, inC, outC):
        super(MyE, self).__init__()

        with self.name_scope():
            self.inC = inC
            self.outC = outC

            self.w = self.params.get(
                'weight',
                shape=(inC, outC),
                allow_deferred_init=True
            )

    def hybrid_forward(self, F, x, w):
        zw = F.concat(
            F.zeros((1, self.outC)), w, dim=0)
        return F.Embedding(x, w, self.inC, self.outC)

class MyEB(nn.HybridBlock):
    def __init__(self, dims, inC, outC):
        super(MyEB, self).__init__()

        self.d = dims#a list of fields' max length
        self.e = MyE(inC, outC)


    def onDone(self, F, result):
        return result

    def hybrid_forward(self, F, x):
        e = self.e(x)#e*v？
        result, start, end = [], 0, 0
        for i, size in enumerate(self.d):
            start, end = end, end + size
            sliced = F.slice_axis(e, 1, start, end)
            result.append(F.mean(sliced, 1, True))#av_pool

        return self.onDone(F, result)

class MyAUEB(nn.HybridBlock):
    def __init__(self, dims, inC, outC, User ,Ad):
        super(MyAUEB, self).__init__()

        self.d = dims#a list of fields' max length
        self.e = MyE(inC, outC)
        self.User = [int(x) for x in User]
        self.Ad = [int(x) for x in Ad]


    def onDone(self, F, result, User, Ad):
        for k in User:
            result[k] = User[k]
        for k in Ad:
            result[k] = Ad[k]
        keys = sorted(result.keys())
        res = []
        for k in keys:
            res.append(result[k])
        return res

    def hybrid_forward(self, F, x):
        e = self.e(x)  # e*v？
        result, User, Ad, start, end = {}, {}, {}, 0, 0
        for i, size in enumerate(self.d):
            start, end = end, end + size
            sliced = F.slice_axis(e, 1, start, end)
            if i in self.User:
                User[i] = sliced
            elif i in self.Ad:
                Ad[i] = sliced
            else:
                result[i] = sliced

        # print(User[1].infer_shape(data=(2048,590)))#B*field*K
        # print(Ad[0].infer_shape(data=(2048, 590)))
        # User = F.concat(*User, dim=1)#B*F*K
        # print(User.infer_shape(data=(2048, 590)))
        # Ad = F.concat(*Ad, dim=1)#B*1*K
        # print(Ad.infer_shape(data=(2048, 590)))
        return self.onDone(F, result, User, Ad)

class MyAttention(MyAUEB):
    def __init__(self, dims, inC, outC, User, Ad):
        super(MyAttention, self).__init__(dims, inC, outC, User, Ad)
        self.K = outC
        self.ip = MyIDP(80, 40)
        self.dims = dims

    # def onDone(self, F, result, User, Ad):
    #     #concat embedding
    #     for k in User:
    #         result[k] = User[k]
    #     for k in Ad:
    #         result[k] = Ad[k]
    #     keys = sorted(result.keys())
    #     res = []
    #     for k in keys:
    #         res.append(result[k])
    #     return result

    def hybrid_forward(self, F, x):
        e = self.e(x)  # e*v？
        result, User, Ad, start, end = {}, {}, {}, 0, 0
        for i, size in enumerate(self.d):
            start, end = end, end + size
            sliced = F.slice_axis(e, 1, start, end)
            if i in self.User:
                User[i] = sliced
            elif i in self.Ad:
                Ad[i] = sliced
            else:
                result[i] = sliced

        # print(User[1].infer_shape(data=(2048,590)))#B*field*K
        # print(Ad[0].infer_shape(data=(2048, 590)))
        # User = F.concat(*User, dim=1)#B*F*K
        # print(User.infer_shape(data=(2048, 590)))
        # Ad = F.concat(*Ad, dim=1)#B*1*K
        # print(Ad.infer_shape(data=(2048, 590)))
        #AT part
        AU = {}
        for k in User:#[B, F, K]
            queries = Ad[list(Ad.keys())[0]]
            shape = User[k].infer_shape(data=(2048, 590))[1][0]#(,,)
            shapeF = list(self.dims)[k]
            print(queries.infer_shape(data=(2048, 590)))
            queries = F.tile(queries, [1, shapeF])
            print(queries.infer_shape(data=(2048, 590)))
            queries = F.reshape(queries, [-1, shapeF, self.K])
            print(queries.infer_shape(data=(2048, 590)))
            din_all = F.concat(queries, User[k], queries - User[k], queries * User[k], dim=-1)
            print(din_all.infer_shape(data=(2048, 590)))
            din_all = self.ip(din_all)
            print(din_all.infer_shape(data=(2048, 590)))
            outputs = F.reshape(din_all, [-1, 1, shapeF])#[B, 1, F]
            print(outputs.infer_shape(data=(2048, 590)))
            key_masks = F.broadcast_greater(User[k], F.zeros_like(User[k]))
            paddings = F.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = F.where(key_masks, outputs, paddings)
            outputs = outputs / (shape[-1] ** 0.5)
            print(outputs.infer_shape(data=(2048, 590)))
            # Activation
            outputs = F.softmax(outputs)  # [B, 1, T]
            # Weighted sum
            outputs = F.batch_dot(outputs, User[k])  # [B, 1, H]
            print(outputs.infer_shape(data=(2048, 590)))
            AU[k] = outputs

        return self.onDone(F, result, AU, Ad)

class MyED(MyEB):#维度（各每行最大取值数），weight输入维度，输出维度
    def __init__(self, dims, inC, outC):
        super(MyED, self).__init__(dims, inC, outC)

    def onDone(self, F, result):
        return F.concat(*result, dim=1)

class MyER(nn.HybridBlock):
    def __init__(self, dims, inC, outC):
        super(MyER, self).__init__()

        self.d = dims
        self.e = MyE(inC, outC)

    def hybrid_forward(self, F, x):
        flag = autograd.is_recording()

        e = self.e(x)
        result, start, end = [], 0, 0
        for i, size in enumerate(self.d):
            start, end = end, end + size

            t1, t2 = start, end
            if flag and size > 5:
                t1, t2 = randomRange(start, end)

            sliced = F.slice_axis(e, 1, t1, t2)
            result.append(F.mean(sliced, 1, True))

        return F.concat(*result, dim=1)

class MyU(nn.HybridBlock):
    def __init__(self, inC, outC):
        super(MyU, self).__init__()

        with self.name_scope():
            self.inC = inC
            self.outC = outC

            self.w = self.params.get(
                'weight',
                shape=(inC-1, outC),
                allow_deferred_init=True,
                init=mx.init.Uniform(0.1),
            )

    def hybrid_forward(self, F, x, w):
        zw = F.concat(
            F.zeros((1, self.outC)), w, dim=0)
        return F.Embedding(x, zw, self.inC, self.outC)

class MyUD(nn.HybridBlock):
    def __init__(self, dims, inC, outC):
        super(MyUD, self).__init__()

        self.d = dims
        self.e = MyU(inC, outC)

    def hybrid_forward(self, F, x):
        e = self.e(x)
        result, start, end = [], 0, 0
        for i, size in enumerate(self.d):
            start, end = end, end + size
            sliced = F.slice_axis(e, 1, start, end)
            result.append(F.mean(sliced, 1, True))
        return F.concat(*result, dim=1)

class MyZ(nn.HybridBlock):
    def __init__(self, inC, outC):
        super(MyZ, self).__init__()

        with self.name_scope():
            self.inC = inC
            self.outC = outC

            self.w = self.params.get(
                'weight',
                shape=(inC, outC),
                allow_deferred_init=True
            )

    def hybrid_forward(self, F, x, w):
        return F.Embedding(x, w, self.inC, self.outC)

class MyZB(nn.HybridBlock):
    def __init__(self, dims, inC, outC):
        super(MyZB, self).__init__()

        self.d = dims
        self.e = MyZ(inC, outC)

    def onDone(self, F, result):
        return result

    def hybrid_forward(self, F, x):
        e = self.e(x)
        result, start, end = [], 0, 0
        for i, size in enumerate(self.d):
            start, end = end, end + size
            sliced = F.slice_axis(e, 1, start, end)
            result.append(F.mean(sliced, 1, True))

        return self.onDone(F, result)

###################################################
class MyDPO(nn.HybridSequential):
    def __init__(self, rate=.5):
        super(MyDPO, self).__init__()
        self.rate = rate

        with self.name_scope():
            self.add(nn.Dropout(.5))


class Chen_DFM(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit, User, Ad):
        super(Chen_DFM, self).__init__()

        with self.name_scope():
            self.ba = MyBA('relu')
            #self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyAttention(dims, inC, outC, User, Ad)
            self.lr = MyAttention(dims, inC, 1, User, Ad)
            self.dpo = MyDPO()
            self.n = len(dims)

    def hybrid_forward(self, F, x):
        # e = self.be(self.ed(x))
        #
        # lr = self.be(self.lr(x))
        e = self.ed(x)
        lr = self.lr(x)

        order1 = self.dpo(F.sum(lr, axis=2))

        order2_1 = F.square(F.sum(e, axis=1))
        order2_2 = F.sum(F.square(e), axis=1)
        order2 = self.dpo(0.5 * (order2_1 - order2_2))

        dp = self.dpo(F.flatten(e))
        # for i in range(0, 2):
        dp = self.dpo(self.ip(dp))
        # in_shape, out_shape, uax_shape = out.infer_shape(data=(256,207))
        # print(in_shape, out_shape, uax_shape)
        return self.ba(
            F.concat(dp, order1, order2, dim=1))

class Guo_DFFM(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(Guo_DFFM, self).__init__()

        with self.name_scope():
            self.n = n = len(dims)
            #self.be = MyBA('relu')
            self.ed = MyED(dims, inC, outC*n)

    def hybrid_forward(self, F, x):
        e = self.ed(x)
        #e = self.be(e)
        print(e.infer_shape(data=(2048, 590)))
        ss, es = [], F.split(e, self.n, 1)
        print(es.infer_shape(data=(2048, 590)))
        es1= F.split(es[0], self.n, 2)
        print(es1.infer_shape(data=(2048, 590)))
        for s in es:
            ss.append(F.split(s, self.n, 2))#[n*1*8,n*1*8,...]
###############################################################################
        import itertools
        embed_var_dict = {}
        for (i1, i2) in itertools.combinations(list(range(0, self.n)), 2):
            c1, c2 = i1, i2
            embed_var_dict.setdefault(c1, {})[c2] = ss[c1][i2]  # None * k
            embed_var_dict.setdefault(c2, {})[c1] = ss[c2][i1]  # None * k
        x_mat = []
        y_mat = []
        input_size = 0
        for (c1, c2) in itertools.combinations(embed_var_dict.keys(), 2):
            input_size += 1
            x_mat.append(embed_var_dict[c1][c2]) #input_size * None * k
            y_mat.append(embed_var_dict[c2][c1]) #input_size * None * k
        print(x_mat[0].infer_shape(data=(2048, 590)))
        x_mat = F.concat(*x_mat,dim=1)
        print(x_mat.infer_shape(data=(2048, 590)))
        y_mat = F.concat(*y_mat, dim=1)
        res = x_mat*y_mat
        print(res.infer_shape(data=(2048, 590)))
###########################################################
        # result = []
        # for i in range(self.n):
        #     for k in range(i, self.n):
        #         result.append(ss[i][k]*ss[k][i])#ele-wise
        #         #print(ss[i][k].infer_shape(data=(2048, 590)))
        # res = F.concat(*result, dim=1)
        # print(res.infer_shape(data=(2048, 590)))
        order2=F.sum(res, axis=1)
        print(order2.infer_shape(data=(2048, 590)))

        return order2
class MyDCN(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit, depth):
        super(MyDCN, self).__init__()

        with self.name_scope():
            self.be = MyBA('relu')
            self.ba = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyED(dims, inC, outC)

            self.depth = depth
            for i in range(depth):
                setattr(
                    self, 'MyC#%02d'%i,
                    MyC((len(dims)*outC, 1))
                )
                setattr(
                    self, 'MyBA#%02d'%i,
                    MyBA()
                )

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))
        deep = self.ip(F.flatten(e))

        y = e = F.expand_dims(F.flatten(e), -1)
        for i in range(self.depth):
            mc = getattr(self, 'MyC#%02d'%i)
            ba = getattr(self, 'MyBA#%02d'%i)

            y = ba(mc(e, y))
        cross = F.flatten(y)

        return self.ba(F.concat(deep, cross, dim=1))

class MyDFM(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFM, self).__init__()

        with self.name_scope():
            self.ba = MyBA('relu')
            self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyED(dims, inC, outC)


    def hybrid_forward(self, F, x):
        ed = self.ed(x)
        # in_shape, out_shape, uax_shape = ed.infer_shape(data=(2048,590))
        # print('inshape:',in_shape,'outshape:', out_shape)
        e = self.be(ed)
        in_shape, out_shape, uax_shape = e.infer_shape(data=(2048,590))
        print('inshape:',in_shape,'outshape:', out_shape)
        deep = self.ip(e)
        in_shape, out_shape, uax_shape = deep.infer_shape(data=(2048,590))
        print('inshape:',in_shape,'outshape:', out_shape)
        order1 = F.sum(e, axis=2)
        order2_1 = F.square(F.sum(e, axis=1))
        order2_2 = F.sum(F.square(e), axis=1)
        order2 = 0.5*(order2_1-order2_2)

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFM2(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFM2, self).__init__()

        with self.name_scope():
            self.n = len(dims)
            self.ba = MyBA('relu')
            self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyEB(dims, inC, outC)

    def hybrid_forward(self, F, x):
        x = self.ed(x)
        in_shape, out_shape, uax_shape = x[0].infer_shape(data=(2048,590))
        print('inshape:',in_shape,'outshape:', out_shape)
        y = F.concat(*x, dim=1)
        e = self.be(y)
        in_shape, out_shape, uax_shape = e.infer_shape(data=(2048,590))
        print('inshape:',in_shape,'outshape:', out_shape)
        deep = self.ip(e)
        order1 = F.sum(e, axis=2)

        # order2_1 = F.square(F.sum(e, axis=1))
        # order2_2 = F.sum(F.square(e), axis=1)
        # order2 = 0.5*(order2_1-order2_2)
        order2 = []
        for i in range(self.n):
            for k in range(i, self.n):
                order2.append(x[i]*x[k])
        order2 = F.flatten(F.concat(*order2, dim=1))

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFU(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFU, self).__init__()

        with self.name_scope():
            self.ba = MyBA('relu')
            self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyUD(dims, inC, outC)

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        deep = self.ip(e)
        order1 = F.sum(e, axis=2)
        order2_1 = F.square(F.sum(e, axis=1))
        order2_2 = F.sum(F.square(e), axis=1)
        order2 = 0.5*(order2_1-order2_2)

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFR(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFR, self).__init__()

        with self.name_scope():
            self.ba = MyBA('relu')
            self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyER(dims, inC, outC)

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        deep = self.ip(e)
        order1 = F.sum(e, axis=2)
        order2_1 = F.square(F.sum(e, axis=1))
        order2_2 = F.sum(F.square(e), axis=1)
        order2 = 0.5*(order2_1-order2_2)

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFZ(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFZ, self).__init__()

        with self.name_scope():
            self.n = len(dims)
            self.ba = MyBA('relu')
            self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyZB(dims, inC, outC)

    def hybrid_forward(self, F, x):
        x = self.ed(x)
        y = F.concat(*x, dim=1)

        e = self.be(y)
        deep = self.ip(e)
        order1 = F.sum(e, axis=2)

        order2 = []
        for i in range(self.n):
            for k in range(i+1, self.n):
                order2.append(x[i]*x[k])
        order2 = F.flatten(F.concat(*order2, dim=1))

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFCN(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFCN, self).__init__()

        with self.name_scope():
            self.n = n = len(dims)
            self.be = MyBA('relu')
            self.ba = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyED(dims, inC, outC*n)

            for i in range(n):
                layer = nn.HybridSequential()
                layer.add(MyBA('relu'))
                layer.add(MyCBA((n,3,1,1), 'relu'))
                layer.add(MyCBA((n,3,1,1), 'relu'))
                layer.add(MyCBA((n,3,1,1), 'relu'))
                setattr(
                    self, 'MyLayer#%02d'%i, layer
                )

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        deep = self.ip(e)
        order1 = F.sum(e, axis=2)

        ss, es = [], F.split(e, self.n, 1)
        for s in es: ss.append(F.split(s,self.n,2))

        order2 = []
        for i in range(self.n):
            temp = []
            for k in range(self.n):
                temp.append(ss[i][k]*ss[k][i])
            temp = F.concat(*temp, dim=1)
            temp = F.swapaxes(temp, 1, 2)

            layer = getattr(self, 'MyLayer#%02d'%i)
            temp = F.swapaxes(layer(temp), 1, 2)
            order2.append(F.sum(temp, axis=1))
        order2 = F.concat(*order2, dim=1)

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFCM(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFCM, self).__init__()

        with self.name_scope():
            self.n = len(dims)
            self.ba = MyBA('relu')
            self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyEB(dims, inC, outC)

    def hybrid_forward(self, F, x):
        es = self.ed(x)
        e = self.be(F.concat(*es, dim=1))

        deep = self.ip(e)
        order1 = F.sum(e, axis=2)

        order2 = []
        for i in range(self.n):
            for k in range(i+1, self.n):
                order2.append(F.batch_dot(
                    es[i], es[k], True, False))
        order2 = F.flatten(F.concat(*order2, dim=1))

        return self.ba(
            F.concat(deep, order1, order2, dim=1))



class MyDIN(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit, unit2=0):
        super(MyDIN, self).__init__()

        with self.name_scope():
            self.n = len(dims)
            self.ba = MyBA('relu')
            self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyEB(dims, inC, outC)

            unit2 = unit2 or outC
            for i in range(self.n):
                for k in range(i + 1, self.n):
                    setattr(
                        self, 'fc#%02d#%02d' % (i, k),
                        MyIBA(unit2, 'relu')
                    )

    def hybrid_forward(self, F, x):
        x = self.ed(x)
        y = F.concat(*x, dim=1)

        e = self.be(y)
        deep = self.ip(e)
        order1 = F.sum(e, axis=2)

        order2 = []
        for i in range(self.n):
            for k in range(i + 1, self.n):
                xi, xk = x[i], x[k]
                fi = F.concat(xi, xi - xk, xk, dim=1)
                f = getattr(self, 'fc#%02d#%02d' % (i, k))

                order2.append(f(fi))
        order2 = F.concat(*order2, dim=1)

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFIN(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFIN, self).__init__()

        with self.name_scope():
            self.n = n = len(dims)
            self.be = MyBA('relu')
            self.ba = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyED(dims, inC, outC*n)

            for i in range(n):
                for k in range(i+1, n):
                    setattr(
                        self, 'fc#%02d#%02d'%(i,k),
                        MyIBA(outC, 'relu')
                    )

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        deep = self.ip(e)
        order1 = F.sum(e, axis=2)

        ss, es = [], F.split(e, self.n, 1)
        for s in es: ss.append(F.split(s, self.n, 2))

        order2 = []
        for i in range(self.n):
            temp = []
            for k in range(i+1, self.n):
                eik, eki = ss[i][k], ss[k][i]
                fi = F.concat(eik, eik-eki, eki, dim=1)
                f = getattr(self, 'fc#%02d#%02d' % (i,k))

                temp.append(f(fi))
            if len(temp) == 0: continue
            order2.append(F.concat(*temp, dim=1))
        order2 = F.concat(*order2, dim=1)

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFFM(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFFM, self).__init__()

        with self.name_scope():
            self.n = n = len(dims)
            self.be = MyBA('relu')
            self.ed = MyED(dims, inC, outC*n)

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        ss, es = [], F.split(e, self.n, 1)

        for s in es:
            ss.append(F.split(s, self.n, 2))#[n*1*8,n*1*8,...]


        result = []
        for i in range(self.n):
            for k in range(i, self.n):
                result.append(ss[i][k]*ss[k][i])#ele-wise
                #print(te.infer_shape(data=(2048, 590)))

        res = F.concat(*result, dim=1)
        print(res.infer_shape(data=(2048, 590)))
        order2=F.sum(res, axis=1)
        print(order2.infer_shape(data=(2048, 590)))
        return order2

class MyDFFM2(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFFM2, self).__init__()

        with self.name_scope():
            self.n = n = len(dims)
            self.be = MyBA('relu')
            self.ed = MyED(dims, inC, outC*n)

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        ss, es = [], F.split(e, self.n, 1)
        for s in es:
            ss.append(F.split(s, self.n, 2))

        result = []
        for i in range(self.n):
            temp = []
            for k in range(self.n):
                temp.append(ss[i][k]*ss[k][i])
            temp = F.concat(*temp, dim=1)
            result.append(F.sum(temp, axis=1))

        return F.concat(*result, dim=1)

class MyDFFM3(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFFM3, self).__init__()

        with self.name_scope():
            self.n = n = len(dims)
            self.be = MyBA('relu')
            self.ba = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyED(dims, inC, outC*n)
            self.dpo = MyDPO()

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        deep = self.ip(e)
        order1 = F.sum(e, axis=2)

        ss, es = [], F.split(e, self.n, 1)
        for s in es: ss.append(F.split(s,self.n,2))

        order2 = []
        for i in range(self.n):
            temp = []
            for k in range(self.n):
                temp.append(ss[i][k]*ss[k][i])
            temp = F.concat(*temp, dim=1)
            order2.append(F.sum(temp, axis=1))
        order2 = F.concat(*order2, dim=1)

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

###################################################

class MyModel(object):
    def __init__(self, gpu=0):
        self.gpu = gpu

    def getNet(self, ctx):
        raise NotImplementedError

    def getModel(self):
        return './model'

    def getEpoch(self):
        return 1

    def getMetric(self):
        raise NotImplementedError

    def getTrainer(self, params, iters):
        opt = mx.optimizer.SGD(
            wd=1e-3,
            momentum=0.9,
            learning_rate=1e-2,
            lr_scheduler=mx.lr_scheduler.FactorScheduler(
                iters*30, 1e-1, 1e-3
            )
        )

        return mx.gluon.Trainer(params, opt)

    def forDebug(self, out):
        pass

    def train(self):
        model = self.getModel()
        if not os.path.exists(model):
            os.mkdir(model)

        ctx = mx.cpu(self.gpu)
        net, myLoss = self.getNet(ctx)
        trainI, testI = self.getData(ctx)
        metric, monitor = self.getMetric()
        trainer = self.getTrainer(
            net.collect_params(), trainI.iters)

        logging.info('')
        result, epochs = 0, self.getEpoch()

        for epoch in range(1, epochs+1):
            train_l_sum = mx.nd.array([0], ctx=ctx)
            logging.info('Epoch[%04d] start ...' % epoch)

            list(map(lambda x: x.reset(), [trainI, metric]))
            for batch_i, (data, label) in enumerate(trainI):
                with autograd.record():

                    out = net.forward(data)#2048,590
                    self.forDebug(out)
                    loss = myLoss(out, label)
                loss.backward()
                # grads = [p.grad(ctx) for p in net.collect_params().values()]
                # mx.gluon.utils.clip_global_norm(
                #     grads, .2 * 5 * data.shape[0])
                trainer.step(data.shape[0])#trainer.step(batch_size)
                print('train loss:', loss.mean().asnumpy())
                ###############################################
                # train_l_sum += loss.sum() / data.shape[0]
                # eval_period = 1
                # if batch_i % eval_period == 0 and batch_i > 0:
                #     cur_l = train_l_sum / eval_period
                #     print('epoch %d, batch %d, train loss %.6f, perplexity %.2f'
                #           % (epoch, batch_i, cur_l.asscalar(),
                #              cur_l.exp().asscalar()))
                #     train_l_sum = mx.nd.array([0], ctx=ctx)
                #############################################
                metric.update(label, out)
            for name, value in metric.get():
                logging.info('Epoch[%04d] Train-%s=%f ...', epoch, name, value)

            _result = None
            list(map(lambda x: x.reset(), [testI, metric]))
            l_sum = mx.nd.array([0], ctx=ctx)
            n = 0
            for data, label in testI:
                out = net.forward(data)
                l = myLoss(out, label)
                l_sum += l.sum()
                n += l.size
                self.forDebug(out)
                metric.update(label, out)
            print('valid loss:', (l_sum / n).mean().asnumpy())
            for name, value in metric.get():
                if name == monitor : _result = value
                logging.info('Epoch[%04d] Validation-%s=%f', epoch, name, value)

            if _result > result:
                result = _result
                name = '%s/%04d-%3.3f%%.params' % (model, epoch, result*100)
                net.save_params(name)
                logging.info('Save params to %s ...', name)

            logging.info('Epoch[%04d] done ...\n' % epoch)

class MyPredict(object):
    def __init__(self, gpu=0):
        self.gpu = gpu

    def getNet(self, ctx):
        raise NotImplementedError

    def preProcess(self, data):
        return data

    def postProcess(self, data, pData, output):
        raise NotImplementedError

    def onDone(self):
        pass

    def predict(self):
        ctx = mx.cpu(self.gpu)

        net = self.getNet(ctx)
        testI = self.getData(ctx)

        logging.info('Start predicting ...')

        testI.reset()
        for data in testI:
            pData = self.preProcess(data)
            out = net.forward(pData)
            self.postProcess(data, pData, out)

        self.onDone()
        logging.info('Prediction done ...')

###################################################

class MyLoss(nn.HybridBlock):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.loss = loss.SigmoidBCELoss()#l2

    def hybrid_forward(self, F, pred, label):
        sampleWeight = 20*label+(1-label)
        return self.loss(pred, label)

class MyMetric(object):
    def get(self):
        assert len(self.labels) != 0, 'Failed ...'
        assert len(self.outputs) != 0, 'Failed ...'

        labels = np.hstack(self.labels)
        outputs = np.hstack(self.outputs)
        results = (
            (
                'aucMetric',
                self.myAuc(labels, outputs)
            ),
            (
                'accMetric',
                self.myAcc(labels, outputs)
            ),
        )

        return results

    def myAuc(self, labels, outputs):
        return roc_auc_score(labels, outputs)

    def myAcc(self, labels, outputs):
        return np.mean(labels == (outputs>=0.5))

    def reset(self):
        self.labels = []
        self.outputs = []

    def update(self, label, output):
        label = label.asnumpy()
        output = mx.nd.sigmoid(output)
        output = output.asnumpy()[:,0]

        self.labels.append(label)
        self.outputs.append(output)

###################################################

class MyLoss2(nn.HybridBlock):
    def __init__(self):
        super(MyLoss2, self).__init__()
        self.loss = loss.SoftmaxCELoss()

    def hybrid_forward(self, F, pred, label):
        sampleWeight = 20*label+(1-label)
        return self.loss(pred, label, sampleWeight)

class MyMetric2(object):
    def get(self):
        assert len(self.labels) != 0, 'Failed ...'
        assert len(self.outputs) != 0, 'Failed ...'

        labels = np.hstack(self.labels)
        outputs = np.hstack(self.outputs)
        results = (
            (
                'aucMetric',
                self.myAuc(labels, outputs)
            ),
            (
                'accMetric',
                self.myAcc(labels, outputs)
            ),
        )

        return results

    def myAuc(self, labels, outputs):
        return roc_auc_score(labels, outputs)

    def myAcc(self, labels, outputs):
        return np.mean(labels == (outputs>=0.5))

    def reset(self):
        self.labels = []
        self.outputs = []

    def update(self, label, output):
        label = label.asnumpy()
        output = mx.nd.softmax(output)
        output = output.asnumpy()[:,1]

        self.labels.append(label)
        self.outputs.append(output)

###################################################

class MyLoss3(nn.HybridBlock):
    def __init__(self):
        super(MyLoss3, self).__init__()
        self.loss1 = loss.SoftmaxCELoss()
        self.loss2 = loss.SigmoidBCELoss()

    def hybrid_forward(self, F, pred, label):
        sampleWeight = 20*label+(1-label)

        pred1 = F.slice_axis(pred, 1, 0, 2)
        pred2 = F.slice_axis(pred, 1, 2, 3)

        loss1 = self.loss1(pred1, label, sampleWeight)
        loss2 = self.loss2(pred2, label, sampleWeight)

        return 0.5 * (loss1 + loss2)

class MyMetric3(object):
    def get(self):
        assert len(self.labels) != 0, 'Failed ...'
        assert len(self.outputs) != 0, 'Failed ...'

        labels = np.hstack(self.labels)
        outputs = np.hstack(self.outputs)
        results = (
            (
                'aucMetric',
                self.myAuc(labels, outputs)
            ),
            (
                'accMetric',
                self.myAcc(labels, outputs)
            ),
        )

        return results

    def myAuc(self, labels, outputs):
        return roc_auc_score(labels, outputs)

    def myAcc(self, labels, outputs):
        return np.mean(labels == (outputs>=0.5))

    def reset(self):
        self.labels = []
        self.outputs = []

    def update(self, label, output):
        label = label.asnumpy()

        output1 = output[:, 0:2]
        output2 = output[:, 2:3]

        output1 = mx.nd.softmax(output1)
        output1 = output1.asnumpy()[:,1]

        output2 = mx.nd.sigmoid(output2)
        output2 = output2.asnumpy()[:,0]

        output = 0.5*(output1 + output2)

        self.labels.append(label)
        self.outputs.append(output)

###################################################

