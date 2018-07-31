# -*- coding: utf-8 -*-

import xlearn as xl
import pandas as pd

ffm_model=xl.create_ffm()
#ffm_model=xl.setTrain("ffm_train.txt")
#ffm_model.setValidate("ffm_val.txt")
ffm_model.setTrain("./data/train.ffm")
ffm_model.setTest("./data/test.ffm")
ffm_model.setSigmoid()
# ffm_model.setSign()

param={'task':'binary','lr':0.05,'lambda':0.00002,'metric':'auc','k':8,'epoch':22,'stop_window':3}

ffm_model.fit(param,"./data/xlearn_result/model.out")
ffm_model.predict("./data/xlearn_result/model.out","./data/xlearn_result/output.txt")


#ffm_model.cv(param)
#ffm_model.fit(param,"ffm_result.txt")

result=pd.read_table("./data/xlearn_result/output.txt",header=None)
test=pd.read_csv("./data/test1.csv")
test['score']=result[0]
test.to_csv('./data/xlearn_result/submission_xlearn.csv',index=False)