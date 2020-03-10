# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:50:48 2017

@author: saadik1
"""

import pandas as pd
from explore import *

steps = processSteps('models2017-Jan-12_141407') 
steps.load()

sasLibname = "Y:\\racmi\\data\\analysis\\saadik\\"

BiData = pd.read_sas( sasLibname +"data_test_frd.sas7bdat", encoding='utf-8')

transToCategorical(data=BiData)

steps.imputeObj.imputeNew(BiData)
steps.standarizeObj.standarizeNew(BiData)





xx = toNumeric(BiData) 
scoreData = xx[steps.variables.index]
print("----- Scoring one model ------")
models = steps.models
#yscore = models[0].predict(scoreData)

#---- some stats ---------------
#---getting fraud flags 
frauds = pd.read_sas( sasLibname + "data_fr.sas7bdat", encoding='utf-8')
frauds['fraud_flag']=1.0
#------ merging data
BiData.rename(columns={'Claim_No': 'claim_no'}, inplace=True)
allData = pd.merge(BiData, frauds, how='left', on='claim_no')
allData['fraud_flag'].fillna(0.0,inplace = True)
freq(data=allData, var ='fraud_flag',graph =True)
frdflag = allData['fraud_flag']


import diagnostics as diag

mod = models[0]
yscore = mod.predict(scoreData)
diag.classAccuracy(frdflag.as_matrix(), yscore)
predprb = mod.predict_proba(scoreData)
auc1, fpr1,tpr1 = diag.AUROC(frdflag.as_matrix(),predprb[:,1])
print("AUC = %2.4f "% auc1)
#diag.plotROC(frdflag.as_matrix(),predprb[:,1])

mod = models[1]
yscore = mod.predict(scoreData)
diag.classAccuracy(frdflag.as_matrix(), yscore)
predprb = mod.predict_proba(scoreData)
auc2, fpr2,tpr2 = diag.AUROC(frdflag.as_matrix(),predprb[:,1])
print("AUC = %2.4f "% auc2)
#diag.plotROC(frdflag.as_matrix(),predprb[:,1])

mod = models[2]
yscore = mod.predict(scoreData)
diag.classAccuracy(frdflag.as_matrix(), yscore)
predprb = mod.predict_proba(scoreData)
auc3, fpr3,tpr3 = diag.AUROC(frdflag.as_matrix(),predprb[:,1])
print("AUC = %2.4f "% auc3)
#diag.plotROC(frdflag.as_matrix(),predprb[:,1])

mod = models[3]
yscore = mod.predict(scoreData)
diag.classAccuracy(frdflag.as_matrix(), yscore)
predprb = mod.predict_proba(scoreData)
auc4, fpr4,tpr4 = diag.AUROC(frdflag.as_matrix(),predprb[:,1])
print("AUC = %2.4f "% auc4)
#diag.plotROC(frdflag.as_matrix(),predprb[:,1])


mod = models[4]
yscore = mod.predict(scoreData)
diag.classAccuracy(frdflag.as_matrix(), yscore)
predprb = mod.predict_proba(scoreData)
auc5, fpr5,tpr5 = diag.AUROC(frdflag.as_matrix(),predprb[:,1])
print("AUC = %2.4f "% auc5)
#diag.plotROC(frdflag.as_matrix(),predprb[:,1])


mod = models[5]
yscore = mod.predict(scoreData)
diag.classAccuracy(frdflag.as_matrix(), yscore)
predprb = mod.predict_proba(scoreData)
auc6, fpr6,tpr6 = diag.AUROC(frdflag.as_matrix(),predprb[:,1])
print("AUC = %2.4f "% auc6)
#diag.plotROC(frdflag.as_matrix(),predprb[:,1])





import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.plot(fpr1,tpr1,'r',label=steps.modelNames[0] + ' :'+str(auc1)[0:5])
plt.plot(fpr2,tpr2,'b',label=steps.modelNames[1] + ' :'+str(auc2)[0:5])
plt.plot(fpr3,tpr3,'k',label=steps.modelNames[2] + ' :'+str(auc3)[0:5])
plt.plot(fpr4,tpr4,'m',label=steps.modelNames[3] + ' :'+str(auc4)[0:5])
plt.plot(fpr5,tpr5,'g',label=steps.modelNames[4] + ' :'+str(auc5)[0:5])
plt.plot(fpr6,tpr6,'c',label=steps.modelNames[5] + ' :'+str(auc6)[0:5])
plt.grid()
plt.xlabel('False positives')
plt.ylabel('True positives')
plt.legend(loc='lower right')
