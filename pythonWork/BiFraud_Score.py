# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:50:48 2017

@author: saadik1
"""

import pandas as pd
from explore import *

steps = processSteps('models2017-Jan-19_115724') 
steps.load()

sasLibname = "Y:\\racmi\\data\\analysis\\saadik\\"

BiData = pd.read_sas( sasLibname +"data_test_frd.sas7bdat", encoding='utf-8')

transToCategorical( data=BiData, nLevels = steps.nLevels )

steps.imputeObj.imputeNew(BiData)
steps.standarizeObj.standarizeNew(BiData)


xx = toNumeric(BiData) 
scoreData = xx[steps.variables.index]
print("----- Scoring one model ------")
#models = steps.models
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

scoreDiag=[]
scoreRoc =[]

prob =[]
for mod in steps.models:
    i = steps.models.index(mod)
    print("------------> " + steps.modelNames[i] )
    #---------------------- training --------------------
    yhat    = mod.predict(scoreData)
    predprb = mod.predict_proba(scoreData)
    prob.append(predprb[:,1])
    acc, sens, prec = diag.classAccuracy(frdflag.as_matrix(), yhat)
    auc, fpr,tpr    = diag.AUROC(frdflag.as_matrix(),predprb[:,1])
    scoreDiag.append([auc,acc, sens, prec])
    scoreRoc.append([auc,fpr,tpr])
    print(" SCORE : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))



def rocPic(data, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    style = ['b','r','g','k','m','c','b--','r--','k--','m--']
    for i in range(len(steps.models)):
        auc,fpr,tpr = data[i]
        legend = steps.modelNames[i]+ ' :'+str(auc)[0:5]
        plt.plot(fpr,tpr,style[i],label=legend)
    plt.grid()    
    plt.title(title)
    plt.xlabel('False positives')
    plt.ylabel('True positives')
    plt.legend(loc='lower right')  
    plt.savefig(steps.folder+'/score_roc')
    
    
rocPic(scoreRoc,"Score")


oveProb = np.array(prob)
oveProb = oveProb.transpose()
scoreProb = oveProb.mean(axis=1)
scoreFlag = np.array([ int(r>0.60) for r in  scoreProb])


acc, sens, prec = diag.classAccuracy(frdflag.as_matrix(), scoreFlag)
auc, fpr,tpr    = diag.AUROC(frdflag.as_matrix(),scoreProb)
print(" SCORE : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))





#import matplotlib.pyplot as plt
#plt.figure(figsize=(8, 8))
#plt.plot(fpr1,tpr1,'r',label=steps.modelNames[0] + ' :'+str(auc1)[0:5])
#plt.plot(fpr2,tpr2,'b',label=steps.modelNames[1] + ' :'+str(auc2)[0:5])
#plt.plot(fpr3,tpr3,'k',label=steps.modelNames[2] + ' :'+str(auc3)[0:5])
#plt.plot(fpr4,tpr4,'m',label=steps.modelNames[3] + ' :'+str(auc4)[0:5])
#plt.plot(fpr5,tpr5,'g',label=steps.modelNames[4] + ' :'+str(auc5)[0:5])
#plt.plot(fpr6,tpr6,'c',label=steps.modelNames[5] + ' :'+str(auc6)[0:5])
#plt.grid()
#plt.xlabel('False positives')
#plt.ylabel('True positives')
#plt.legend(loc='lower right')
