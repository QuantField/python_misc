# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:00:45 2016

@author: saadik1
"""

import numpy as np
import sklearn.metrics as sk
import matplotlib.pyplot as plt
import numpy as np


def R_Sqr(act, pred): 
    return  sk.r2_score(act, pred)
        
def R_Sqr_Ajust(act, pred, model_df): 
    n = len(act)
    aj = model_df/(n-model_df-1)
    R2 = sk.r2_score(act, pred)
    return  R2 -(1-R2)*aj

def AUROC(act, pred):
    fpr, tpr, thresholds = sk.roc_curve(act, pred)
    return (sk.auc(fpr, tpr),fpr,tpr)

def svmTargetMisc(act,pred):
    # assumes target are binary (-1,+1)
    return (np.sign(pred)!=act).sum()/len(act)

def classAccuracy(act, pred):
    P  = max(act)
    N  = min(act)
    TP = np.logical_and(pred==P, act==P).sum()
    FP = np.logical_and(pred==P, act==N).sum()
    TN = np.logical_and(pred==N, act==N).sum()
    FN = np.logical_and(pred==N, act==P).sum()
    ACC  = (TP+TN)/len(act)
    SENS = TP/(TP+FN)
    PREC  = TP/(TP+FP)
    #print("ACC=  SENS= PREC= ", ACC,SENS,PREC)    
    #print ("ACC=%2.4f  SENS=%2.4f PREC= %2.4f"%(ACC,SENS,PREC))
    return (ACC,SENS,PREC)
        
    
    
def AUROC(act, pred):
    fpr, tpr, thresholds = sk.roc_curve(act, pred)
    return (sk.auc(fpr, tpr),fpr,tpr)    
    
def plotROC(act, pred):
    auc, fpr, tpr = AUROC(act, pred)
    n = len(act)
    x = np.random.rand(n)
    randClass = np.apply_along_axis(lambda w:w>=0.5, 0, x)
    randProb  = np.random.rand(n)
    auc2,fpr2,tpr2 = AUROC(randClass,randProb  )
    #-------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.plot(fpr,tpr,'r',label='Model AUROC : '+str(auc)[0:5])
    plt.plot(fpr2,tpr2,'b',label='Baseline AUROC : '+str(auc2)[0:5])
    plt.grid()
    plt.xlabel('False positives')
    plt.ylabel('True positives')
    plt.legend(loc='lower right')
    
    
    
    
    
    
    