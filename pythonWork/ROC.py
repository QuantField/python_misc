# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:39:53 2016

@author: saadik1
"""
import pandas as pd

s = pd.read_excel("C:\\KS_temp\\rocVal.xlsx")
auc,fpr,tpr = diag.AUROC(s['fraud'],s['Prob']  )

import matplotlib.pyplot as plt
import numpy as np
n = 1000
x = np.random.rand(n)
randClass = np.apply_along_axis(lambda w : w>=0.5,0,x)
randProb  = np.random.rand(n)
auc2,fpr2,tpr2 = diag.AUROC(randClass,randProb  )

plt.figure(figsize=(8, 8))
plt.plot(fpr,tpr,'r',label='Model AUROC : '+str(auc)[0:5])
plt.plot(fpr2,tpr2,'b',label='Baseline AUROC : '+str(auc2)[0:5])
plt.grid()
plt.xlabel('False positives')
plt.ylabel('True positives')
plt.legend(loc='upper left')