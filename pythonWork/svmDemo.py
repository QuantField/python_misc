# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 09:49:00 2016

@author: saadik1
"""


import diagnostics as diag

import pandas as pd


allData = pd.read_csv("C:\\KS_temp\\banana_data1000.csv")


trainData = allData.sample(frac=0.7, random_state = 564)        
testData  = allData.loc[~allData.index.isin(trainData.index)] 


y = trainData['Class']
X = trainData[['X','Y']]


yt = testData['Class']
Xt = testData[['X','Y']]




#-----------------------------------------------------------------
import sklearn.discriminant_analysis as da
import sklearn.linear_model as lm
from sklearn.svm import SVC

lda    = da.LinearDiscriminantAnalysis()
logreg = lm.LogisticRegression()
qda    = da.QuadraticDiscriminantAnalysis()
svm = SVC(kernel='linear',probability=True)



model1 = lda.fit(X,y)
yhat = model1.predict(X)
p = diag.svmTargetMisc(y,yhat)
pred1 = model1.predict(Xt)
p1 = diag.svmTargetMisc(yt,pred1)
print("LDA : Train = %2.2f   Test = %2.2f"%(p,p1))



model2 = logreg.fit(X,y)
yhat = model2.predict(X)
p = diag.svmTargetMisc(y,yhat)
pred1 = model2.predict(Xt)
p1 = diag.svmTargetMisc(yt,pred1)
print("LOGISTIC : Train = %2.2f   Test = %2.2f"%(p,p1))


model3 = qda.fit(X,y)
yhat = model3.predict(X)
p = diag.svmTargetMisc(y,yhat)
pred1 = model3.predict(Xt)
p1 = diag.svmTargetMisc(yt,pred1)
print("QDA : Train = %2.2f   Test = %2.2f"%(p,p1))


svm.fit(X, y ) 
pred1 = svm.predict_proba(X)
pred2 = svm.predict_proba(Xt)[:,1]






import matplotlib.pyplot as plt

lda_pred = model1.predict_proba(Xt)
auc,fpr,tpr = diag.AUROC(yt,lda_pred[:,1])
#
lreg_pred = model2.predict_proba(Xt)
auc2,fpr2,tpr2 = diag.AUROC(yt,lreg_pred[:,1])
#
qda_pred = model3.predict_proba(Xt)
auc3,fpr3,tpr3 = diag.AUROC(yt,qda_pred[:,1])
#
plt.plot(fpr,tpr,'b',label='LDA : '+str(auc)[0:5])
plt.plot(fpr2,tpr2,'r',label='Logistic : '+str(auc2)[0:5])
plt.plot(fpr3,tpr3,'c',label='QDA : '+str(auc3)[0:5])
plt.legend()
plt.title("ROC Test set")
print("----- AUROC TEST SET-------")
print("LDA : %2.4f    LOGREG : %2.4f   QDA : %2.4f"%(auc,auc2,auc3) )

#------------------------------------------------------------------










res = diag.

predt = clf.predict(Xt)



res2 = diag.classAccuracy(yt, predt, disp=True)

import numpy as np

cmp = np.concatenate((y.as_matrix(),pred),axis=1)




