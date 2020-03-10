# -*- coding: utf-8 -*-
"""
- Artificial Neural Network with 1 hidden Layer
- Cost Function : Cross Entropy
- Activation Function : Sigmoid (Logistic)

- Data : Wisconsin Breat Cancer

https://archive.ics.uci.edu/ml/machine-learning-databases/
       breast-cancer-wisconsin/
       
Data can be downloaded from :       
https://archive.ics.uci.edu/ml/machine-learning-databases/
breast-cancer-wisconsin/breast-cancer-wisconsin.data     

Target in {0,1}    

Sample code number            id number
Clump Thickness               1 - 10
Uniformity of Cell Size       1 - 10
Uniformity of Cell Shape      1 - 10
Marginal Adhesion             1 - 10
Single Epithelial Cell Size   1 - 10
Bare Nuclei                   1 - 10
Bland Chromatin               1 - 10
Normal Nucleoli               1 - 10
Mitoses                       1 - 10
Class:                        (2 for benign, 4 for malignant)

Author : Dr Kamel Saadi
Date   : 07/12/2017
"""

import numpy as np
import pandas as pd
from sklearn.metrics import  confusion_matrix 
import sklearn.metrics as sk


train = pd.read_csv("C:\\KS_temp\\breast_cancer_data.csv", header=None)
train.loc[(train.loc[:,6]=='?'),6]=1 # replase ? with 1, just a guess
train.loc[:,6]= train.loc[:,6].astype(float)
n,m = train.shape
X = train.loc[:,1:m-2].as_matrix()
Y = train.loc[:,m-1].as_matrix().reshape(-1,1)
Y[(Y==4)]=1
Y[(Y==2)]=0

  

class layer:
    def __init__(self, n_nodes, n_inputs):
        self.n_nodes  = n_nodes
        self.n_inputs = n_inputs
        self.w = 2*np.random.random([n_nodes, n_inputs])-1
    def feed(self,x):
        return  1.0/(1.0+np.exp(-np.dot(self.w,x.reshape(-1,1))))
    def feedData(self,X):
        d = np.dot(X,self.w.T)
        return 1.0/(1.0+np.exp(-d))
    

np.random.seed(3148)     
    
n_nodes = 20             
in_dim = X.shape[1]

L1    = layer(n_nodes ,in_dim)
L2    = layer(1, n_nodes)

eta = 0.0001

iters, SSE = 1000,np.Infinity
crossEntroy = [] 
yErr        = []
for i in range(iters):    
    err  = [] 
    err2 = []
    #np.random.shuffle(X)
    for obs in range(X.shape[0]):
        o0 , y = X[obs,:], float(Y[obs])
        o1 = L1.feed(o0)
        o2 = float(L2.feed(o1))
        
        
        if o2>0.99999: o2=0.99999;
        if o2<0.00001: o2=0.00001;
        
        dy  = o2-y
        
        entropy   = -(y*np.log2(o2)+(1-y)*np.log2(1-o2))
        dEntr_dO2 = -(y/o2 - (1-y)/(1-o2))
        
        err.append(entropy)
        err2.append(dy)               
        
        delta2 = dEntr_dO2*o2*(1-o2)
        L2.w  += -eta*delta2*o1.reshape(1,-1)
            
        delta1 = delta2*o1*(1-o1)    
        L1.w += -eta*delta1*o0
    SSE0 = SSE    
    SSE  = sum(err)
    yErr.append(np.dot(err2,err2))
    crossEntroy.append(SSE)    
    if SSE>SSE0: 
        break    
    print("epoch = ",i," SSE =",SSE)
    

O1    = L1.feedData(X)
O2    = L2.feedData(O1)
pred  = O2.copy()
pred[pred<=0.5]=0
pred[pred>0.5]=1
    


conf = confusion_matrix(Y, pred) 
confMat = conf[0], conf[1]    
[TN, FN], [FP, TP] = confMat

fpr, tpr, thresholds = sk.roc_curve(Y, pred)
auroc = sk.auc(fpr, tpr)


import  matplotlib.pyplot as plt

coef = max(crossEntroy)/max(yErr)

tt = [coef*t for t in   yErr]
plt.plot(crossEntroy,label='Cross Entropy')
plt.plot(tt, label='Scaled SSE')
plt.xlabel('iterations')
plt.legend(loc='upper right')  
plt.title('Training set')
plt.grid()
plt.show()






        