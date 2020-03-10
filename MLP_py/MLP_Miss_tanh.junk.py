# -*- coding: utf-8 -*-
"""

- Artificial Neural Network with 1 hidden Layer
- Cost Function : misclassification (1-yf)
- Activation Function : Tanh

- Data : Wisconsin Breat Cancer

https://archive.ics.uci.edu/ml/machine-learning-databases/
       breast-cancer-wisconsin/
       
Data can be downloaded from :       
https://archive.ics.uci.edu/ml/machine-learning-databases/
breast-cancer-wisconsin/breast-cancer-wisconsin.data     

Target in {-1,1}  

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


train = pd.read_csv("C:\\Users\\ks_work\\Desktop\\Code\\breast_cancer_data.csv", header=None)
train.loc[(train.loc[:,6]=='?'),6]=1 # replase ? with 1, just a guess
train.loc[:,6]= train.loc[:,6].astype(float)
n,m = train.shape
X = train.loc[:,1:m-2].as_matrix()
Y = train.loc[:,m-1].as_matrix().reshape(-1,1)
Y[(Y==4)] =1.0
Y[(Y==2)] =-1.0


class layer:
    def __init__(self, n_nodes, n_inputs):
        self.n_nodes  = n_nodes
        self.n_inputs = n_inputs
        self.w = 2*np.random.random([n_nodes, n_inputs])-1
    def feed(self,x):
        return  np.tanh(np.dot(self.w,x.reshape(-1,1)))
    def feedData(self,X):
        d = np.dot(X,self.w.T)
        return np.tanh(d)
    
eta  = 0.001    
n_nodes = 20             
in_dim = X.shape[1]
np.random.seed(3148) 
L1    = layer(n_nodes ,in_dim)
L2    = layer(1, n_nodes)
iters = 1000
SSE = np.Infinity
for i in range(iters):    
    err  = [] 
    for obs in range(X.shape[0]):
        o0 = X[obs,:]
        o1 = L1.feed(o0)
        o2 = float(L2.feed(o1))
        
        y = float(Y[obs])
        dE_df = -y*(1-y*o2)
        
        err.append(1-y*o2)
        delta2 = -y*(1-y*o2)*(1-o2*o2)
        L2.w  += -eta*delta2*o1.reshape(1,-1)
            
        delta1 = delta2*(1-o1*o1)    
        L1.w += -eta*delta1*o0
    SSE0 = SSE    
    SSE  = np.dot(err,err)    
    if SSE>SSE0: 
        break    
    print("epoch = ",i," SSE =",SSE)
    

O1    = L1.feedData(X)
O2    = L2.feedData(O1)
pred  = O2.copy()
pred[pred<0]=-1
pred[pred>0]=1
    
print("train acc =",float(sum(np.not_equal(pred,Y))/len(Y)))
    
    

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix 
conf = confusion_matrix(Y, pred) 
confMat = conf[0], conf[1]    
[TN, FN], [FP, TP] = confMat


import sklearn.metrics as sk
fpr, tpr, thresholds = sk.roc_curve(Y, pred)
auroc = sk.auc(fpr, tpr)













        