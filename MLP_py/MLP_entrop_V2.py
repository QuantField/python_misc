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

Version 2 : the training is done properly, the tuning is done as 
            to lower the erro on a validation/test set

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

n_tr = int(0.7*n)  # split 70/30


X = train.loc[:,1:m-2].as_matrix()
Y = train.loc[:,m-1].as_matrix().reshape(-1,1)
Y[(Y==4)]=1
Y[(Y==2)]=0

Xtr, Ytr = X[:n_tr,:], Y[:n_tr]
Xte, Yte = X[n_tr:,:], Y[n_tr:]

 

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

def entropyX(y, yhat):
    D = -(y*np.log2(yhat)+(1-y)*np.log2(1-yhat))
    return float(sum(D))/len(D)

def score(x):
    o1    = L1.feedData(x)
    o2    = L2.feedData(o1)
    return o2


np.random.seed(3148)     
    
n_nodes = 20             
in_dim = X.shape[1]

L1    = layer(n_nodes ,in_dim)
L2    = layer(1, n_nodes)


eta = 0.001

iters, sErr = 1000, np.Infinity
TrEnt       = []
TeEnt       = []
for i in range(iters):          
    #np.random.shuffle(X)
    for obs in range(Xtr.shape[0]):
        o0 , y = X[obs,:], float(Ytr[obs])
        o1 = L1.feed(o0)
        o2 = float(L2.feed(o1))
       
        if o2>0.99999: o2=0.99999;
        if o2<0.00001: o2=0.00001;
        
        dEntr_dO2 = -(y/o2 - (1-y)/(1-o2))
        
        delta2 = dEntr_dO2*o2*(1-o2)
        L2.w  += -eta*delta2*o1.reshape(1,-1)
            
        delta1 = delta2*o1*(1-o1)    
        L1.w += -eta*delta1*o0
    
    train_err = entropyX(Ytr,score(Xtr))
    test_err  = entropyX(Yte,score(Xte))
    sErr0     = sErr    
    sErr      = test_err  
    if (sErr-sErr0)/sErr0>1e-6: 
       break    
    print("epoch = ",i," Validation =",round(sErr,4)," Training =", 
                                                   round(train_err,4) )
    TrEnt.append(train_err)    
    TeEnt.append(test_err)    


pred  = score(Xtr) 
pred[pred<=0.5]=0
pred[pred>0.5]=1
conf = confusion_matrix(Ytr, pred) 
confMat = conf[0], conf[1]    
[TN, FN], [FP, TP] = confMat
fpr, tpr, thresholds = sk.roc_curve(Ytr, pred)
auroc = sk.auc(fpr, tpr)


import  matplotlib.pyplot as plt



plt.plot(TrEnt,label='Train')
plt.plot(TeEnt, label='Validation')
plt.xlabel('iterations')
plt.legend(loc='upper right')  
plt.title('Entropy')
plt.grid()
plt.show()











        