# -*- coding: utf-8 -*-
"""

simple one hidden layer nnet

This version is good

Dr Kamel Saadi
05/12/2017


"""

import numpy as np
import pandas as pd



np.random.seed(2340)
path  = "C:\\myData\\banana\\"
train = pd.read_csv(path+"banana_train_1.csv", header = None)
n,m = train.shape
X = train.loc[:,:m-2].as_matrix()
Y = train.loc[:,m-1].as_matrix().reshape(-1,1)
Y[(Y==-1)]=0




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
    
n_nodes = 20             
in_dim = X.shape[1]

L1    = layer(n_nodes ,in_dim)
L2    = layer(1, n_nodes)
iters, SSE = 1000,  np.Infinity

err = np.zeros(n)

eta = 0.01
for i in range(iters):  
    for obs in range(X.shape[0]):
        o0 = X[obs,:]
        o1 = L1.feed(o0)
        o2 = float(L2.feed(o1))
        
        y = float(Y[obs])
        err[obs]=(y-o2)
        
        delta2 = -(y-o2)*o2*(1-o2)
        L2.w  += -eta*delta2*o1.reshape(1,-1)
            
        delta1 = delta2*o1*(1-o1)    
        L1.w += -eta*delta1*o0
    SSE0 = SSE    
    SSE  = np.dot(err,err)    
    if SSE>SSE0: 
        break    
    print("epoch = ",i," SSE =",SSE)
    

O1    = L1.feedData(X)
O2    = L2.feedData(O1)
pred  = O2
pred[pred<=0.5]=0
pred[pred>0.5]=1
ll=np.not_equal(Y,pred)
print("\ntrain error = ", sum(ll)/len(ll))

   

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix 
conf = confusion_matrix(Y, pred) 
confMat = conf[0], conf[1]    
[TN, FN], [FP, TP] = confMat


import sklearn.metrics as sk
fpr, tpr, thresholds = sk.roc_curve(Y, pred)
auroc = sk.auc(fpr, tpr)



#------------------testing

te = pd.read_csv(path+"banana_test_1.csv", header = None)

nfeat = te.shape[1]-1
# for this code to work faster we need to transform to np.arrays
Xt = te.loc[:,0:nfeat-1].as_matrix() 

Yt = te.loc[:,nfeat].as_matrix()

# for this code to work faster we need to transform to np.arrays
#X = train.loc[:,0:1].as_matrix() 

#Y = train.loc[:,2].as_matrix()

Yt[(Yt<0)]=0.0

O1    = L1.feedData(Xt)
O2    = L2.feedData(O1)
pred  = O2
pred[pred<=0.5]=0
pred[pred>0.5]=1


ll=np.not_equal(Yt.reshape(-1,1),pred)
print("\ntest error = ", sum(ll)/len(ll))









        