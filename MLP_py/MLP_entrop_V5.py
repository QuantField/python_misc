# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:48:47 2017

@author: saadik1
"""

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

Version 4 : the training is done properly, the tuning is done as 
            to lower the erro on a validation/test set
            
            
        Two classe, layer and output_layer, this makes more flexible
        the choice of error functions and activation ones.

Author : Dr Kamel Saadi
Date   : 12/12/2017

incorporating bias for each node
most likely to enable bias only in the output node.

"""




import numpy as np
import pandas as pd
from sklearn.metrics import  confusion_matrix 
import sklearn.metrics as sk


#-------------------------- Data --------------------------------------
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

#---------------------------------------------------------------------

#---------------------- Classes --------------------------------------  

class layer:
    def __init__(self, activ, n_nodes, n_inputs,  biasFlag = True):
        self.n_nodes  = n_nodes
        self.n_inputs = n_inputs
        self.activ    = activ
        self.biasVal  = float(biasFlag) # this is to use later in the training
        self.acfunc, self.deriv_out = self.__generate_funcs() 
        self.w = 2*np.random.random([n_nodes, n_inputs+1])-1
        if not activ in ['linear','sigmoid','tanh']:
            print("Allowed activation function : linear,sigmoid,tanh")
        # if no use of bias it is set to 0
        # th last value in the weight represent the bias
        if not biasFlag: self.w[:,-1] = 0.0                                       
    
    def feed(self,x):
        d = np.dot(self.w, np.hstack((x,1.0)))
        return  self.acfunc(d)
    
    def feedData(self,X):        
        one = np.ones([X.shape[0],1])        
        d   = np.dot(np.hstack((X,one)),self.w.T)
        return self.acfunc(d) 
   
    def __generate_funcs(self):
        if self.activ=='linear':
            def tmp(x):return x   
            def tmp1p(y):return 1.0       
        elif self.activ=='sigmoid':
            def tmp(x): return 1.0/(1.0+np.exp(-x))
            def tmp1p(y): return y*(1-y)                        
        elif self.activ=='tanh':
            def tmp(x): return np.tanh(x)
            def tmp1p(y): return (1-y*y)            
        else: pass
        return tmp, tmp1p    
    
    def bias(self):
        return self.w[:,-1]  
    

class output_layer(layer):
    def __init__(self, cost, activ, n_nodes, n_inputs, biasFlag ):
        layer.__init__(self,activ, n_nodes, n_inputs, biasFlag)
        self.costType = cost
        if not cost in ['sse','entropy']:
            print("Allowed activation function : 'sse','entropy'") 
        #  costFunc is the error function (y-f)^2
        #  costFuncDeriv is derivative of the cost function w.r.t f 
        #  for example 2(y-f)(-1)
        #  y is the target and f is the predicted value
        self.costFunc, self.costFuncDeriv = self.__costFunction()            
            
    def __costFunction(self):
        if self.costType=='sse':
            def tmp(y,f):return (y-f)**2
            def tmp1p(y,f) : return 2*(f-y)
        elif self.costType=='entropy':
            def tmp(y,f): return -(y*np.log(f)+(1-y)*np.log(1-f)) 
            def tmp1p(y,f): return -(y/f - (1-y)/(1-f))      
        else: pass
        return tmp, tmp1p
 
    def score(self,x, prevLayer):
        # default is one hidden layer
        o1 = prevLayer.feedData(x)
        o2 = self.feedData(o1)
        return o2
    
    def score2(self,x, Layer2,Layer1 ):
        # NNET with 3 layers (2 hidden) current is L3
        o1 = L1.feedData(x)
        o2 = L2.feedData(o1)
        o3 = self.feedData(o2)
        return o3

#--------------------------------------------------------------------------

np.random.seed(3148)     
np.random.seed(45)
    
n_nodes = 20             
in_dim = X.shape[1]

L1    = layer('sigmoid',n_nodes ,in_dim, False) # no bias
L2    = output_layer('entropy','sigmoid',1, n_nodes, True) # bias 

eta = 0.001

iters, sErr = 1000, np.Infinity
TrEnt       = []
TeEnt       = []
for i in range(iters):          
    #np.random.shuffle(X)
    for obs in range(Xtr.shape[0]):
        o0 , y = Xtr[obs,:], float(Ytr[obs])
        o1 = L1.feed(o0)
        o2 = float(L2.feed(o1))
       
        if o2>0.99999: o2=0.99999;
        if o2<0.00001: o2=0.00001;
        
        dCost_do2 = L2.costFuncDeriv(y,o2)
        
        delta2 = dCost_do2*L2.deriv_out(o2)        
        L2.w  += -eta*delta2*np.hstack((o1, L2.biasVal ))
            
        delta1 = delta2*L1.deriv_out(o1)            
        L1.w  += -eta*delta1.reshape(-1,1)*np.hstack((o0, L1.biasVal ))
        
    _tr = L2.costFunc(Ytr,L2.score(Xtr,L1))
    _te = L2.costFunc(Yte,L2.score(Xte,L1))
    
    train_err = float(sum(_tr))
    test_err  = float(sum(_te))
    sErr0     = sErr    
    sErr      = test_err  
    if (sErr-sErr0)/sErr0>1e-6: 
       break    
    print("epoch = ",i," Validation =",round(sErr,4)," Training =", 
                                                   round(train_err,4) )
    TrEnt.append(train_err)    
    TeEnt.append(test_err)    


pred  = L2.score(Xtr,L1) 
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











        