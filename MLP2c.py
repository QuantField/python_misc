# -*- coding: utf-8 -*-
"""
same as MLP.py but using matrices

@author: ks_work
"""

import numpy as np
import pandas as pd

np.random.seed(2340)

path  = "C:\\myData\\banana\\"
train = pd.read_csv(path+"banana_train_1.csv", header = None)

#path  = "C:\\myData\\image\\"
#train = pd.read_csv(path+"image_train_1.csv", header = None)

nfeat = train.shape[1]-1
# for this code to work faster we need to transform to np.arrays
X = train.loc[:,0:nfeat-1].as_matrix() 

Y = train.loc[:,nfeat].as_matrix()

# for this code to work faster we need to transform to np.arrays
#X = train.loc[:,0:1].as_matrix() 

#Y = train.loc[:,2].as_matrix()

Y[(Y<0)]=0.0

n = X.shape[0] # number of observations 
m = X.shape[1] # number of features

def sigmoid(x):
    f      = 1.0/(1.0+np.exp(-x))
    return f

class layer:
    def __init__(self, num_nodes, num_input):
        self.n_nodes = num_nodes
        self.n_inp   = num_input
        # initial random weights in [-1,1]
        self.w = 2*np.random.random([ num_nodes,num_input])-1      
 
    def w_sum(self,x):
        return np.dot(self.w,x)
    
    def input(self,x):       
        return sigmoid(np.dot(self.w,x))

# layer 0, number of features or variables 
n_input1 = m    

#---- Hidden layer -----
n_nodes  = 20
layer1 = layer(n_nodes,m)
#---- Output layer, one node only
n_inputs2 = layer1.n_nodes
layer2 = layer(1,n_inputs2)


def score(x):
    o1 = layer1.input(x)        
    o2 = layer2.input(o1)
    return o2

def scoreMat(M):
    n = M.shape[0];
    pred = np.zeros(n)
    for i in range(n):
        pred[i] = score(M[i,:])
    pred[(pred>0.5)] = 1
    pred[(pred<=0.5)] = 0  
    return pred

eta = 0.1
yhat = np.zeros(n)
iters = 1000
SSE =  np.Infinity
for iter in range(iters):  
    err = []
    for obs in range(n):
        x  = X[obs,:]
        y  = Y[obs]
        #---- traversing the network -----
        o0 = x          
        o1 = layer1.input(o0)        
        o2 = layer2.input(o1)
        #--- compute the error on observation obs
        yhat[obs] = o2              
        err.append(y-yhat[obs])
        #--- updating the weights ---------
        #--- ouput layer        
        delta_2   = -(y-o2)*o2*(1-o2) 
        layer2.w += -eta*delta_2*o1 # input layer
        #--- hidden layer ----  
        delta_1 = o1*(1-o1)*delta_2
        layer1.w += -eta*delta_1.reshape(-1,1)*o0
        
    SSE0 = SSE 
    SSE  = np.dot(err,err)
    if SSE > SSE0 : break
    print("epoch :", iter, "SSE =",SSE)
    
   
pred = scoreMat(X) 

ll=np.not_equal(Y,pred)
print("\ntrain error = ", sum(ll)/len(ll))
        
#comp2 = np.vstack((Y,t)).T
#------------------testing

#te = pd.read_csv(path+"heart_test_1.csv", header = None)
#
#nfeat = te.shape[1]-1
## for this code to work faster we need to transform to np.arrays
#Xt = te.loc[:,0:nfeat-1].as_matrix() 
#
#Yt = te.loc[:,nfeat].as_matrix()
#
## for this code to work faster we need to transform to np.arrays
##X = train.loc[:,0:1].as_matrix() 
#
##Y = train.loc[:,2].as_matrix()
#
#Yt[(Yt<0)]=0.0
#pred = scoreMat(Xt) 
#
#ll=np.not_equal(Yt,pred)
#print("\ntest error = ", sum(ll)/len(ll))
#    
#        
        
        
        