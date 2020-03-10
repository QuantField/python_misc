"""
This is a simple implementation of a ann with one hidden layer.
The activation function is a sigmoid.
This is in no way efficient, but rather educational.

Author : Dr Kamel Saadi
Date   : 04/12/2017
"""
import numpy as np
import pandas as pd
from math import exp

np.random.seed(2340)
path  = "C:\\myData\\banana\\"
train = pd.read_csv(path+"banana_train_1.csv", header = None)

n = train.shape[0] # number of observations 
m = train.shape[1] # number of features +1(target)

X = train.loc[:,0:m-2].as_matrix()
Y = train.loc[:,m-1].as_matrix()

Y[(Y<0)]=0.0 #  transform the target as sigmoid in [0, 1]


def sigmoid(x):
    f      = 1.0/(1.0+exp(-x))
    return f

class node:
    def __init__(self,n):
        self.n_inp = n
        self.w     = 2*np.random.rand(self.n_inp)-1 # w in [-1,1]
    
    def w_sum(self,x):
        return np.dot(self.w,x)
    
    def input(self,x):
        return sigmoid(self.w_sum(x))
    
# layer 0, number of features or variables      
n_input1 = m-1   

#---- Hidden layer -----
n_nodes  = 20
layer1 = [node(n_input1)  for _ in range(n_nodes)] # first layer 

#---- Output layer, one node only
n_inputs2 = len(layer1) 
layer2    = node(n_inputs2) 

eta      = 0.1 # learning rate( step coefficient in gradient descent)
maxiters = 1000 
SSE =  np.Infinity
for iter in range(maxiters):
    err  = []
    yhat = []
    for obs in range(n):
        x  = X[obs,:]
        y  = Y[obs]
        #---- traversing the network -----
        o0 = x
        o1 = [ node.input(o0)  for node in layer1]        
        o2 = layer2.input(o1)
        #--- compute the error on observation obs
        yhat.append(o2)
        err.append(y-o2)
        #--- updating the weights ---------
        #--- ouput layer
        delta_1 = -(y-o2)*o2*(1-o2)
        layer2.w += -eta*delta_1*np.array(o1)
        #--- hidden layer ----            
        for j, nodej in enumerate(layer1):
            delta_j = o1[j]*(1-o1[j])*delta_1
            nodej.w += -eta*delta_j*o0  
    SSE0 = SSE        
    SSE  = sum([r*r for r in err])
    if SSE > SSE0 : break
    print("epoch :", iter, "SSE =",SSE)

def score(x):
    o1 = [ node.input(x)  for node in layer1]        
    o2 = layer2.input(o1)
    return o2    
    
pred = np.zeros(len(Y))
for i in range(len(Y)):
    pred[i] = score(X[i,:])
 
pred[(pred>0.5)] = 1
pred[(pred<=0.5)] = 0     

ll=np.not_equal(Y,pred)
print("\ntrain error = ", sum(ll)/len(ll))
        
#--------- testing ---------
te = pd.read_csv(path+"banana_test_1.csv", header = None)
Xt = te.loc[:,0:m-2].as_matrix()
Yt = te.loc[:,m-1].as_matrix()
Yt[(Yt<0)]=0.0
pred = np.zeros(len(Yt))
for i in range(len(Yt)):
    pred[i] = score(Xt[i,:])
 
pred[(pred>0.5)] = 1
pred[(pred<=0.5)] = 0     

ll=np.not_equal(Yt,pred)
print("test error  = ", sum(ll)/len(ll))




    
        
        
        
        