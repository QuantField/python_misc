# -*- coding: utf-8 -*-
"""
This continuing on MLP2

try to implement a nnet with two hidden layers

@author: ks_work
"""

# -*- coding: utf-8 -*-
"""
same as MLP.py but using matrices

@author: ks_work
"""

import numpy as np
import pandas as pd

np.random.seed(2340)

path  = "C:\\myData\\image\\"
train = pd.read_csv(path+"image_train_1.csv", header = None)

#path  = "C:\\myData\\heart\\"
#train = pd.read_csv(path+"heart_train_1.csv", header = None)

nfeat = train.shape[1]-1
# for this code to work faster we need to transform to np.arrays
X = train.loc[:,0:nfeat-1].as_matrix() 

Y = train.loc[:,nfeat].as_matrix()

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


# ------ Creatign the layers --------

n_input1 = m    
layer1 = layer(40,m)     # layer1 20 nodes

n_inputs2 = layer1.n_nodes
layer2 = layer(10,n_inputs2) # layer2 5 nodes

n_inputs3 = layer2.n_nodes
layer3 = layer(1,n_inputs3) # layer3 1 node


eta = 0.01
yhat, err = np.zeros(n),np.zeros(n)
iters = 1000
err0 = 1E+20
for iter in range(iters):  
    for obs in range(n):
        x  = X[obs,:]
        y  = Y[obs]
        o0 = x  # convert to array very important for speed         
        o1 = layer1.input(o0)        
        o2 = layer2.input(o1)
        o3 = layer3.input(o2)
        
        yhat[obs] = o3              
        err[obs]= (y-yhat[obs])
          
        #---update of weights         
        delta_3   = -(y-o3)*o3*(1-o3) 
        layer3.w += -eta*delta_3*o2 # input layer
       
        delta_2 = o2*(1-o2)*delta_3
        for node_id in range(layer2.n_nodes):              
            layer2.w[node_id,:] += -eta*delta_2[node_id]*o1
            
        #layer2.w += -eta*delta_2.reshape(-1,1)*o1    
        
        tt = o1*(1-o1)
        for i in range(len(delta_2)):
            delta_1 = tt*delta_2[i]
            for node_id in range(layer1.n_nodes):              
                layer1.w[node_id,:] += -eta*delta_1[node_id]*o0
            
    dd = np.dot(err,err)
    print("iter =",iter,"SSE =",dd)
    if dd>err0: break
    err0 = dd
   
def score(x):
    o1 = layer1.input(x)        
    o2 = layer2.input(o1)
    o3 = layer3.input(o2)    
    return o3

pred = np.zeros(len(Y))
for i in range(len(pred)):
    pred[i] = score(X[i,:])

    
pred[(pred>0.5)] = 1
pred[(pred<=0.5)] = 0     

ll=np.not_equal(Y,pred)
print("train error = ", sum(ll)/len(ll))
        
#comp2 = np.vstack((Y,t)).T



    
        
        
        
        