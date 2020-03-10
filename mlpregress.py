
import numpy as np

class layer:
    def __init__(self, activ, n_nodes, n_inputs):
        self.n_nodes  = n_nodes
        self.n_inputs = n_inputs
        self.activ    = activ
        self.acfunc, self.deriv_out = self.__generate_funcs() 
        self.w = 2*np.random.random([n_nodes, n_inputs])-1
        if not activ in ['linear','sigmoid','tanh']:
            print("Allowed activation function : linear,sigmoid,tanh")                                       
    
    def feed(self,x):
        return  self.acfunc(np.dot(self.w,x.reshape(-1,1)))
    
    def feedData(self,X):
        d = np.dot(X,self.w.T)
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

#--------------------------------------------------------------------------

X   = np.linspace(-10,10,500).reshape(-1,1)
n,m = X.shape
ytrue = np.sin(X)/X

Y = ytrue + 0.4*np.random.randn(n,m)

            

n_te = int(n*0.5) 
import random   

te_ind = random.sample(list(range(n)),n_te)
tr_ind = list(set(list(range(n)))-set(te_ind))

Xte, Yte = X[te_ind,:], Y[te_ind]
Xtr, Ytr = X[tr_ind,:], Y[tr_ind]

def score(x):
    o1 = L1.feedData(x)
    o2 = L2.feedData(o1)    
    return o2

     


       
in_dim = Xte.shape[1]


np.random.seed(197)
n_nodes = 100    
L1      = layer('tanh',n_nodes ,in_dim)
L2      = layer('tanh',1 ,n_nodes)

eta = 0.001
iters, sErr = 2000, np.Infinity
for i in range(iters):   
    for obs in range(Xtr.shape[0]):
        o0 , y = Xtr[obs,:], float(Ytr[obs])
        o1 = L1.feed(o0)
        o2 = float(L2.feed(o1))
 
        dCost_do2 = 2*(o2-y)
        
        delta2 = dCost_do2*L2.deriv_out(o2)
        L2.w  += -eta*delta2*o1.reshape(1,-1)
            
        delta1 = delta2*L1.deriv_out(o1)    
        L1.w += -eta*delta1*o0    
    
    resid = Yte - score(Xte)
    valid_err = float(sum(resid**2))    
    sErr0     = sErr    
    sErr      = valid_err    
    if sErr>sErr0 and i>20: #(sErr-sErr0)/sErr0>1e-6: 
       break    
    print("epoch  ",i," Validation error =", round(valid_err,4) )
     
       





import  matplotlib.pyplot as plt

pred  = score(X) 
plt.figure(figsize=(7,6))
plt.plot(X,Y,'.', label='noise sine')
plt.plot(X,ytrue,'-k', label='true sine')
plt.plot(X,pred,'-r', label='nnet')
plt.grid()
plt.xlabel('x')
plt.legend(loc='upper right')  
plt.title('NNET regression')
plt.show()



    
    
  
    
    
    
    
    
            