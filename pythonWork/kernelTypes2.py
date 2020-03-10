import numpy as np
import numpy.linalg as la
import math as mth
import sklearn.metrics as sk
import abc
    
class Kernel:
    def __init__(self,kerntype):
        self._type="Not defined"
        s = kerntype.strip().upper()
        if s in ('LINEAR','POLYNOMIAL','RBF'):
            self._type = s
        else:
            print("invalid Kernel, only  LINEAR, POLYNOMIAL,RBF are allowed")
    def __str__(self):        
        return self._type    
    @abc.abstractmethod    
    def evaluate(self,x1,x2): 
        return

class Linear(Kernel):
    def __init__(self):
        super(Linear,self).__init__('Linear')        
    def evaluate(self,x1,x2):
        return x2.dot(x1.transpose())
        
class Polynomial(Kernel):
    def __init__(self, order, ofset):
        super(Polynomial,self).__init__('Polynomial')
        self.__order  = order
        self.__ofset  = ofset        
    def evaluate(self,x1,x2):
        return (x2.dot(x1.transpose())+self.__ofset)**self.__order
           
class RBF(Kernel):
    def __init__(self, sigma=0.5):
        super(RBF,self).__init__('RBF')
        self.__width = sigma
        self.__type  ='RBF'
    def  width(self):
        return self.__width
    # Good starting point for the width
    def setInitWidh(self,trData):
        self.__width = 0.5*la.norm(trData.std(0))
    def evaluate(self,x1,x2):
        n1 = x1.shape[0]
        n2 = x2.shape[0]
   
        p = (x1**2).sum(axis=1)
        p.shape=(n1,1)
        par1 = p.dot(np.ones([1,n2]))
        
        q=(x2**2).sum(axis=1)
        q.shape=(1,n2)
        par2 = np.ones([n1,1]).dot(q)
       
        K = par1 + par2 -2*x1.dot(x2.transpose())
        vExp = np.vectorize(mth.exp)
        w = (self.__width)**2
        return vExp(-K/w).transpose()        

        
class LSSVM:
    import numpy.linalg as al
    muArray = 10**np.arange(-3,2.1,0.1)

    def __init__(self, kern =RBF(), mu=0.1):
        self.alpha  = np.zeros([10,1]) 
        self.ntp    = 0.0
        self.bias   = 0.0
        self.x      = np.zeros([10,1]) 
        self.y      = np.zeros([10,1]) 
        self.mu     = mu
        self.Kernel = kern
        self.yhat   = np.zeros([10,1])
      
    def train(self,x,y):
        #target y  must be in (-1,1)
        n = len(y)
        self.ntp = n;
        self.x   = x
        self.y   = y
        K = self.Kernel.evaluate(x,x)
        T = K + self.mu*np.eye(n) 
        T = np.concatenate((T,np.ones([n,1])),axis=1 )
        T = np.concatenate((T,np.ones([1,n+1])),axis=0 )
        T[n][n]=0.0        
        tar   = np.append(y,0.0)
        Sol   = np.linalg.solve(T,tar)
        self.alpha = Sol[0:n]
        self.bias  = Sol[n]    
        self.yhat  = (K.transpose()).dot(self.alpha)+self.bias
        self.errStats(act=self.y, pred=self.yhat)
    
    def predict(self,xt):
        K = self.Kernel.evaluate(xt,self.x)
        vlogit = np.vectorize(lambda x: 1/(1+mth.exp(-x)))
        pred = (K.transpose()).dot(self.alpha)+self.bias
        prob = vlogit(pred)
        sign = np.sign(pred)
        return (pred,prob,sign)
        
    def errStats(self, act, pred): 
        error = (np.sign(pred)!=act).sum()/len(act)
        R2 = sk.r2_score(act, pred)
        n = len(act)
        p = self.x.shape[1]
        aj = p/(n-p-1)
        ajustR2 = R2 -(1-R2)*aj
        error = 100*error
        print("missCassification rate = %2.2f"%error+"%")
        print("R squared              = %2.4f"%R2)
        print("Ajusted R squared      = %2.4f"%ajustR2 )      
        
    # usage : bestmu , bestpress = <something>.tuneRegularisationParameter()
    def tuneRegularParam(self, Mu = muArray):        
        y = self.y;       	
        eigVal, V = al.eigh(self.Kernel.evaluate(self.x,self.x));
        Vt_y   = V.transpose().dot(y);
        Vt_sqr = V.transpose()**2;
        xi     = (V.sum(axis=0)).transpose()
        xi2    = xi**2;
        PRESS = np.zeros(len(Mu))
        for i in range(len(Mu)):		
            u     = xi/(eigVal+Mu[i]);
            g     = eigVal/(eigVal+Mu[i]);
            sm    = -(xi2/(eigVal+Mu[i])).sum();						
            theta = Vt_y/(eigVal+Mu[i])-(-(u*Vt_y).sum()/sm)*u;									
            h     = Vt_sqr.transpose().dot(g) + (V.dot(u*eigVal)-1)*(V*u)/sm;		            
            f     = V*(eigVal*theta) + -sum(u*Vt_y)/sm;
            loo_resid = (y - f)/(1-h); 			
            PRESS[i] = (loo_resid**2).sum();	
            print("Mu= %2.4f  PRESS=%f"%(Mu[i],PRESS[i]))
        return (Mu,PRESS);
			     
        
  

import pandas as pd
import sklearn.metrics as sk

allData = pd.read_csv("C:\\KS_temp\\banana_data1000.csv")


trainData = allData.sample(frac=0.7)        

y = trainData['Class']
x = trainData[['X','Y']]

net = LSSVM( kern=RBF(0.5), mu = 0.001)

print("---------- Diagnostics -------------")
print("----------Train Partition-----------")
net.train(x.as_matrix(),y.as_matrix())
fpr1, tpr1, thresholds = sk.roc_curve(net.y, net.yhat)
print("AUROC = %2.4f"%sk.auc(fpr1, tpr1))

print("----------Test Partition-----------")
testData  = allData.loc[~allData.index.isin(trainData.index)] 
yt = testData['Class']
xt = testData[['X','Y']]
(pred,prob,sign) = net.predict(xt.as_matrix())
net.errStats(yt,pred)
fpr2, tpr2, thresholds = sk.roc_curve(yt.as_matrix(), pred)
print("AUROC = %2.4f"%sk.auc(fpr2, tpr2))

'''
import matplotlib.pyplot as plt
plt.plot(fpr1,tpr1,'b-',label='Train')
#plt.legend(loc='upper righ', shadow=True)
plt.plot(fpr2,tpr2,'r-',label='Test')
plt.legend(loc='upper right', shadow=True)
'''

mu, press = net.tuneRegularParam()
plt.plot(mu,press)


         
         












		 
        
        
