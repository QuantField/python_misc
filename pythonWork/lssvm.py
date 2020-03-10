# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:27:56 2016

@author: saadik1
"""
import numpy as np
import numpy.linalg as al
import math as mth
import abc
import diagnostics as diag
import copy
    
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
    def params(self):
        return ""
    @abc.abstractmethod    
    def evaluate(self,x1,x2): 
        return

class Linear(Kernel):
    def __init__(self):
        Kernel.__init__(self,'Linear')        
    def evaluate(self,x1,x2):
        return x2.dot(x1.transpose())
        
class Polynomial(Kernel):
    def __init__(self, order, ofset):
        super(Polynomial,self).__init__('Polynomial')
        self.__order  = order
        self.__ofset  = ofset        
    def evaluate(self,x1,x2):
        return (x2.dot(x1.transpose())+self.__ofset)**self.__order
    def params(self):
        s = "Order = "+str(self.__order) + "   Ofset = "+str(self.__ofset)
        return s
           
class RBF(Kernel):
    def __init__(self, sigma=0.5):
        super(RBF,self).__init__('RBF')
        self.__width = sigma
        self.__type  ='RBF'
    def  width(self):
        return self.__width
    # Good starting point for the width
    def setInitWidh(self,trData):
        self.__width = 0.5*al.norm(trData.std(0))
    def getWidth(self):
        return self.__width
        
    def squared_distance(self, x1, x2):  
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        p = (x1**2).sum(axis=1)
        p.shape=(n1,1)
        par1 = p.dot(np.ones([1,n2]))
        if (x1 is x2):
            par2 = par1.transpose()
        else:           
            q = (x2**2).sum(axis=1)
            q.shape=(1,n2)
            par2 = np.ones([n1,1]).dot(q)
        return (par1 + par2 -2*x1.dot(x2.transpose()))
        
    def evaluate(self,x1,x2):
        vExp = np.vectorize(mth.exp)
        w = (self.__width)**2
        K = self.squared_distance(x1, x2)
        return vExp(-K/w).transpose()        
   
    def params(self):
        return 'Width = '+str(self.__width)
        
class LSSVM:
   
    muArray = 10**np.arange(-3,2.1,0.1)

    def __init__(self, kern =RBF(), mu=0.1, PrbType="class"):
        self.alpha   = None #np.zeros([10,1]) 
        self.ntp     = 0.0
        self.bias    = 0.0
        self.x       = None #np.zeros([10,1]) 
        self.y       = None #np.zeros([10,1]) 
        self.mu      = mu
        self.optimMu = 0.0
        self.optimRBFWidth = 0.0
        self.optiPRESS = 0.0
        self.Kernel  = kern
        self.yhat    = None #np.zeros([10,1])
        self.type    = PrbType.upper()
        if self.type not in ('CLASS','REG'):
            print("LSSVM type is either 'class' or 'reg'..defaulted to class")
            self.type = 'CLASS'
        
    def copy(self):
        return copy.deepcopy(self)       
      
    def train(self,x,y):
        #target y  must be in (-1,1)
        n = len(y)
        self.ntp = n;
        self.x   = x
        self.y   = y
        K = self.Kernel.evaluate(x,x)
        T = np.ones([n+1,n+1]) ; T[n][n] = 0.0
        T[:n,:n] =  K + self.mu*np.eye(n)           
        Sol   = np.linalg.solve(T , np.append(y,0))
        self.alpha, self.bias = Sol[0:n], Sol[n]
        self.yhat  = K.dot(self.alpha)+self.bias
        self.errStats(act=self.y, pred=self.yhat)
    
#    def predict(self,xt):
#        K = self.Kernel.evaluate(xt,self.x)
#        pred = (K.transpose()).dot(self.alpha)+self.bias
#        prob, sign = 0, 0
#        if self.type == 'CLASS':
#            vlogit = np.vectorize(lambda x: 1/(1+mth.exp(-x)))
#            prob = vlogit(pred)
#            sign = np.sign(pred)
#        return (pred,prob,sign)
       
    def predict(self,xt):
        P = xt.shape[0]
        N = 2000 # size of chunks to score
        Steps = int(P/N)
        pred = []
        prob = []
        sign = []        
        for i in range(Steps+1):
            start = i*N + (i!=0)
            stop  = min(P-1,(i+1)*N)
            #print(" scoring chunk :",  start," - ", stop)
            K  = self.Kernel.evaluate(xt[list(range(start,stop+1)), :],self.x)
            prd = (K.transpose()).dot(self.alpha)+self.bias
            prb, sin = 0, 0
            if self.type == 'CLASS':
                vlogit = np.vectorize(lambda x: 1/(1+mth.exp(-x)))
                prb = vlogit(prd)
                sin = np.sign(prd)
            pred = np.append(pred, prd)
            prob = np.append(prob, prb)
            sign = np.append(sign, sin)
        return (pred, prob, sign)    
        
    
    def residuals(self):
        return (self.y-self.yhat)
        
    def looResiduals(self): 
        n = self.ntp;
        K = self.Kernel.evaluate(self.x,self.x);	       
        T = np.ones([n+1,n+1]) ; T[n][n] = 0.0
        T[:n,:n] =  K + self.mu*np.eye(n)                
        H = np.concatenate((K, np.ones([n,1])),axis=1).dot(al.inv(T)) 
        looResid = (self.y-self.yhat)/(1-H.diagonal())
        press = (looResid**2).sum()
        return (looResid, press)
        
        
    def errStats(self, act, pred): 
        R2      = diag.R_Sqr(act, pred)
        p       = self.x.shape[1]
        ajustR2 = diag.R_Sqr_Ajust(act,pred,p)
        if self.type == 'CLASS':
            error   = diag.svmTargetMisc(act,pred)
            error = 100*error
            auc,fp,tp = diag.AUROC(act,pred)
            print("missCassification rate = %2.2f"%error+"%")
            print("AUROC                  = %2.4f"%auc )  
        print("R squared              = %2.4f"%R2)
        print("Ajusted R squared      = %2.4f"%ajustR2 )      
            
    
    def setOptimalRegularParam(self, Mu = muArray):        
        y = self.y      	
        eigVal, V = al.eigh(self.Kernel.evaluate(self.x,self.x))
        Vt_y   = V.transpose().dot(y)
        Vt_sqr = V.transpose()**2
        xi     = (V.sum(axis=0)).transpose()
        xi2    = xi**2;
        PRESS = np.zeros(len(Mu))
        for i in range(len(Mu)):		
            u     = xi/(eigVal+Mu[i])
            g     = eigVal/(eigVal+Mu[i])
            sm    = -(xi2/(eigVal+Mu[i])).sum()						
            theta = Vt_y/(eigVal+Mu[i]) + (u.dot(Vt_y)/sm)*u		
            h     = Vt_sqr.transpose().dot(g) + (V.dot(u*eigVal)-1)*(V.dot(u))/sm		            
            f     = V.dot(eigVal*theta) -sum(u*Vt_y)/sm
            loo_resid = (y - f)/(1-h); 			
            PRESS[i] = (loo_resid**2).sum()
            #print("Mu= %2.4f  PRESS=%f"%(Mu[i],PRESS[i]))
        #---- Retrain LSSVM
        optmu = Mu[PRESS.argmin()]
        self.optimMu   = optmu
        self.optiPRESS = min(PRESS) 
        
    def optimRetrain(self):
        if (self.optimMu !=0):
            self.mu = self.optimMu
            self.train(self.x,self.y)
        else:
            print("No optimal regularisation done yet")
            print("run getOptimalRegularParam method first")
             
    def setXY(self,x,y):
         self.x = x
         self.y = y
    
    def optRBF(self):
         kn = RBF()
         kn.setInitWidh(self.x)
         sig = kn.getWidth()
         sigma = (10**np.arange(-3,2.25,0.25))*sig
         muX    = np.zeros(len(sigma))
         pressX = np.zeros(len(sigma))        
         for i in range(len(sigma)):
             ls = LSSVM(RBF(sigma[i]))
             ls.type = self.type
             ls.setXY(self.x,self.y)
             ls.setOptimalRegularParam() 
             muX[i]    = ls.optimMu
             pressX[i] = ls.optiPRESS
             print("Width = %4.4f  Mu =%4.4f  PRESS=%8.4f"%(sigma[i],muX[i],pressX[i]))
         muOpt  = muX[pressX.argmin()]
         sigOpt = sigma[pressX.argmin()]
         print("Optimal Parameters: RBF Width =%4.6f, Regular Param =%4.6f" %(sigOpt, muOpt))
         netOpt = LSSVM(RBF(sigOpt),muOpt,self.type)
         netOpt.optimMu = muOpt
         netOpt.optimRBFWidth = sigOpt
         print("training with opt parameters...")
         netOpt.train(self.x, self.y)
         return netOpt     