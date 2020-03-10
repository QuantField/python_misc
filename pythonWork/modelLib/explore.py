# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:28:49 2016

@author: saadik1
"""

import pandas as pd
import numpy as np
import datetime
import os
import pickle 
import copy

# Equivalent to proc freq in SAS
def freq(data , var, graph=False, missing=False ):
     Vr   = data[var]
     obs  = Vr.shape[0]
     miss = sum(Vr.isnull())
     if (missing==True):
         Vr = Vr.fillna('_MISS_')
     dc = Vr.value_counts(dropna = True).sort_index()
     dc.name = '        Freq'
     dc2 = (dc/dc.sum())*100
     dc2.name = 'Perc'
     dc3 = dc.cumsum()
     dc3.name ='CumFreq'
     dc4 = dc2.cumsum()
     dc4.name = 'CumPerc'
     dc2 = dc2.map('{:,.2f}'.format)
     dc4 = dc4.map('{:,.2f}'.format)
     print("="*55)
     print(var)
     print("N       = %d"%obs)
     print("Missing = %d(%2.2f%%)"%(miss,miss*100/obs))
     print("-"*55) 
     print(pd.concat([dc, dc2, dc3, dc4], axis=1, join='inner'))
     if (graph==True):
         import matplotlib.pyplot as plt
         print("-"*55)     
         labels = dc2.index
         x_pos  = range(len(labels))
         y_pos  = dc2.astype('d').as_matrix()
         plt.bar(x_pos, y_pos,  align='center', alpha=0.5)
         plt.xticks(x_pos, labels, rotation=60)      
         plt.show()
     print("="*55)
     
def numOfmissings(var):
        nmiss = sum(var.isnull());
        return (nmiss, nmiss*100.0/len(var))     
        
def countMissings(data, graph=False):
    miss= data.apply(lambda var: sum(var.isnull()), axis=0)      
    MM = pd.DataFrame(miss, columns=['Nmiss'])
    MM['Perc'] = MM['Nmiss']/data.shape[0]
    MM  = MM.sort_values(by='Perc') 
    if (graph==True):
        import matplotlib.pyplot as plt
        y_pos = range(len(MM.index))    
        size = 10;
        if len(y_pos)>10:
            size = round(0.34*len(y_pos))
        plt.figure(figsize=(4, size))
        plt.barh(y_pos, MM['Perc'], align='center', alpha=0.5)
        plt.yticks(y_pos, MM.index)
        plt.title('Missing Values percentage')
        plt.ylim([0,len(y_pos)])
        plt.show()        
    return MM    
    
def transToCategorical(data, nLevels=15):
    # convert to categorical if number of levels is small
    vInfo = numVarInfo(data)
    inf = vInfo.transpose()
    print("converted from numeric to string")
    for var in inf.index:
        if inf.ix[var]['levels'] <=nLevels:    
           print(var) 
           data[var]=data[var].astype(str)
           #data[data[var]=='nan'][var]=None           
           data.loc[(data[var]=='nan'),var]=None       

# returns the list of variables to drop with respect to a missing threshold
def varsToDrop(data, threshold=0.8):
   w,r = getImputeValue(data, missing=False, mean=False, threshold=threshold)
   return r
    
def getImputeValue(data, missing=False, mean=False, threshold=1.0):
    #if missing is True a new level _MISS_ is created
    miss = countMissings(data) 
    missVals = {}
    toDrop   = []
    for var in miss.index:
        if miss.ix[var]['Perc']!=0 and miss.ix[var]['Perc']<=threshold:
            typ = data[var].dtype
            # Numeric Data either with mode or mean
            if (typ == np.float64 or typ == np.int64):
                if mean==False:
                    missVals.update({var:data[var].mode().at[0]})
                else:
                    missVals.update({var:data[var].mean()})                   
            else:
                if missing==True:
                    missVals.update({var:'_MISS_'})
                else:
                    missVals.update({var:data[var].mode().at[0]})
        if miss.ix[var]['Perc']>threshold:
            toDrop.append(var)            
    return (missVals,toDrop)
  
def impute(data, missing=False, mean=False, imputeVal=None, 
           threshold=1.0, drop=False, toDrop=None): 
    if imputeVal is None:
        missVals, toDrop = getImputeValue(data, missing= missing, 
                                  mean=mean, threshold=threshold) 
    else:
        missVals = imputeVal
    for w in data.columns:
        if w in  missVals:
            print("imputed var = "+w+"  with value = "+str(missVals[w])) 
            #data[w]=data[w].fillna(missVals[w])
            data[w].fillna(missVals[w], inplace=True)
    if drop==True:
        data.drop(toDrop,1, inplace=True)           
    return (missVals, toDrop)           
    
def numVarInfo(data):
    s = {}
    for var in data.columns:
        typ = data[var].dtype
        if (typ == np.float64 or typ == np.int64):
            d = data[var].describe()
            levels = len(data[var].unique())            
            s.update({d.name:[d.ix['mean'],d.ix['std'], 
                              d.ix['min'], d.ix['max'],levels]})
    return pd.DataFrame(s, index=['mean','std','min','max','levels'])              

class imputeObj:
    def __init__(self, missing=False , mean=False, threshold =1.0):
        # if missing=True missing value will be replaced by _MISS_
        self.threshold = threshold        
        self.missing   = missing
        self.mean      = mean # the defaut impute is mode, if true it is the mean
        self.status    = False
        self.drop      = False

    def impute(self, trainData, drop=False):
        if self.status == False:
            self.impValues, self.dropped = impute(data=trainData, missing=self.missing,
                                    mean=self.mean, threshold=self.threshold,
                                    drop = drop)
            self.drop   = drop  
            self.status = True
        else:
            print("Impute process in train data already done")
    
    # this replicates exact transformation on newdata    
    def imputeNew(self,newdata):
        if self.status==True:
            dumm1, dumm2 = impute(data=newdata, missing=self.missing, 
                                  mean=self.mean, imputeVal = self.impValues,
                                  toDrop = self.dropped)
        else:
           print("Run impute() first") 
        if self.drop==True :
           newdata.drop(self.dropped,1, inplace=True) 
           
class standarizeObj:
    # need to fix missings before running this
    def __init__(self):
        self.status         = False
        self.standarized    = False
        self.standarizedNew = False

    def getInfo(self, data):
        self.info = numVarInfo(data)
        self.status = True
        
    def __stdrz__(self, data):
        for var in self.info.columns:
            mean = self.info[var].ix['mean']
            std  = self.info[var].ix['std']
            data[var] = data[var].apply(lambda x: (x-mean)/std) 
        
    def standarize(self, data):
        if self.status==False:            
            self.getInfo(data)    
        if self.standarized==False:  
            self.standarized = True
            self.__stdrz__(data)
        else:
             print("Data already standarized")
    
    def standarizeNew(self,newData):
        if self.standarized==True:
            self.__stdrz__(newData)
        else:
            print("Must standarize train data")

# ---------- Variable selection ---------------------------#
def variablesSelection(X_train, y_train):
    from sklearn.feature_selection import SelectPercentile
    from sklearn.feature_selection import f_regression
    from sklearn.feature_selection import f_classif
    data = X_train
    num=[] # will be filled with numerical variables
    cat=[] # will be filled with categorical variables
    for r in data.columns:
        typ = data[r].dtype
        if typ == np.float64 or typ==np.int64:
            num.append(r)
        else:
            cat.append(r)
    #--------------- Numerical Variables ------------------------#    
    # even if we use f_regression it will use f_classif
    # as it detects that the target is categorical
    Selector_f = SelectPercentile(f_regression, percentile=10)
    Selector_f.fit(X_train[num],y_train)
    vars1 =pd.DataFrame(Selector_f.scores_, columns=['F_Score'], index=data[num].columns)
    #--------------- Categorical Variables ------------------------#    
    # even if we use f_regression it will use f_classif
    # as it detects that the target is categorical
    data = X_train
    # creating empty data
    data2 = pd.DataFrame(data.index, index=data.index)
    data2.pop(0)
    # transforming to dummy variables
    for r in data[cat]:
      if len(data[r].unique())<=30:   #levels max = 30
          p = pd.get_dummies(data[r], prefix= r , prefix_sep='$', drop_first=True)    
          data2 = pd.concat([data2, p], axis=1)
    
    Selector_f = SelectPercentile(f_regression, percentile=25)
    Selector_f.fit(data2,y_train)
    vars2 =pd.DataFrame(Selector_f.scores_, columns=['F_Score'], index=data2.columns)
    # recovering original var names with score       
    vars2['name'] = [r[0:r.rfind('$')]  for r in vars2.index] 
    CatVars = vars2.groupby('name').max()
    score   = pd.concat([vars1,CatVars], axis=0)
    score = score.sort_values(by='F_Score', ascending=False)
    varList = [ r   for r in score.index]
    return score, varList
    
def toNumeric(data_in):
    MAX_LEVELS = 50
    data = data_in.copy()
    for r in data:
      dtyp = data[r].dtype
      if  not (dtyp in [np.float64, np.int64]) and len(data[r].unique())>MAX_LEVELS:   #levels max = 30
          print("dropped: "+ r)
          data.drop(r,1, inplace=True)
    return pd.get_dummies(data, prefix_sep='$', drop_first=True)  
    
def variablesSelection2(X_train, y_train):
    from sklearn.feature_selection import SelectPercentile
    from sklearn.feature_selection import f_regression
    from sklearn.feature_selection import f_classif
    #
    Selector_f = SelectPercentile(f_regression, percentile=25)
    Selector_f.fit(X_train,y_train)   
    vars1 =pd.DataFrame(Selector_f.scores_, columns=['F_Score'], index=X_train.columns)
    vars2 = vars1.sort_values(by='F_Score', ascending =False) 
    vars2['rowNo'] = list(range(len(X_train.columns)))
    vars2['diff'] = abs(vars2['F_Score'].diff())
    vars2['diff'].ix[0]=9999 # this value was nan
    return  vars2
    
def overSample(data, target, negFolds, seed = 12345):
    pos = data[(data[target]>0)]
    P = pos.shape[0]
    N = P*negFolds               
    neg = data[~data.index.isin(pos.index)]   
    neg2 = neg.sample(n =N, random_state=seed)
    return pos.append(neg2)
    
class processSteps:  # keeps track of transoformations       
    def __init__(self, folder = None):
        if folder is None:
            self.folder= 'models'+'{:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
            self.folder= self.folder.replace(' ','_')
            self.folder= self.folder.replace(':','')            
        else:
            self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.standarize = False
        self.imputeObj  = None
        self.standarizeObj = None
        self.transfToCateg = False
        self.variables = None
        self.models = []
        self.modelNames = []
        self.modelsSummary = None
       
    def __dump(self, obj, filename):        
        pickle.dump(obj, open(self.folder+'/'+filename,"wb"))
        
    def copyStandarize(self, obj):
        self.standarizeObj = copy.deepcopy(obj) 
        self.standarize = True
        
    def copyImpute(self, obj):
        self.imputeObj = copy.deepcopy(obj) 
    
    def copyVariables(self, obj):
        self.variables = copy.deepcopy(obj) 
        
    def copyTransfToCateg(self, obj):
        self.transfToCateg = copy.deepcopy(obj) 
        
    def copyModels(self, obj):
        # obj is a list
        self.models = copy.deepcopy(obj)
        self.modelNames = [type(obj[i]).__name__ for i in range(len(obj))]
        
    def copyModelsSummary(self,obj):
       self.modelsSummary =  copy.deepcopy(obj)                        
                           
    def __load(self, filename):  
       return pickle.load(open(self.folder+'/'+filename,"rb"))            
        
    def save(self):           
        if len(os.listdir(self.folder))<5: # simple way to check is already saved
            self.__dump(self.standarize, 'standarize')
            self.__dump(self.imputeObj, 'imputeObj') 
            self.__dump(self.standarizeObj, 'standarizeObj')
            self.__dump(self.transfToCateg, 'transfToCateg')
            self.__dump(self.variables, 'variables')
            self.__dump(self.models, 'models')
            self.__dump(self.folder, 'folder')
            self.__dump(self.modelNames, 'modelNames')
            self.__dump(self.modelsSummary,'modelsSummary')
        else:
            print("directory not empty") 
        
    def load(self):
        self.standarize     = self.__load('standarize')
        self.imputeObj      = self.__load('imputeObj')
        self.standarizeObj  = self.__load('standarizeObj')
        self.transfToCateg  = self.__load('transfToCateg')
        self.variables      = self.__load('variables')
        self.models         = self.__load('models')
        self.modelNames     = self.__load('modelNames')
        self.modelsSummary  = self.__load('modelsSummary')
        #self.folder         = self.__load('folder')
'''
def  dataTypes(data, varList):
    S = {}
    for var in varList:
        desc = data[var].describe()
        if (desc.dtype == np.float64 or desc.dtype==np.int64):
            S.update({var:[desc.ix['count'], desc.dtype]})
        else:
            S.update({var:[desc.ix['unique'], desc.dtype]})
    return S
    
    
infoVars = dataTypes(data = BiData, varList = modelVariables)
'''
    
    
    
    
    