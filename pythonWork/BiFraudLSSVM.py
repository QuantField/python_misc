# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:51:50 2017

@author: saadik1
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:30:40 2017

@author: saadik1


Same as before using more oversampled datasets 
"""

import pandas as pd
from explore import *


#modelSteps = processSteps()  



sasLibname = "Y:\\racmi\\data\\analysis\\saadik\\"
BiData = pd.read_sas( sasLibname +"bidatapersfraud.sas7bdat", encoding='utf-8')



#datvare = [ r  for r in X_train.columns if r.upper().find('TIME')>0 ]
#initVars = [r for r in BiData.columns]           
       
# CLM_CAUSE_CDE is the same as  CAUSE_CODE, drop one or the other        

nogood = ['CLM_ID_EDW'	,
'CLM_CAUSE_CDE',          
'CLM_CRRT_STS'	,
'CLM_DTE_TRANS'	,
'CLM_TRANS_TIME'	,
'CLM_POL_REF'	,
'CLM_THG_IDFR'	,
'CLM_CLUBLINE_REF'	,
'CLM_DAMAGE_SEV_CDE'	,
'CLM_POL_CRRT_STS'	,
'CLM_PRD_CDE'	,
'CLM_PREV_IND'	,
'CLM_PREV_NOTE'	,
'Incident_Number',
 'Inci_Create_Date_Time',
 'CLM_BI_DTE_FIRST_EST']


BiData.drop(nogood,1, inplace=True) 
 
 
transToCategorical(data=BiData, nLevels = 20) 
#modelSteps.transfToCateg=True
#------ Reading the fraud flags -------
frauds = pd.read_sas( sasLibname + "data_fr.sas7bdat", encoding='utf-8')
frauds['fraud_flag']=1.0

#------ merging data --------
BiData.rename(columns={'CLAIM_NO': 'claim_no'}, inplace=True)
allData = pd.merge(BiData, frauds, how='left', on='claim_no')
allData['fraud_flag'].fillna(-1.0,inplace = True)

freq(data=allData, var ='fraud_flag',graph =True)


#=============================================================================
#=============================================================================


dt3 = allData
# data partitioning  
from sklearn.cross_validation import train_test_split
# working with oversampled data
frd =dt3.pop('fraud_flag')
X_trainT, X_test, y_trainT, y_test = train_test_split( dt3,frd,                        
                                                      test_size=0.3, 
                                                    random_state=42)
print("Training..")
freq(data=pd.DataFrame(y_trainT), var='fraud_flag')
print("Testing..")
freq(data=pd.DataFrame(y_test), var='fraud_flag')


ff = pd.concat([X_trainT, y_trainT], axis = 1)

# Generating four samples oversampling fraud 1/3 ratio
X_train  = overSample(data=ff, target='fraud_flag', negFolds=2)
X_train2 = overSample(data=ff, target='fraud_flag', negFolds=2, seed=38494)
X_train3 = overSample(data=ff, target='fraud_flag', negFolds=2, seed=5324)
X_train4 = overSample(data=ff, target='fraud_flag', negFolds=2, seed=11257)

#freq(X_train,'fraud_flag')

y_train  = X_train.pop('fraud_flag')
y_train2 = X_train2.pop('fraud_flag')
y_train3 = X_train3.pop('fraud_flag')
y_train4 = X_train4.pop('fraud_flag')

#del dt3

# data wrangling
#imputing data : dealing with missings
# if missing=True, miss values for categ data = _MISS_
# the defautl values for imput are the  mode
impTrain = imputeObj( missing=True, mean=True, threshold = 0.6 )        
impTrain.impute(X_train,  drop=True) # dropping vars with missing>=60%
# doing the same for the rest of the set for the sake of keeeping same vars for all
impTrain.imputeNew(X_train2)
impTrain.imputeNew(X_train3)
impTrain.imputeNew(X_train4)
# anyhting we do to train set we must to to test set
impTrain.imputeNew(X_test)


#modelSteps.imputeObj =  impTrain


# if need standarization
stdTrain = standarizeObj()
stdTrain.standarize(X_train)
#
stdTrain.standarizeNew(X_train2)
stdTrain.standarizeNew(X_train3)
stdTrain.standarizeNew(X_train4)
#
stdTrain.standarizeNew(X_test)
#modelSteps.standarizeObj= stdTrain



xx = toNumeric(X_train)  #<- important for modelling
    
vars = variablesSelection2(xx, y_train)    

# this is before standarization vals of 100 
# modelVars = vars[(vars['F_Score']>=100) & (vars['diff']>=0.01)  ]

modelVars = vars[(vars['F_Score']>=50) & (vars['diff']>=0.01)  ] #50 is good 

#modelSteps.variables = modelVars

#names = [r[0:r.rfind('$')]  for r in modelVars.index] 
#   .groupby('name').max()

# if diff==0 variable and it's exact oposite are used, so we
# choose one
finalModData = xx[modelVars.index]

xx = toNumeric(X_train2)
finalModData2 = xx[modelVars.index]
xx = toNumeric(X_train3)
finalModData3 = xx[modelVars.index]
xx = toNumeric(X_train4)
finalModData4 = xx[modelVars.index]
# Now prepare the test/score set 
yy = toNumeric(X_test)
testData = yy[modelVars.index]
   




X = finalModData.as_matrix()
y = y_train.as_matrix()

Xt = testData.as_matrix()
yt = y_test.as_matrix()



import lssvm as ls
import diagnostics as diag


net = ls.LSSVM( kern=ls.Polynomial(1,0.1), mu = 0.1)




net.train(X,y)
(yhat,prob,sign) = net.predict(X)
acc, sens, prec = diag.classAccuracy(y, sign)
auc, fpr,tpr    = diag.AUROC(y,prob)
print(" TRAIN : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))
#print("----------Test Partition-----------")
(pred,prob,sign) = net.predict(Xt)
acc, sens, prec = diag.classAccuracy(yt, sign)
auc, fpr,tpr    = diag.AUROC(yt,prob)
print(" TEST  : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))
#print("----------Test Partition-----------")






print("------ opt Regularisation -------------")
net.setOptimalRegularParam()
net.optimRetrain()
#
(yhat,prob,sign) = net.predict(X)
acc, sens, prec = diag.classAccuracy(y, sign)
auc, fpr,tpr    = diag.AUROC(y,prob)
print(" TRAIN : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))
#print("----------Test Partition-----------")
(pred,prob,sign) = net.predict(Xt)
acc, sens, prec = diag.classAccuracy(yt, sign)
auc, fpr,tpr    = diag.AUROC(yt,prob)
print(" TEST  : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))



print("------ Full optimisation RBF -------------")
print("------ Train partition ------------")
optNet = net.optRBF()
(yhat,prob,sign) = optNet.predict(X)
acc, sens, prec = diag.classAccuracy(y, sign)
auc, fpr,tpr    = diag.AUROC(y,prob)
print(" TRAIN : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))
#print("----------Test Partition-----------")
(pred,prob,sign) = optNet.predict(Xt)
acc, sens, prec = diag.classAccuracy(yt, sign)
auc, fpr,tpr    = diag.AUROC(yt,prob)
print(" TEST  : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))




print("------ Linear Kernel ---------")
print("------ opt Regularisation -------------")
net = ls.LSSVM(kern=ls.Linear())
net.train(X,y)
net.setOptimalRegularParam()
net.optimRetrain()
#
(yhat,prob,sign) = net.predict(X)
acc, sens, prec = diag.classAccuracy(y, sign)
auc, fpr,tpr    = diag.AUROC(y,prob)
print(" TRAIN : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))
#print("----------Test Partition-----------")
(pred,prob,sign) = net.predict(Xt)
acc, sens, prec = diag.classAccuracy(yt, sign)
auc, fpr,tpr    = diag.AUROC(yt,prob)
print(" TEST  : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))

diag.plotROC(yt,prob)



#--------------------------------------------

pos = (y>0).nonzero()[0] # index of y>0
neg = (y<0).nonzero()[0] # index of y>0
Xp  = X[pos,:]
Xn  = X[neg,:]  
meanP = Xp.mean(axis = 0)
meanN = Xn.mean(axis = 0)

