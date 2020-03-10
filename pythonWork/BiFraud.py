# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:23:09 2016

@author: saadik1
"""


#
# when transforming to numric need to do k-1 levels instead of k
#


import pandas as pd

from explore import *

sasLibname = "Y:\\racmi\\data\\analysis\\saadik\\"

BiData = pd.read_sas( sasLibname +"data_test_frd.sas7bdat", encoding='utf-8')


# testing few things from explore package
freq(BiData,'ACCIDENT_DAY')
freq(data=BiData,var='ACCIDENT_TIME')     
freq(data=BiData,var='Damage_Severity' )     
freq(data=BiData,var='Damage_Severity', graph=True)     
freq(data=BiData,var='Damage_Severity', graph=True, missing=True)
#------ missing rates ----------------
#----- infor about missing for every variable
miss = countMissings(BiData)  
miss = countMissings(data =BiData, graph=True)


#problm transform nan float to nan String

#if we want to see the vars we want to drop
#dropVars = varsToDrop(data=BiData, threshold=0.6)



#----- getting the values for impute  
# threshold is the threshold of missing.. drop anything more
# mean=True means the impute will use mean to fill missings
# if mean=false than the mode will be used. Here no transformation
# to BiData
toKeep,toDrop = getImputeValue(BiData, mean=True, threshold = 0.6)    
  

# detecting categorical variables
transToCategorical(data=BiData)     #<--- important      




#------ Reading the fraud flags
frauds = pd.read_sas( sasLibname + "data_fr.sas7bdat", encoding='utf-8')
frauds['fraud_flag']=1.0

#------ merging data
BiData.rename(columns={'Claim_No': 'claim_no'}, inplace=True)
allData = pd.merge(BiData, frauds, how='left', on='claim_no')
allData['fraud_flag'].fillna(0.0,inplace = True)

freq(data=allData, var ='fraud_flag',graph =True)


#------ oversample --------#

# allData.drop_duplicates(['claim_no']) # dedup per claim_no
#del BiData

# example of histogram
allData.TP_KA_DISTANCE_MX.hist(bins=100, alpha=0.5)


#=======================================================================

dt3 = allData
# data partitioning  
from sklearn.cross_validation import train_test_split
# working with oversampled data
frd =dt3.pop('fraud_flag')
X_train, X_test, y_train, y_test = train_test_split( dt3,frd, 
                                                    test_size=0.3, 
                                                    random_state=42)
print("Training..")
freq(data=pd.DataFrame(y_train), var='fraud_flag')
print("Testing..")
freq(data=pd.DataFrame(y_test), var='fraud_flag')

#del allData
#del dt3

# data wrangling
#imputing data : dealing with missings
# if missing=True, miss values for categ data = _MISS_
# the defautl values for imput are the  mode
impTrain = imputeObj(X_train, missing=False, threshold = 0.6 )        
impTrain.impute(drop=True) # dropping vars with missing>=60%
# anyhting we do to train set we must to to test set
impTrain.imputeNew(X_test)

import pickle

pickle.dump(impTrain,open( "imputeObj.sav", "wb" ))




# if need standarization
#stdTrain = standarizeObj(X_train)
#stdTrain.standarize()
#stdTrain.standarizeNew(X_test)



xx = toNumeric(X_train)  #<- important for modelling
    
vars = variablesSelection2(xx, y_train)    

# this is before standarization vals of 100 
# modelVars = vars[(vars['F_Score']>=100) & (vars['diff']>=0.01)  ]

modelVars = vars[(vars['F_Score']>=100) & (vars['diff']>=0.01)  ]

pickle.dump(modelVars,open( "modelVars.sav", "wb" ))


#names = [r[0:r.rfind('$')]  for r in modelVars.index] 
#   .groupby('name').max()

# if diff==0 variable and it's exact oposite are used, so we
# choose one
finalModData = xx[modelVars.index]
# Now prepare the test/score set 
yy = toNumeric(X_test)
testData = yy[modelVars.index]
   




import sklearn.discriminant_analysis as da
import sklearn.linear_model as lm
from sklearn.svm import SVC
lda    = da.LinearDiscriminantAnalysis()
logreg = lm.LogisticRegression()
qda    = da.QuadraticDiscriminantAnalysis()
svm = SVC(kernel='poly',degree=2, probability=True)


import diagnostics as diag


# with standarisation:
# Train	ACC=0.6089  SENS=0.7014 PREC= 0.0941
# Test 	ACC=0.6093  SENS=0.6924 PREC= 0.0940
#

# without standarization...and vars['F_Score']>=100
#  49 vars used
#ACC=0.8213  SENS=0.4090 PREC= 0.1345
#ACC=0.8156  SENS=0.3919 PREC= 0.1272
#
X = finalModData
model3 = qda.fit(X,y_train)
yhat = model3.predict(X)
diag.classAccuracy(y_train.as_matrix(), yhat)
#p = diag.svmTargetMisc(y,yhat)
Xt = testData
pred1 = model3.predict(Xt)
diag.classAccuracy(y_test.as_matrix(), pred1)
#ACC=0.8424  SENS=0.3316 PREC= 0.1323
#ACC=0.8386  SENS=0.3238 PREC= 0.1278

model3 = lda.fit(X,y_train)
yhat = model3.predict(X)
diag.classAccuracy(y_train.as_matrix(), yhat)
#p = diag.svmTargetMisc(y,yhat)
Xt = testData
pred1 = model3.predict(Xt)
diag.classAccuracy(y_test.as_matrix(), pred1)
#--- no standarization
#ACC=0.9438  SENS=0.0233 PREC= 0.3859
#ACC=0.9434  SENS=0.0211 PREC= 0.3852




model2 = logreg.fit(X,y_train)
yhat = model2.predict(X)
diag.classAccuracy(y_train.as_matrix(), yhat)
#p = diag.svmTargetMisc(y,yhat)
Xt = testData
pred1 = model3.predict(Xt)
diag.classAccuracy(y_test.as_matrix(), pred1)
#ACC=0.9445  SENS=0.0014 PREC= 0.4118
#ACC=0.9434  SENS=0.0211 PREC= 0.3852

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
nbmodel = gnb.fit(X,y_train)
yhat = nbmodel.predict(X)
diag.classAccuracy(y_train.as_matrix(), yhat)
pred1 = nbmodel.predict(Xt)
diag.classAccuracy(y_test.as_matrix(), pred1)
#ACC=0.6943  SENS=0.6069 PREC= 0.1060
#ACC=0.6928  SENS=0.6121 PREC= 0.1071

















