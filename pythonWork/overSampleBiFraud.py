# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:30:40 2017

@author: saadik1
"""

import pandas as pd
from explore import *


modelSteps = processSteps()  





#------------------------------------------------------------------------------#
#--------------------------- Data ---------------------------------------------#
sasLibname = "Y:\\racmi\\data\\analysis\\saadik\\"
BiData = pd.read_sas( sasLibname +"data_test_frd.sas7bdat", encoding='utf-8')
transToCategorical(data=BiData) 

modelSteps.copyTransfToCateg(True)
#modelSteps.transfToCateg=True
#------ Reading the fraud flags -------
frauds = pd.read_sas( sasLibname + "data_fr.sas7bdat", encoding='utf-8')
frauds['fraud_flag']=1.0

#------ merging data --------
BiData.rename(columns={'Claim_No': 'claim_no'}, inplace=True)
allData = pd.merge(BiData, frauds, how='left', on='claim_no')
allData['fraud_flag'].fillna(0.0,inplace = True)

freq(data=allData, var ='fraud_flag',graph =True)
#------------------------------------------------------------------------------#


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

# oversampling fraud 1/3 ratio
X_train = overSample(data=ff, target='fraud_flag', negFolds=2)
freq(X_train,'fraud_flag')


y_train = X_train.pop('fraud_flag')

#del dt3

# data wrangling
#imputing data : dealing with missings
# if missing=True, miss values for categ data = _MISS_
# the defautl values for imput are the  mode
impTrain = imputeObj( missing=False, mean=True, threshold = 0.6 )        
impTrain.impute(X_train,  drop=True) # dropping vars with missing>=60%
# anyhting we do to train set we must to to test set
impTrain.imputeNew(X_test)
modelSteps.copyImpute(impTrain)
#modelSteps.imputeObj =  impTrain


# if need standarization
stdTrain = standarizeObj()
stdTrain.standarize(X_train)
stdTrain.standarizeNew(X_test)
#modelSteps.standarizeObj= stdTrain
modelSteps.copyStandarize(stdTrain)


xx = toNumeric(X_train)  #<- important for modelling
    
vars = variablesSelection2(xx, y_train)    

# this is before standarization vals of 100 
# modelVars = vars[(vars['F_Score']>=100) & (vars['diff']>=0.01)  ]

modelVars = vars[(vars['F_Score']>=50) & (vars['diff']>=0.01)  ]

#modelSteps.variables = modelVars
modelSteps.copyVariables(modelVars)
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
svm    = SVC(kernel='poly',degree=2, probability=True)


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
Xt = testData
print("----- QDA ------")
qda  = qda.fit(X,y_train)
yhat = qda.predict(X)
diag.classAccuracy(y_train.as_matrix(), yhat)
#p = diag.svmTargetMisc(y,yhat)
pred1 = qda.predict(Xt)
diag.classAccuracy(y_test.as_matrix(), pred1)
#------------------ AUROC ----------------------
predprb = qda.predict_proba(X)
auc, fpr,tpr = diag.AUROC(y_train.as_matrix(),predprb[:,1])
print("AUC = %2.4f "% auc)
pred1prb = qda.predict_proba(Xt)
auc, fpr,tpr = diag.AUROC(y_test.as_matrix(),pred1prb[:,1])
print("AUC = %2.4f "% auc)
#ACC=0.6937  SENS=0.5846 PREC= 0.5372
#ACC=0.7063  SENS=0.5910 PREC= 0.1088
#AUC = 0.7105 
#AUC = 0.6922


print("----- LDA ------")
lda  = lda.fit(X,y_train)
yhat = lda.predict(X)
diag.classAccuracy(y_train.as_matrix(), yhat)
#p = diag.svmTargetMisc(y,yhat)
pred1 = lda.predict(Xt)
diag.classAccuracy(y_test.as_matrix(), pred1)
#------------------ AUROC ----------------------
predprb = lda.predict_proba(X)
auc, fpr,tpr = diag.AUROC(y_train.as_matrix(),predprb[:,1])
print("AUC = %2.4f "% auc)
pred1prb = lda.predict_proba(Xt)
auc, fpr,tpr = diag.AUROC(y_test.as_matrix(),pred1prb[:,1])
print("AUC = %2.4f "% auc)
#ACC=0.6937  SENS=0.5846 PREC= 0.5372
#ACC=0.7063  SENS=0.5910 PREC= 0.1088
#AUC = 0.7105 
#AUC = 0.6922



print("----- Logistic Reg ------")
logreg = logreg.fit(X,y_train)
yhat = logreg.predict(X)
diag.classAccuracy(y_train.as_matrix(), yhat)
#p = diag.svmTargetMisc(y,yhat)
pred1 = logreg.predict(Xt)
diag.classAccuracy(y_test.as_matrix(), pred1)
#------------------ AUROC ----------------------
predprb = logreg.predict_proba(X)
auc, fpr,tpr = diag.AUROC(y_train.as_matrix(),predprb[:,1])
print("AUC = %2.4f "% auc)
pred1prb = logreg.predict_proba(Xt)
auc, fpr,tpr = diag.AUROC(y_test.as_matrix(),pred1prb[:,1])
print("AUC = %2.4f "% auc)
#ACC=0.7347  SENS=0.4353 PREC= 0.6530
#ACC=0.8665  SENS=0.3767 PREC= 0.1761
#AUC = 0.7588 
#AUC = 0.7486 




from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
print("----- Naive Bayes ------")
gnb = gnb.fit(X,y_train)
yhat = gnb.predict(X)
diag.classAccuracy(y_train.as_matrix(), yhat)
pred1 = gnb.predict(Xt)
diag.classAccuracy(y_test.as_matrix(), pred1)
#------------------ AUROC ----------------------
predprb =gnb.predict_proba(X)
auc, fpr,tpr = diag.AUROC(y_train.as_matrix(),predprb[:,1])
print("AUC = %2.4f "% auc)
pred1prb = gnb.predict_proba(Xt)
auc, fpr,tpr = diag.AUROC(y_test.as_matrix(),pred1prb[:,1])
print("AUC = %2.4f "% auc)
#ACC=0.6323  SENS=0.7023 PREC= 0.4659
#ACC=0.5985  SENS=0.7188 PREC= 0.0944


print("----- k nearest neighbours ------")
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=8)
neigh.fit(X, y_train) 
yhat = neigh.predict(X)
diag.classAccuracy(y_train.as_matrix(), yhat)
#p = diag.svmTargetMisc(y,yhat)
Xt = testData
pred1 = neigh.predict(Xt)
diag.classAccuracy(y_test.as_matrix(), pred1)
#------------------ AUROC ----------------------
predprb = neigh.predict_proba(X)
auc, fpr,tpr = diag.AUROC(y_train.as_matrix(),predprb[:,1])
print("AUC = %2.4f "% auc)
pred1prb = neigh.predict_proba(Xt)
auc, fpr,tpr = diag.AUROC(y_test.as_matrix(),pred1prb[:,1])
print("AUC = %2.4f "% auc)

print("----- k nearest neighbours ------")
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X, y_train)
yhat = clf.predict(X)
diag.classAccuracy(y_train.as_matrix(), yhat)
#p = diag.svmTargetMisc(y,yhat)
pred1 = clf.predict(Xt)
diag.classAccuracy(y_test.as_matrix(), pred1)
#
predprb = clf.predict_proba(X)
auc, fpr,tpr = diag.AUROC(y_train.as_matrix(),predprb[:,1])
print("AUC = %2.4f "% auc)
pred1prb = clf.predict_proba(Xt)
auc, fpr,tpr = diag.AUROC(y_test.as_matrix(),pred1prb[:,1])
print("AUC = %2.4f "% auc)




#-------------------------------------------------------------------------#
#------------ Saving process flow -------------------#

modelSteps.copyModels([qda,lda,logreg, gnb,neigh,clf ])
#modelSteps.models=[qda,lda,logreg, gnb]


modelSteps.save()
