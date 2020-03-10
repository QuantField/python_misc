# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:30:40 2017

@author: saadik1


Same as before using more oversampled datasets 
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
impTrain = imputeObj( missing=False, mean=True, threshold = 0.6 )        
impTrain.impute(X_train,  drop=True) # dropping vars with missing>=60%
# doing the same for the rest of the set for the sake of keeeping same vars for all
impTrain.imputeNew(X_train2)
impTrain.imputeNew(X_train3)
impTrain.imputeNew(X_train4)
# anyhting we do to train set we must to to test set
impTrain.imputeNew(X_test)

modelSteps.copyImpute(impTrain)
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

xx = toNumeric(X_train2)
finalModData2 = xx[modelVars.index]
xx = toNumeric(X_train3)
finalModData3 = xx[modelVars.index]
xx = toNumeric(X_train4)
finalModData4 = xx[modelVars.index]
# Now prepare the test/score set 
yy = toNumeric(X_test)
testData = yy[modelVars.index]
   


import sklearn.discriminant_analysis as da
import sklearn.linear_model as lm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

lda    = da.LinearDiscriminantAnalysis()
logreg = lm.LogisticRegression()
qda    = da.QuadraticDiscriminantAnalysis()
svm    = SVC(kernel='poly',degree=2, probability=True)
gnb   = GaussianNB()
neigh = KNeighborsClassifier(n_neighbors=8)
clf   = DecisionTreeClassifier(max_depth=4)
rmf   = RandomForestClassifier(max_depth=5)
adab  = AdaBoostClassifier()


models   = [lda,logreg,qda,gnb,clf,svm, neigh,adab, rmf ] # models neigh,
modNames = [type(r).__name__ for r in models]   # model Names




import diagnostics as diag

X = finalModData
Xt = testData

trainDiag =[]
testDiag  =[]
roc_train=[]           
roc_test =[]
for mod in models:
    i = models.index(mod)
    print("------------> " + modNames[i] )
    mod.fit(X,y_train)
    #---------------------- training --------------------
    yhat    = mod.predict(X)
    predprb = mod.predict_proba(X)
    acc, sens, prec = diag.classAccuracy(y_train.as_matrix(), yhat)
    auc, fpr,tpr    = diag.AUROC(y_train.as_matrix(),predprb[:,1])
    trainDiag.append([auc,acc, sens, prec])
    roc_train.append([auc,fpr,tpr])
    print(" TRAIN : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))
    #---------------------- testing --------------------
    pred1    = mod.predict(Xt)
    pred1prb = mod.predict_proba(Xt)
    acc, sens, prec = diag.classAccuracy(y_test.as_matrix(), pred1)
    auc, fpr,tpr = diag.AUROC(y_test.as_matrix(),pred1prb[:,1])
    testDiag.append([auc,acc, sens, prec])
    roc_test.append([auc,fpr,tpr])
    print(" TEST  : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))
    

summaryTrain = pd.DataFrame(trainDiag, index = modNames, 
                            columns =   ['AUROC(train)','ACC(train)',
                            'SENS(train)','PREC(train)'])
summaryTest = pd.DataFrame(testDiag, index = modNames, 
                            columns =   ['AUROC(test)','ACC(test)','SENS(test)','PREC(test)'])

    
modSummary =  pd.concat([summaryTrain,summaryTest], axis =1)

#--------------------- saving soem reports -----------------------

save_directory = modelSteps.folder+'/'

def rocPic(data, title, savelocation):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    style = ['b','r','g','k','m','c','b--','r--','k--','m--']
    for i in range(len(models)):
        auc,fpr,tpr = data[i]
        legend = modNames[i]+ ' :'+str(auc)[0:5]
        plt.plot(fpr,tpr,style[i],label=legend)
    plt.grid()    
    plt.title(title)
    plt.xlabel('False positives')
    plt.ylabel('True positives')
    plt.legend(loc='lower right')  
    #---- saving to pdf
    plt.savefig(savelocation)  
    
rocPic(roc_train, "Training",save_directory+"train_roc.png")   

rocPic(roc_test, "Testing",save_directory+"test_roc.png")      
    
modSummary.to_csv(save_directory+"csvfile.csv")



#-------------------------------------------------------------------------#
#------------ Saving process flow -------------------#

modelSteps.copyModels(models)
modelSteps. copyModelsSummary(modSummary)
#modelSteps.models=[qda,lda,logreg, gnb]


modelSteps.save()
