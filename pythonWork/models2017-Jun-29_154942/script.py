# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:58:58 2017

@author: saadik1
"""



import pandas as pd
from explore import *


modelSteps = processSteps()  



#------------------------------------------------------------------------------#
#--------------------------- Data ---------------------------------------------#
#sasLibname = "Y:\\racmi\\data\\analysis\\saadik\\"
sasLibname = "C:\\Users\\ks_work\\Desktop\\PythonWork\\pyData\\"
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
'CLM_BI_DTE_FIRST_EST',
'Incident_Cause', # Incident_Cause same as another field
'COST_PAID_TOT_CLMNT_AMT',
'COST_PAID_AD_CLMNT_AMT',
'COST_PAID_PD_CLMNT_AMT']
 
 
BiData.drop(nogood,1, inplace=True) 


# works but slower
# keep Incident_Veh_Yr_Made within 1990-2017
#stryears = [ str(r) for r in  list(range(1990,2018)) ]          
#for ind, r in BiData.iterrows():
#    #if not (r['AGE'] is None) :
#    if BiData.get_value(ind,'Incident_Veh_Yr_Made') not in stryears:
#        BiData.set_value(ind,'Incident_Veh_Yr_Made',None)


# faster way
stryears = [ str(r) for r in  list(range(1990,2018)) ]          
def filterYears(x):
    if x in stryears:
        return x        
    else:
        return None

BiData['Incident_Veh_Yr_Made'] = BiData['Incident_Veh_Yr_Made'].apply(lambda x: filterYears(x))        
        
        
#freq(data=BiData, var='Incident_Veh_Yr_Made')          

 
 
transToCategorical(data=BiData, nLevels = 20) 
modelSteps.copyTransfToCateg(True, nLevels = 20)
#modelSteps.transfToCateg=True
#------ Reading the fraud flags -------
frauds = pd.read_sas( sasLibname + "data_fr.sas7bdat", encoding='utf-8')
frauds['fraud_flag']=1.0

#------ merging data --------
BiData.rename(columns={'CLM_CUST_NO': 'claim_no'}, inplace=True)
allData = pd.merge(BiData, frauds, how='left', on='claim_no')
allData['fraud_flag'].fillna(0.0,inplace = True)

freq(data=allData, var ='fraud_flag',graph =True)


	

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
X_train  = overSample(data=ff, target='fraud_flag', negFolds=4)
X_train2 = overSample(data=ff, target='fraud_flag', negFolds=4, seed=38494)
X_train3 = overSample(data=ff, target='fraud_flag', negFolds=4, seed=5324)
X_train4 = overSample(data=ff, target='fraud_flag', negFolds=4, seed=11257)

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
#impTrain = imputeObj( missing=False, mean=True, threshold = 0.6 )   --> version 1     
impTrain = imputeObj( missing=True, mean=True, threshold = 0.6 )        
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


#xx = toNumeric(X_train) 
xx = toNumericB(X_train)  #<- important for modelling
    
vars = variablesSelection2(xx, y_train)    

# this is before standarization vals of 100 
# modelVars = vars[(vars['F_Score']>=100) & (vars['diff']>=0.01)  ]

modelVars = vars[(vars['F_Score']>=50) & (vars['diff']>=0.01)  ] #50 is good 
#variables = list(modelVars.index)
#modelSteps.variables = modelVars
modelSteps.copyVariables(modelVars)
#names = [r[0:r.rfind('$')]  for r in modelVars.index] 
#   .groupby('name').max()

# if diff==0 variable and it's exact oposite are used, so we
# choose one
#finalModData = xx[variables]


   
#xx  =  toNumericB(data_in = X_train) 
   
#vars = variablesSelection2(xx, y_train)    
# this is before standarization vals of 100 
# modelVars = vars[(vars['F_Score']>=100) & (vars['diff']>=0.01)  ]
#modelVars = vars[(vars['F_Score']>=30) & (vars['diff']>=0.01)  ] #50 is good 


variables = list(modelVars.index) # actual variables num + dummies

# original variable names 
orgVarNames = list(set([ r.split('$')[0]  for r in variables])) 
  

finalModData = xx[variables]                      
   

finalModData2 = toNumericB(data_in = X_train2, 
                           keep = variables, 
                           origVars = orgVarNames )

finalModData3 = toNumericB(data_in = X_train3, 
                           keep = variables,
                           origVars = orgVarNames )

finalModData4 = toNumericB(data_in = X_train4, 
                           keep = variables,
                           origVars = orgVarNames )

testData     = toNumericB(data_in = X_test, 
                           keep = variables,
                           origVars = orgVarNames )


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
svm    = SVC(kernel='poly',cache_size=500,   degree=3, 
             probability=True, C=.75)
#svm    = SVC(kernel='poly',C = 0.5, degree=2, probability=True)
#svm    = SVC(gamma=0.03,C = 0.5,  probability=True)
gnb   = GaussianNB()
neigh = KNeighborsClassifier(n_neighbors=6)
clf   = DecisionTreeClassifier(max_depth=4)
rmf   = RandomForestClassifier(max_depth=5)
adab  = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=4))

#neigh,gnb,qda, adab tends to overfit
models   = [lda,logreg, clf, svm,  rmf ] # models neigh,
modNames = [type(r).__name__ for r in models]   # model Names


import diagnostics as diag

trainSet1 = [finalModData,y_train]
trainSet2 = [finalModData2,y_train2]
trainSet3 = [finalModData3,y_train3]
trainSet4 = [finalModData4,y_train4]


X = [trainSet1,trainSet2,trainSet3,trainSet4,trainSet1,trainSet2,
     trainSet3,trainSet4,trainSet1,trainSet2]
     
#modelSteps.folder+     
f = open(modelSteps.folder+"/log.txt", 'w')     

#======================================================================================================
#---------------------------------- Reporting ------------------------------------

f.write('*--------------- Modelling Summary ---------------*\n')
import datetime
f.write('\nTimeStamp:' + '{:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())+'\n\n')
#------ Significant Variables -----
vv = copy.deepcopy(modelVars)
vv['name'] = [ r.split('$')[0]  for r in vv.index]
varSign = vv.groupby(['name'])['F_Score'].max()
varSign.sort_values(ascending=False, inplace=True)
f.write('Variable Significance: '+ '(' + str(varSign.shape[0]) +' variables) \n' )
tmp = str(varSign)
tmp = tmp[0:tmp.rfind('Name')]
f.write(tmp)
f.write('\n\n')
#---------------------------------
Nobs = len(y_train)
Pos  = sum(y_train==1)
Neg  = Nobs - Pos
pos_rate = str((Pos/Nobs)*100)[0:6]+'%'
f.write("Training Info:\n")
f.write('N= ' +str(Nobs)+'  Pos= '+str(Pos)+ '  Neg= '+str(Neg)+ '  Pos_rate= '+ pos_rate +'\n\n')
f.write("Testing Info:\n")
Nobs = len(y_test)
Pos  = sum(y_test==1)
Neg  = Nobs - Pos
pos_rate = str((Pos/Nobs)*100)[0:6]+'%'
f.write('N= ' +str(Nobs)+'  Pos= '+str(Pos)+ '  Neg= '+str(Neg) + '  Pos_rate= '+ pos_rate + '\n\n')
f.write('*-------------------------------------------------*\n\n')
#--------------------------------------------------------------------------------
#======================================================================================================

from sklearn.externals.six import StringIO
     
Xt = testData

trainDiag =[]
testDiag  =[]
roc_train =[]           
roc_test  =[]
for mod in models:
    i = models.index(mod)
    print("------------> " + modNames[i] )
    f.write("------------> " + modNames[i] +"\n")
    Xtrain = X[i][0]
    ytrain = X[i][1]
    mod.fit(Xtrain,y_train)
    #---------------------- training --------------------
    yhat    = mod.predict(Xtrain)
    predprb = mod.predict_proba(Xtrain)
    acc, sens, prec = diag.classAccuracy(ytrain.as_matrix(), yhat)
    auc, fpr,tpr    = diag.AUROC(ytrain.as_matrix(),predprb[:,1])
    trainDiag.append([auc,acc, sens, prec])
    roc_train.append([auc,fpr,tpr])
    print(" TRAIN : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))
    f.write(" TRAIN : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f\n"%(auc,acc, sens, prec))
    #---------------------- testing --------------------
    pred1    = mod.predict(Xt)
    pred1prb = mod.predict_proba(Xt)
    acc, sens, prec = diag.classAccuracy(y_test.as_matrix(), pred1)
    auc, fpr,tpr = diag.AUROC(y_test.as_matrix(),pred1prb[:,1])
    testDiag.append([auc,acc, sens, prec])
    roc_test.append([auc,fpr,tpr])
    print(" TEST  : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))
    f.write(" TEST  : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f\n"%(auc,acc, sens, prec))

f.close()
    
summaryTrain = pd.DataFrame(trainDiag, index = modNames, 
                            columns =   ['AUROC(train)','ACC(train)',
                            'SENS(train)','PREC(train)'])
summaryTest  = pd.DataFrame(testDiag, index = modNames, 
                            columns =   ['AUROC(test)','ACC(test)','SENS(test)','PREC(test)'])

    
modSummary   =  pd.concat([summaryTrain,summaryTest], axis =1)

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
modelSteps.copyModelsSummary(modSummary)
#modelSteps.models=[qda,lda,logreg, gnb]

modelSteps.save()

#------ copying the current script --------#
currentCode = 'PersonalBiFraudOversample.py'
f1 = open( currentCode,'r')
f  = open(modelSteps.folder+'/'+'script.py','w')

for line in f1.readlines():
    f.write(line)
f.close()
f1.close()