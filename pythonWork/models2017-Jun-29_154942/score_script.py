# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:53:22 2017

@author: saadik1
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:50:48 2017

@author: saadik1
"""

import pandas as pd
from explore import *

steps = processSteps('models2017-Jun-29_154942') 
steps.load()

sasLibname = "C:\\Users\\ks_work\\Desktop\\PythonWork\\pyData\\"
BiData = pd.read_sas( sasLibname +"bidatapersfraud.sas7bdat", encoding='utf-8')



'''
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
'Incident_Cause'] # Incident_Cause same as another field
 
BiData.drop(nogood,1, inplace=True) 
'''

stryears = [ str(r) for r in  list(range(1990,2018)) ]          
def filterYears(x):
    if x in stryears:
        return x        
    else:
        return None

BiData['Incident_Veh_Yr_Made'] = BiData['Incident_Veh_Yr_Made'].apply(lambda x: filterYears(x))        



#BiData.to_csv( steps.folder +'/'+'modelData.bz2', compression='bz2')


transToCategorical( data=BiData, nLevels = steps.nLevels )

steps.imputeObj.imputeNew(BiData)
steps.standarizeObj.standarizeNew(BiData)


variables = list(steps.variables.index)
orgVarNames = list(set([ r.split('$')[0]  for r in variables])) 


scoreData = toNumericB(data_in = BiData, 
                           keep = variables, 
                           origVars = orgVarNames )

print("----- Scoring one model ------")
#models = steps.models
#yscore = models[0].predict(scoreData)

#---- some stats ---------------
#---getting fraud flags 
frauds = pd.read_sas( sasLibname + "data_fr.sas7bdat", encoding='utf-8')
frauds['fraud_flag']=1.0
#------ merging data
BiData.rename(columns={'CLM_CUST_NO': 'claim_no'}, inplace=True)
allData = pd.merge(BiData, frauds, how='left', on='claim_no')
allData['fraud_flag'].fillna(0.0,inplace = True)
freq(data=allData, var ='fraud_flag',graph =True)
frdflag = allData['fraud_flag']


f = open(steps.folder+'/ScoreAll.txt','w')

#---------------------------------
y_scoreTest = frdflag.as_matrix()
Nobs = len(y_scoreTest)
Pos  = sum(y_scoreTest==1)
Neg  = Nobs - Pos
pos_rate = str((Pos/Nobs)*100)[0:6]+'%'
f.write("Scoring..:\n")
f.write('N= ' +str(Nobs)+'  Pos= '+str(Pos)+ '  Neg= '+str(Neg)+ '  Pos_rate= '+ pos_rate +'\n\n')

import diagnostics as diag

scoreDiag=[]
scoreRoc =[]

prob =[]
for mod in steps.models:
    i = steps.models.index(mod)
    print("------------> " + steps.modelNames[i] )
    f.write("------------> " + steps.modelNames[i]+'\n')
    #---------------------- training --------------------
    yhat    = mod.predict(scoreData)
    predprb = mod.predict_proba(scoreData)
    prob.append(predprb[:,1])
    acc, sens, prec = diag.classAccuracy(frdflag.as_matrix(), yhat)
    auc, fpr,tpr    = diag.AUROC(frdflag.as_matrix(),predprb[:,1])
    scoreDiag.append([auc,acc, sens, prec])
    scoreRoc.append([auc,fpr,tpr])
    print(" SCORE : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))
    f.write(" SCORE : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f\n"%(auc,acc, sens, prec))

f.close()    


def rocPic(data, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    style = ['b','r','g','k','m','c','b--','r--','k--','m--']
    for i in range(len(steps.models)):
        auc,fpr,tpr = data[i]
        legend = steps.modelNames[i]+ ' :'+str(auc)[0:5]
        plt.plot(fpr,tpr,style[i],label=legend)
    plt.grid()    
    plt.title(title)
    plt.xlabel('False positives')
    plt.ylabel('True positives')
    plt.legend(loc='lower right')  
    plt.savefig(steps.folder+'/score_roc')
    
rocPic(scoreRoc,"Score")

oveProb = np.array(prob)
oveProb = oveProb.transpose()
scoreProb = oveProb.mean(axis=1)
scoreFlag = np.array([ int(r>0.60) for r in  scoreProb])
print("----Ensemble average > 0.6---------")
acc, sens, prec = diag.classAccuracy(frdflag.as_matrix(), scoreFlag)
auc, fpr,tpr    = diag.AUROC(frdflag.as_matrix(),scoreProb)
print(" SCORE : AUROC=%2.4f  ACC=%2.4f  SENS=%2.4f  PREC= %2.4f"%(auc,acc, sens, prec))

#------ copying the current script --------#
currentCode = 'BiFraudPersonal_Score.py'
f1 = open( currentCode,'r')
f  = open(modelSteps.folder+'/'+'score_script.py','w')

for line in f1.readlines():
    f.write(line)
f.close()
f1.close()