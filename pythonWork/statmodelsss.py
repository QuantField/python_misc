# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:47:18 2017

@author: saadik1
"""


import pandas as pd
from explore import *

import explore as expl




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
 
expl.transToCategorical(data=BiData, nLevels = 20) 

#------ Reading the fraud flags -------
frauds = pd.read_sas( sasLibname + "data_fr.sas7bdat", encoding='utf-8')
frauds['fraud_flag']=1.0

#------ merging data --------
BiData.rename(columns={'CLAIM_NO': 'claim_no'}, inplace=True)
allData = pd.merge(BiData, frauds, how='left', on='claim_no')
allData['fraud_flag'].fillna(0.0,inplace = True)

freq(data=allData, var ='fraud_flag',graph =True)



X_train  = expl.overSample(data=allData, target='fraud_flag', negFolds=2)


impTrain = expl.imputeObj( missing=True, mean=True, threshold = 0.6 )        

impTrain.impute(X_train,  drop=True) # dropping vars with missing>=60%
# doing the same for the rest of the set for the sake of keeeping same vars for all
#stdTrain = standarizeObj()
#stdTrain.standarize(X_train)

vars = list(X_train.columns)

vars.remove('fraud_flag')
vars.remove('claim_no')
vars.remove('CLM_CUST_NO')



S = 'fraud_flag'
for r in vars:
    if vars.index(r) ==0: 
        op = ' ~ '
    else: 
        op = ' + '    
    S = S + op + r
    
import statsmodels.api as sm
import statsmodels.formula.api as smf

#--------------- Fitting regression -----------------
#----------------------------------------------------------------------------------
model1 =  smf.glm(S , 
                    data = X_train, 
                    family=sm.families.Binomial())
                    
                    
results1 = model1.fit()
print(results1.summary())



model =  smf.ols(S , data=X_train)

results = model.fit()
print (results.summary())


# selecting top vars

S=  'fraud_flag ~ CLM_BI_TP_IND	+\
CRED_HH_COMPOSITION	+\
ARI1	+\
CLT_SOLICITOR_OWN_IND	+\
TP_MOSAIC_MN	+\
RSPNSBLTY	+\
end_excs	+\
COST_PAID_PD_CLMNT_AMT	+\
DIST_FREQ_AD	+\
AGE_YAD	+\
AGE_PH	+\
CLM_CLMNT_CT	+\
TP_MOSAIC_MN	+\
TP_MOSAIC_MX	+\
COST_EST_PD_AMT	+\
CLT_DRV_IND	+\
CLM_CLMNT_CT	+\
PH_KA_DISTANCE	'

S.replace('\t', ' ')


model =  smf.ols(S , data=X_train)

results = model.fit()
print (results.summary())





model1 =  smf.glm(S , 
                    data = X_train, 
                    family=sm.families.Binomial())
                    
                    
results1 = model1.fit()
print(results1.summary())

















