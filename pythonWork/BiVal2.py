# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:12:21 2017

@author: saadik1
"""

import explore  as tls
import pandas as pd
import numpy as np

clms = pd.read_csv("C:\KS_temp\VALDATA.csv")
#print(clms.count())

clms2 = clms.drop(['prognosis2','AGE_BAND','age_band2'], axis=1)

# Some data manipulation
clms2['AGE2']=""
for ind, r in clms2.iterrows():
    #if not (r['AGE'] is None) :
    tmp=None    
    if (r['AGE']<=17): tmp ="0-17"
    elif (r['AGE']>17 and r['AGE']<=60): tmp ="18-60"
    elif (r['AGE']>60): tmp = "60+"
    clms2.set_value(ind,'AGE2',tmp)    
   


#Same effect as above    
'''
def bandAge(age):
    if age<=17: return "0-17"
    elif (age>17 and age<=60): return "18-60"
    elif (age>60): return  "60+"  
    else: return 

clms2['AGE3'] = clms2['AGE'].apply(lambda x : bandAge(x))     
clms2['AGE4'] = clms2.apply(lambda row: bandAge(row['AGE']), axis=1)
'''

#tt = clms2[clms2['claim_number']=='99311BR64758/2']

tls.countMissings(data = clms2, graph=True)


impute_grps = clms.pivot_table(values=["GD_Paid"], index=["injury"],
                               aggfunc=[ np.mean, np.std, np.min, np.max])

print(impute_grps)
     
tls.freq(data=clms, var='N_inj', graph=True)

#converting from string to numeric
clms['prognosis'] = pd.to_numeric(clms['prognosis'], errors='coerce')


tls.freq(data=clms, var='prognosis', graph=True)



tls.freq(data=clms, var='prognosis', graph=True)
tls.freq(data=clms, var='prognosis2', graph=True)

#-------------------------------------------------------------------------------------
#    Modelling Cost
#-------------------------------------------------------------------------------------
import statsmodels.api as sm
import statsmodels.formula.api as smf

clm3 = clms[     (not clms['prognosis'] is None) & 
                 (clms['prognosis']<=18) & 
                 (clms['prognosis']>0) ]
                 
#dumm = pd.get_dummies(clm3['injury'])



#clms.rename(columns={'N_inj': 'N inj'}, inplace=True)                 

#--------------- Fitting regression -----------------
model =  smf.ols('gd_paid2 ~ injury + prognosis + GP_NumVisits +Complications + \
                    Physio_NumVisits  + N_Inj2 + age_band2+acc_year', data=clms)
results = model.fit()
print (results.summary())

#----------------------------------------------------------------------------------
model1 =  smf.glm('gd_paid2 ~ injury + prognosis + GP_NumVisits +Complications + \
                    Physio_NumVisits  + N_Inj2 + age_band2+acc_year', 
                    data = clms, 
                    family = sm.families.Gaussian(sm.families.links.identity))                    
results1 = model1.fit()
print(results1.summary())


# core modelling daat
modelData = pd.DataFrame(data=model1.exog, columns=model1.exog_names)
modelTarget = model1.endog






#---------------------------------------------------------------------------------


stats.anova_lm(model1, typ=2)


import matplotlib.pyplot as plt

plt.hist(clms['gd_paid2'], bins=40, facecolor='lightgreen', normed=1) 
plt.title('GD_paid')


plt.hist(clms['prognosis2'],facecolor='lightgreen',bins=18) 
plt.title('Prognosis')


D = clms['AGE'][clms['AGE'].notnull()]
plt.hist(D,facecolor='lightgreen',bins=50) 


plt.show(clm2.boxplot(column="GD_Paid",by="injury"))
plt.show(clm2.boxplot(column="GD_Paid",by="prognosis")) # need to convert prognosis to numeric







