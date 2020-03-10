import pandas as pd
import numpy as np

clms = pd.read_csv("C:\KS_temp\VALDATA.csv")
#print(clms.count())

clms2 = clms.drop(['prognosis2','AGE_BAND','age_band2'],1)

# Some data manipulation
clms2['AGE2']=""
for ind, r in clms2.iterrows():
    #if not (r['AGE'] is None) :
    if (r['AGE']<=17):
        tmp ="0-17"
    elif (r['AGE']>17 and r['AGE']<=60):
        tmp ="18-60"
    elif (r['AGE']>60):
        tmp = "60+"
    clms2.set_value(ind,'AGE2',tmp)    

    
tt = clms2[clms2['claim_number']=='99311BR64758/2']




# counting the number of missings and the percentages
def numOfmissings(x):
    nmiss = sum(x.isnull());
    return (nmiss, nmiss*100.0/len(x))

#print(clms["prognosis"].apply(numOfmissings))

print(clms.apply(numOfmissings, axis=0)) #axis=0 defines that function is to be applied on each column


clms.drop(column='prognosis2')


def freq(x):
    return len(x)

impute_grps = clms.pivot_table(values=["GD_Paid"], index=["injury"], aggfunc=[freq, np.mean, np.std, np.min, np.max])

print(impute_grps)

# equiv SAS proc freq
def myFreq(data , var ):
     dc = data[var].value_counts(dropna = False).sort_index()
     dc.name = '        Freq'
     dc2 = (dc/dc.sum())*100
     #dc2 = dc2.map('{:,.2f}'.format)
     dc2.name = 'Perc'
     dc3 = dc.cumsum()
     dc3.name ='CumFreq'
     dc4 = dc2.cumsum()
     #dc4 = dc4.map('{:,.2f}'.format)
     dc4.name = 'CumPerc'
     dc2 = dc2.map('{:,.2f}'.format)
     dc4 = dc4.map('{:,.2f}'.format)
     print("------------------------------------------")
     print(pd.concat([dc, dc2, dc3, dc4], axis=1, join='inner'))
     print("------------------------------------------")
     
myFreq(data=clms, var='N_inj')
#converting from string to numeric
clms['prognosis'] = pd.to_numeric(clms['prognosis'], errors='coerce')

#------------------------- some calculations ----------------------------------
# Matix Values
mat_N_B ={
1	:	1100,
2	:	1250,
3	:	1500,
4	:	2000,
5	:	2100,
6	:	2450,
7	:	2600,
8	:	2750,
9	:	2850,
10	:	2950,
11	:	3000,
12	:	3200,
13	:	3400,
14	:	3700,
15	:	3800,
16	:	4100,
17	:	4300,
18	:	4500,
19	:	4600}
mat_NB = {
1	:	1250,
2	:	1800,
3	:	1950,
4	:	2100,
5	:	2200,
6	:	2600,
7	:	2750,
8	:	2900,
9	:	3000,
10	:	3150,
11	:	3250,
12	:	3450,
13	:	3600,
14	:	3850,
15	:	4100,
16	:	4400,
17	:	4700,
18	:	4800,
19	:	4900}
def calculateMat(x,y):
    if (x in ('NM','BM','N','B')):
        return mat_N_B[y]
    elif (x in ('NBM','NB')):
        return mat_NB[y]
    else:
       return 0	
      
# calculating matrix values       
clms['MatrixValue'] = clms.apply(lambda row : calculateMat(  row['injury'], row['prognosis2']), axis=1 )


#-------------------------------------------------------------------------------------
#    Modelling Cost
#-------------------------------------------------------------------------------------
import statsmodels.api as sm
import statsmodels.formula.api as smf

clm3 = clms[     (not clms['prognosis'] is None) & 
                 (clms['prognosis2']<=18) & 
                 (clms['prognosis2']>0) ]
                 
#dumm = pd.get_dummies(clm3['injury'])



                 
#--------------- Fitting regression -----------------
model =  smf.ols('gd_paid2 ~ injury + prognosis2 + GP_NumVisits +Complications + \
                    Physio_NumVisits  + N_Inj2 + age_band2+acc_year', data=clm3)
results = model.fit()
print (results.summary())

#----------------------------------------------------------------------------------
model1 =  smf.glm('gd_paid2 ~ injury + prognosis2 + GP_NumVisits +Complications + \
                    Physio_NumVisits  + N_Inj2 + age_band2+acc_year', 
                    data = clm3, 
                    family = sm.families.Gaussian(sm.families.links.identity))                    
results1 = model1.fit()
print(results1.summary())


pred = model1.mu # prediction

C = model1.data






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







