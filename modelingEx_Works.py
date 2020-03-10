"""
This is on way of getting data ready for modelling
Not working yet as it needs sorting out. final matrix has problems.

"""

import pandas  as pd
import explore as mt
import numpy   as np

#------------------------------------------------------------------------------#
#--------------------------- Data ---------------------------------------------#
sasLibname = "C:\\Users\\ks_work\\Desktop\\Code\\pyData\\"
BiData = pd.read_sas( sasLibname +"bidatapersfraud.sas7bdat", encoding='utf-8')


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


#----- years only from 1990 to 2017
stryears = [ str(r) for r in  range(1990,2018) ]          
var      = 'Incident_Veh_Yr_Made'    
BiData[var] = BiData[var].apply(lambda x:  x if x in stryears else None )  ;  

       
#--------  changing the ranges of some variables ----------#
# --- X<minR --> X = minR, X>maxR --> X = maxR
def changeRange(data, var, maxR, minR=-np.Infinity ):
    data.loc[(data[var]>maxR),var]=maxR
    if minR!=np.Infinity :	data.loc[(data[var]<minR),var]=minR
"""-----------------------------------------
16<=AGE_ <=94  
TP_Number_Of_Passengers<=10
CLM_CLMNT_BI_CT<=5
CLM_CLMNT_CT  <=10
16<=AGE_ <=94  
VEH_AGE_ACC  <=40
------------------------------------------"""
changeRange(BiData,'AGE_PH',95)
changeRange(BiData,'AGE_YAD',95,16)
changeRange(BiData,'AGE_MD',95,16)
changeRange(BiData,'TP_Number_Of_Passengers',7,0)
changeRange(BiData,'CLM_CLMNT_BI_CT',7,0)
changeRange(BiData,'CLM_CLMNT_CT',10,0)
changeRange(BiData,'VEH_AGE_ACC',30,0)

# ------------------ Adding the target variables ------------------------#

frauds = pd.read_sas( sasLibname + "data_fr.sas7bdat", encoding='utf-8')
frauds['fraud_flag']=1.0

#------ merging data --------
allData = pd.merge(BiData, frauds, how='left', left_on='CLM_CUST_NO',right_on ='claim_no')
allData['fraud_flag'].fillna(0.0,inplace = True)


target = 'fraud_flag'

mt.freq(data=allData, var =target ,graph =True)

data = allData # copy by reference

# ----------------------------------------------------------------------  #
#     create lists of numeric ana categorical varialbles   for modelling  #
# ----------------------------------------------------------------------  #
num_vars, cat_vars = [],[] 
for r in data.columns:
    ln = len(data[r].unique())
    if ln>1 : # number of values per variable should be>1    
        if data[r].dtype in (np.float64, np.int64):
            num_vars.append(r)
        else:
            cat_vars.append(r) 

num_vars.remove(target) # target is not processed with the rest of variables


# --------------------- Missing --------------------------------------- #               
#missing for num vars
print("numerical variabls")
mt.countMissings(data[num_vars], graph=True)    
# from graph we need to ..
for r in  ['TP_VEH_AGE_ACC', 'Impact_speed']: num_vars.remove(r)
#--------
print("categorical variabls")
mt.countMissings(data[cat_vars], graph=True)    
# more to remove 
miss = ['TP_Post_Code',               
'Inci_Liab_Discussion',       
'Transfer_To_ULR_Flag',       
'FP_Passengers',              
'Injured_Party_Injury_Type',  
'Injured_Party_Type',         
'inj_incident_number',        
'claim_no',                   
'ARI2',                       
'Theft_Of_Veh_Recover',       
'Any_Items_Stolen']       
for r in set(miss) & set(cat_vars): 
    cat_vars.remove(r)    
    
# drop categorical variables with levels higher than 20
_MAX_CAT_LEVELS_ = 30 
print("\nDropping categorical variables with high number of levels: ")
#tt = cat_vars.copy() # default copy method is by reference only
for r in cat_vars.copy():
     ln = len(data[r].unique())
     if ln >_MAX_CAT_LEVELS_:
         print(r)
         cat_vars.remove(r)

#convet numerical variable to categorical if the levels are fewer the specified
print("\nConvert numerical varialbes to categorical")
for r in num_vars.copy():
     ln = len(data[r].unique())
     if ln <= 3 : #_MAX_CAT_LEVELS_
         print(r)
         num_vars.remove(r)
         cat_vars.append(r)
         # transforming type to str and nan to None
         data[r] = data[r].astype(str)
         data.loc[(data[r]=='nan'),r]=None       

# remove these fields ... cost with few levels not interesting
for r in {'COST_EST_OTH_AMT', 'COST_PAID_OTH_CLMNT_AMT'} & set(cat_vars):
    cat_vars.remove(r) 

print("\nVarialble to use ")
print("*--------numerical---------*")
for r in num_vars: print(r)
print("*--------categorical-------*")
for r in cat_vars: print(r)

#----------------------------------------------------------------#
#------- processing variables and creating modelling data -------#
#----------------------------------------------------------------#

data = allData[num_vars + cat_vars + [target]] # this is a deep copy
#--------------------------Numeric Values -----------------------#
# possible imputing methods
Mean   =  data[num_vars].mean()
Std    =  data[num_vars].std()
Median =  data[num_vars].median()
# mode doesn't return pandas Series hence the slight difference
Mode   =  data[num_vars].mode().T[0] 

#---- removing extrem values +/- 3STD
print("\n extrem value processing mean +/- 3std")
#---------------------------------------------------

for var in num_vars:
    low, high = Mean[var]-3*Std[var], Mean[var]+3*Std[var]
    if sum(data[var]<low)>0 : 
        print(var," capped to ", low)
        data.loc[(data[var]<low),var]=low
    if sum(data[var]>high)>0:  
        print(var," capped to ", high)
        data.loc[(data[var]>high),var]=high
    

#---------        Filling misssing values            ------------#  
miss_values = dict(Median) # We choose to fill na with median
cMode     =  data[cat_vars].mode().T[0] # and mode for categorical
miss_values.update(dict(cMode)) # I pefer to use dict instead of dataframe


print("filling missing values")
data.fillna(miss_values, inplace = True)




for r in num_vars:
    print(r + "  -->  "+ str(ff[r]) )
    data[r].fillna(ff[r], inplace = True)

#------------------- standarisation -----------------------------#
print()
print("standarise variables")
for r in num_vars:
    print(r)
    mn , sd = Mean[r], Std[r]
    data.loc[:,r] = (data[r]-mn)/sd

#----------------------- Categorical Values ---------------------#

ff = cMode     =  data[cat_vars].mode().T[0]
print("categorical data :filling missing values")
for r in cat_vars:
    print(r + "  -->  "+ str(ff[r]) )
    data[r].fillna(ff[r], inplace = True)
    # in case we want to create an extra level _MISS_
    #data[r].fillna('_MISS_', inplace = True)
    # miss can be generated automatical with pd.
    # with dummy_na=True

#---------- OverSampling data for modelling ------------------------#

data_mod = data

data_mod_p = data_mod[(data_mod[target]==1)]
data_mod_n = data_mod[(data_mod[target]==0)]
N_Pos      = data_mod_p.shape[0]
N_Neg      = 3*N_Pos # oversample fraud rate 0.25
negData    = data_mod_n.sample(n =N_Neg, random_state=3254)
ovSample1  = data_mod_p.append(negData)

mt.freq(ovSample1, target, graph=True)

Y = ovSample1.pop(target)
X = pd.get_dummies(ovSample1, prefix_sep = '$', drop_first=True)

#--------------- variable selection ------------------------------#

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

Selector_f = SelectPercentile(f_classif, percentile=10)
Selector_f.fit(X,Y)
tmp0    = pd.DataFrame(Selector_f.scores_, 
                        columns=['F_Score'], 
                        index  = X.columns )
tmp1      = list(tmp0[(tmp0['F_Score']>50)].index)
selVars  = list( { r[0:r.rfind('$')] if r.rfind('$')>0 else r for r in tmp1} )

# ----------- Decision tree --------------------------------

from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix 
from sklearn.tree import DecisionTreeClassifier

clf   = DecisionTreeClassifier(max_depth=5)
clf.fit(X,Y)

prec, rec, fbeta, supp = precision_recall_fscore_support(Y, clf.predict(X))
# this is the usual stuff we are interested in (rec for recall is same as sensitivity)
Prec, Sens, F1score = prec[1], rec[1], fbeta[1] # F1 score = 2*Prec*Sens/(Prec+Sens)
#----- if we use index 0 above we get same stats for class 0 
#------------- Confusion Matrix ----------------------
conf = confusion_matrix(Y, clf.predict(X)) 
[TN, FN], [FP, TP] = conf[0], conf[1]


import sklearn.metrics as sk
Yhat = clf.predict(X)
fpr, tpr, thresholds = sk.roc_curve(Y, Yhat)
auroc = sk.auc(fpr, tpr)


#---- cross validation ----------
cvValue = cross_val_score(clf, X, Y, cv=10)

# coef of deterimination ( R2 for reg)
r2 = clf.score(X,Y)  

#---- We can select variables from the dtree;

tVars = pd.DataFrame( data = list(zip(X.columns,clf.feature_importances_)), 
                      columns=['Var','Importance'])

#------ Compare tVars to selVars 
  
  
#------------ Another classifier --------------------------     

from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth"        : [3,5,7],    
    "criterion"        : ["gini", "entropy"]
}


clf0 = DecisionTreeClassifier()

gs = GridSearchCV(clf0, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')
gs.fit(X, Y)         
         
         
         
         
         
         
         




#----------------------------------------------------------
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals       import joblib




param_grid = {
    "max_depth"        : [5],
    "max_features"     : [3,4,10],
    "min_samples_split": [2, 3, 10],
    "min_samples_leaf" : [1, 3, 10],
    "bootstrap"        : [True, False],
    "criterion"        : ["gini", "entropy"]
}


Y = ovSample1.pop(target)
X = pd.get_dummies(ovSample1, prefix_sep = '$', drop_first=True)


clf = RandomForestClassifier()
gs = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1, verbose=True, scoring='roc_auc')
gs.fit(X, Y)



featImp = gs.best_estimator_.feature_importances_

featSort = pd.DataFrame(list(zip(X.columns, featImp))).sort_values([1], ascending=False)

# keeping vars with importance > 0.01 
selVars  = list(featSort[(featSort[1]>1e-2)][0])
# clean variable names 
selVars  = {r[0:r.rfind('$')]  for r in selVars}


##----- Another way(easy) to select variables



"""

X = pd.get_dummies(data_mod, prefix_sep='$', drop_first=True)    
Y = X.pop(target)

clf   = RandomForestClassifier(max_depth=5)
clf.fit(X, Y)
# collecting variable importance 
impor = pd.DataFrame(data = clf.feature_importances_, index = X.columns)

from sklearn.metrics import accuracy_score

acc = accuracy_score(Y, clf.predict(X))


param_grid = {
    "max_depth"        : [3, None],
    "max_features"     : [1, 3, 10],
    "min_samples_split": [2, 3, 10],
    "min_samples_leaf" : [1, 3, 10],
    "bootstrap"        : [True, False],
    "criterion"        : ["gini", "entropy"]
}

clf = RandomForestClassifier()
gs = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1, verbose=True, scoring='roc_auc')
gs.fit(X, Y)
"""




   
         
         
         







