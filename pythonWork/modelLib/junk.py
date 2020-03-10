# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:05:15 2017

@author: saadik1
"""


model1 = svm.fit(X,y_train)
yhat = model3.predict(X)
r = classAccuracy(y_train.as_matrix(), yhat, disp=True)
#p = diag.svmTargetMisc(y,yhat)
Xt = testData
pred1 = model3.predict(Xt)
r = classAccuracy(y_test.as_matrix(), pred1, disp=True)





model2 = logreg.fit(X_train[modVars],y_train)
yhat = model2.predict(X)
p = diag.svmTargetMisc(y,yhat)
pred1 = model2.predict(Xt)
p1 = diag.svmTargetMisc(yt,pred1)
print("LOGISTIC : Train = %2.2f   Test = %2.2f"%(p,p1))




#let try with standarization
stdTrain = standarizeObj(X_train)
stdTrain.standarize()
stdTrain.standarizedNew()

score2, varList2 = variablesSelection(X_train, y_train) # same as expected











from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif

data = X_train
num=[] # will be filled with numerical variables
cat=[] # will be filled with categorical variables
for r in data.columns:
    typ = data[r].dtype
    if typ == np.float64 or typ==np.int64:
        num.append(r)
    else:
        cat.append(r)
#--------------- Numerical Variables ------------------------#    
# even if we use f_regression it will use f_classif
# as it detects that the target is categorical
Selector_f = SelectPercentile(f_regression, percentile=10)
Selector_f.fit(X_train[num],y_train)
vars1 =pd.DataFrame(Selector_f.scores_, columns=['F_Score'], index=data[num].columns)


from sklearn.feature_selection import GenericUnivariateSelect


Selector_f = GenericUnivariateSelect(f_classif, mode ='k_best')
Selector_f.fit(X_train[num],y_train)
vars1 =pd.DataFrame(Selector_f.scores_, columns=['Scores'], index=data[num].columns)























#------------------------------------------------------------------------
#------------------------------------------------------------------------
varNames = allData.columns;


posData = allData[(allData.fraud_flag ==1)]
negData = allData[~allData.index.isin(posData.index)]


neg1 = negData.sample(n=2*posData.shape[0], random_state = 564) 

train1 = posData.append(neg1)



t1 =train1[
['TP_AFI_MN'	,
'TP_AFI_MX'	,
'TP_AGE_MN'	,
'TP_AGE_MX'	,
'TP_DEPRIV_INDEX_MN'	,
'TP_DEPRIV_INDEX_MX'	,
'TP_DEPR_IND_BAND_MN'	,
'TP_DEPR_IND_BAND_MX'	,
'TP_DIST_COST_AD_MN'	,
'TP_DIST_COST_AD_MX'	,
'TP_DIST_COST_BI_MN'	,
'TP_DIST_COST_BI_MX'	,
'TP_DIST_COST_PD_MN'	,
'TP_DIST_COST_PD_MX'	,
'TP_DIST_FREQ_AD_MN'	,
'TP_DIST_FREQ_AD_MX'	,
'TP_DIST_FREQ_BI_MN'	,
'TP_DIST_FREQ_BI_MX'	,
'TP_DIST_FREQ_PD_MN'	,
'TP_DIST_FREQ_PD_MX'	,
'TP_KA_DISTANCE_MN'	,
'TP_KA_DISTANCE_MX'	,
'fraud_flag']]


freq(t1,'fraud_flag')


#=================================================================
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


t2 = t1.dropna()
freq(t2,'fraud_flag')
Nvars =t2.shape[1]
fd_data  = t2[list(range(Nvars-1))]
fd_target = t2[[-1]] 

Selector_f = SelectPercentile(f_regression, percentile=25)
Selector_f.fit(fd_data,fd_target)

vars =pd.DataFrame(Selector_f.scores_, columns=['F_Score'], index=fd_data.columns)
print(vars.sort_values(by='F_Score', ascending=False))




from sklearn.svm import SVC
estimator = SVC(kernel="linear")
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(fd_data,fd_target)
selector.support_ 








#=================================================================


t1= t1.dropna()

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

fd_data  = t1.drop(['fraud_flag'],1) 
fd_target = t1['fraud_flag'] 

nbmodel = gnb.fit(fd_data, fd_target)
 

import diagnostics as diag 
 
pred = nbmodel.predict(fd_data)
prob = nbmodel.predict_proba(fd_data)[:,1]
act  = fd_target.as_matrix()



auc,fpr,tpr = diag.AUROC(act,prob)
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)


result = diag.classAccuracy(act,pred,True)
print("ACC=%2.2f  SENS=%2.2f  PREC=%2.2f" % result )
#c4 = pd.get_dummies(v1,dummy_na = True, prefix ='DAMSEV')

#------ Using SVM's ----------

t1 =train1[
['TP_AFI_MN'	,
'TP_AFI_MX'	,
'TP_AGE_MN'	,
'TP_AGE_MX'	,
'TP_DEPRIV_INDEX_MN'	,
'TP_DEPRIV_INDEX_MX'	,
'TP_DEPR_IND_BAND_MN'	,
'TP_DEPR_IND_BAND_MX'	,
'TP_DIST_COST_AD_MN'	,
'TP_DIST_COST_AD_MX'	,
'TP_DIST_COST_BI_MN'	,
'TP_DIST_COST_BI_MX'	,
'TP_DIST_COST_PD_MN'	,
'TP_DIST_COST_PD_MX'	,
'TP_DIST_FREQ_AD_MN'	,
'TP_DIST_FREQ_AD_MX'	,
'TP_DIST_FREQ_BI_MN'	,
'TP_DIST_FREQ_BI_MX'	,
'TP_DIST_FREQ_PD_MN'	,
'TP_DIST_FREQ_PD_MX'	,
'TP_KA_DISTANCE_MN'	,
'TP_KA_DISTANCE_MX'	,
'fraud_flag']]

t1= t1.dropna()
fd_data  = t1.drop(['fraud_flag'],1) 
fd_target = t1['fraud_flag'] 




from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(fd_data, fd_target) 

pred   = clf.predict(fd_data)
result = diag.classAccuracy(t1['fraud_flag'],pred,True)

)







from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn  import grid_search 
from sklearn import cross_validation



C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = grid_search(SVC(), param_grid=param_grid, cv=cv)
grid.fit(fd_data, fd_target)





from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

from sklearn.model_selection import GridSearchCV


import sklearn.cross_validation.StratifiedShuffleSplit


C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)


clf = svm.SVC(kernel='linear', C = 1.0)







