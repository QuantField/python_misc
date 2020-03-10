# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 10:15:25 2017

@author: saadik1
"""

Xtr = finalModData
ytr = y_train



std = Xtr.std().as_matrix()
spheSigma = np.sqrt(np.dot(std,std))

sigmaTune  = np.logspace(-2, 1.5, 15)*spheSigma
gamma_range = 1/sigmaTune**2
C_range = np.logspace(-3, 5, 15)


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

gamma = 1/spheSigma**2
C_range = np.logspace(-3, 5, 5)
param_grid = dict( C=C_range)
cv = StratifiedShuffleSplit(y=ytr, n_iter=5, test_size=0.2) # 5 cv
grid = GridSearchCV(SVC(gamma=gamma), param_grid=param_grid, cv=cv)
grid.fit(Xtr, ytr)





#-------------- Crosss validation tunning for Adaboost ---------------
#--------- Default base algorithm is decision tree -------------------
learning_rate = np.linspace(start= 0.01,stop= 0.1, num =10)
n_estimators = [10*(r+1) for r in range(6)]
# n_estimators doesn't have a lot of impact
#param_grid = dict( learning_rate=learning_rate, n_estimators = np.array(n_estimators, int))
param_grid = dict( learning_rate=learning_rate)
cv = StratifiedShuffleSplit(y=ytr, n_iter=5, test_size=0.2) # 5 cv
#'entropy' default gini
dtree = DecisionTreeClassifier(max_depth=4, random_state=3004)
mlAlgorithm = AdaBoostClassifier(base_estimator=dtree) # defautl in adaboost max_depth=1
grid = GridSearchCV(mlAlgorithm, param_grid=param_grid, cv=cv,  verbose=1)
grid.fit(Xtr, ytr)
bestModel = grid.best_estimator_



#-------------- Crosss validation tunning for nearest neighbour ---------------
n_neighbors =np.array([3,4,5,6,7,8,9,10], int)
param_grid = dict( n_neighbors=n_neighbors)
cv = StratifiedShuffleSplit(y=ytr, n_iter=5, test_size=0.2) # 5 cv
grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=cv,  verbose=1)
grid.fit(Xtr, ytr)
bestModel = grid.best_estimator_
#------------------------------------------------------------------------------

#-------------- Crosss validation tunning for decision tree -------
max_depth = np.array([2,3,4,5,6,7,8,9,10], int)
param_grid = dict( max_depth=max_depth)
dtree   = DecisionTreeClassifier()
cv = StratifiedShuffleSplit(y=ytr, n_iter=10, test_size=0.1) # 10 cv
grid = GridSearchCV(dtree, param_grid=param_grid, cv=cv,  verbose=1)
grid.fit(Xtr, ytr)
bestModel = grid.best_estimator_


#----------------------------------------------

svm = SVC(C=10, verbose=True)
svm.fit(Xtr, ytr)

#------------------------------ SVM tuning ---------------------------------
param_grid = dict(gamma=gamma_range, C=C_range)
#  cv default is 10 cross validation (n_iter=10, test_size=0.1)
cv = StratifiedShuffleSplit(y=ytr, n_iter=5, test_size=0.2) # 5 cv
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(Xtr, ytr)

bestSvm = grid.best_estimator_









