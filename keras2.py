
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder



def train_model(X,y, Xte, yte):

    scale = StandardScaler().fit(X)	
    X     = scale.transform(X)
    Xte   = scale.transform(Xte)
    
    lb  = LabelEncoder().fit(Y)
    y   = lb.transform(Y)
    yte = lb.transform(Yte)
    
    in_dim = X.shape[1] 
    
    cw = class_weight.compute_class_weight('balanced', np.unique(y), y) 
                           
    model = Sequential()
    model.add(Dense(20,activation='relu', input_shape=(in_dim,)))
    model.add(Dense(10,activation='relu', input_shape=(in_dim,)))
    model.add(Dense(5,activation='relu', input_shape=(in_dim,)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
                       
    h=model.fit(X, y,
              epochs=50, 
              batch_size=10, 
              #validation_data=(Xte, yte),
              validation_split = 0.30,
              class_weight = cw,
              #shuffle=True,
              verbose=2, 
              #callbacks=[tbCallBack] 
              )
    return  model.evaluate(Xte,yte)



dataset = {
	'banana':	100,
	'breast_cancer': 100,
	'diabetis': 100,
	'flare_solar':100,
	'german' : 100,
	'heart':100,
	'image':20,
	'ringnorm':100,
	'splice': 20,
	'thyroid': 100,
	'titanic': 100,
	'twonorm': 100,
	'waveform' : 100
}





np.random.seed(2340)

test_res = []

dataset = 'flare_solar'
path  = "C:\\myData\\"+dataset+"\\"	

for i in range(1,2):
    train = pd.read_csv(path+ dataset+"_train_"+str(i)+".csv", header = None)
    #---------- Train set----------------#
    n, m = train.shape
    # for this code to work faster we need to transform to np.arrays
    X = train.loc[:,:m-2].as_matrix() 	
    Y = train.loc[:,m-1].as_matrix()
    #--------- Test set ----------------#
    test  = pd.read_csv(path+dataset+"_test_"+str(i)+".csv", header = None)
    Xte = test.loc[:,:m-2].as_matrix() 	
    Yte = test.loc[:,m-1].as_matrix()
    
    a = train_model(X,Y, Xte, Yte)
    test_res.append(a)
    print(i, a)

s = [ 1-r[1] for r in test_res]
print(np.mean(s), np.std(s))

results = { dataset: s}

import pickle
 
pickle.



  