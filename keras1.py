# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 20:46:30 2017

@author: ks_work
"""


import numpy as np
import pandas as pd

np.random.seed(2340)

path  = "C:\\myData\\diabetis\\"	
train = pd.read_csv(path+"diabetis_train_1.csv", header = None)

#---------- Train set----------------#
n, m = train.shape
# for this code to work faster we need to transform to np.arrays
X = train.loc[:,:m-2].as_matrix() 	
Y = train.loc[:,m-1].as_matrix()

#--------- Test set ----------------#
test  = pd.read_csv(path+"diabetis_test_1.csv", header = None)
Xte = test.loc[:,:m-2].as_matrix() 	
Yte = test.loc[:,m-1].as_matrix()


from sklearn.preprocessing import StandardScaler, LabelEncoder

scale = StandardScaler().fit(X)	

X = scale.transform(X)
Xte = scale.transform(Xte)

lb = LabelEncoder().fit(Y)
y  = lb.transform(Y)
yte = lb.transform(Yte)


from sklearn.utils import class_weight

cw = class_weight.compute_class_weight('balanced', np.unique(y), y) 

# w1, w0 = 0.5*(1+n1/n0), 0.5*(1 + n0/n1)

from keras.models import Sequential
from keras.layers import Dense


in_dim = X.shape[1] 

                  
#tbCallBack = keras.callbacks.TensorBoard(log_dir='./finishedCode/Graph', 
#                    histogram_freq=0, write_graph=True, write_images=True)

                 
                  
model = Sequential()
# Add an input layer 
model.add(Dense(20,activation='tanh', input_shape=(in_dim,)))
#model.add(Dense(5, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
#from keras.losses import 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
h=model.fit(X, y,
          epochs=20, 
          batch_size=1, 
          #validation_data=(Xte, yte),
         validation_split = 0.30,
          class_weight = cw,
          #shuffle=True,
          verbose=2, 
          #callbacks=[tbCallBack] 
          )

print("test accuracy : ", model.evaluate(Xte,yte))    

import  matplotlib.pyplot as plt
plt.plot(h.history['loss'],label='train loss')
plt.plot(h.history['val_loss'],label='validation loss')
plt.legend()
plt.grid()
plt.show()







