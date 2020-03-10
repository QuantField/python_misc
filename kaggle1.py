# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 17:31:00 2018

@author: ks_work
"""

IMG_WIDTH    = 28
IMG_HEIGHT   = 28
IMG_CHANNELS = 1
TRAIN_PATH = 'C:/myData/Kaggle/train.csv'
TEST_PATH  = 'C:/myData/Kaggle/test.csv'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_tr = pd.read_csv(TRAIN_PATH)
data_te = pd.read_csv(TEST_PATH)
label = data_tr.pop('label')
n_tr = data_tr.shape[0]
n_te = data_te.shape[0]
print(n_tr,n_te)

tr = np.zeros((n_tr,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))
for i in range(n_tr):
  s = data_tr.loc[i,:]
  tr[i,:,:,0] = s.values.reshape(IMG_HEIGHT,IMG_WIDTH)
  
te = np.zeros((n_te,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))
for i in range(n_te):
  s = data_te.loc[i,:]
  te[i,:,:,0] = s.values.reshape(IMG_HEIGHT,IMG_WIDTH)
  
x_train = tr.astype(np.float32)
x_test  = te.astype(np.float32)

x_train /= 255
x_test  /= 255


idx = np.random.randint(n_tr)
im = np.squeeze(x_train[idx,:,:,0])
plt.gray()
plt.grid('off')
plt.imshow(im)
print(label[idx])

from keras.models import load_model, Sequential
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import keras.utils

num_classes = len(np.unique(label))
print(num_classes)
y_train = keras.utils.to_categorical(label, num_classes=None)
print(y_train.shape)



earlystopper = EarlyStopping(patience=10, verbose=1)

model_name = 'model_'+str(np.random.randint(99999999))+'.h5'
checkpointer = ModelCheckpoint('C:/myData/Kaggle/'+ model_name, verbose=1, save_best_only=True)



model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), padding='same',activation='relu', 
                     input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, kernel_size=(3, 3), padding='same',activation='relu', 
                     input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same',activation='relu', 
                     input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=2000,
          epochs=100,# need to use 100
          verbose=1,   
          shuffle=True,
          validation_split=0.3,
          callbacks=[earlystopper, checkpointer]
          ) 

#earlystopper = EarlyStopping(patience=10, verbose=1)
#
#model_name = 'model_'+str(np.random.randint(99999999))+'.h5'
#checkpointer = ModelCheckpoint('C:/myData/Kaggle'+ model_name, verbose=1, save_best_only=True)
#
#
#
#model = Sequential()
#
#model.add(Conv2D(64, kernel_size=(3, 3), padding='same',activation='relu', 
#                     input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#
#model.add(Conv2D(10, kernel_size=(4, 4), padding='same',activation='relu', 
#                     input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#
#
#model.add(Flatten())
#
#model.add(Dense(512, activation='relu'))
#
#model.add(Dense(10, activation='relu'))
#
#model.add(Dense(num_classes, activation='softmax'))
#
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer='adam',
#              metrics=['accuracy'])
#
#model.summary()
#
#model.fit(x_train, y_train,
#          batch_size=1000,
#          epochs=50,
#          verbose=1,   
#          #shuffle=True,
#          validation_split=0.3,
#          callbacks=[earlystopper, checkpointer]
#          ) 

net = load_model('C:/myData/Kaggle/'+ model_name)
pred = np.argmax(net.predict(x_train),axis=1)
acc = sum(np.equal(pred,label))/len(label)
print(acc)

pred = np.argmax(net.predict(x_test),axis=1)
res = pd.DataFrame({'ImageId':range(1,n_te+1),'Label':pred })
print(res[:10])
res.to_csv('k.saadi_res_feb2018_111.csv', index = False)

