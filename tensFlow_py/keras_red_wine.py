# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------
MLP with tensorflow and keras
keras is build up either TensorFlow or Theano.

Here is an example on how to use keras and TensorFlow.
keras simpler, but TensorFlow give access to more details.

Dr Kamel Saadi
- 21/12/2017

Data can be download from 

http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
( winequality-white.csv, winequality-red.csv )
----------------------------------------------------------------------
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datFile = "C:\\Users\saadik1\\Downloads\\winequality-white.csv"
white = pd.read_csv(datFile, sep=';', header=0)
white['type']='white'
datFile = "C:\\Users\saadik1\\Downloads\\winequality-red.csv"
red = pd.read_csv(datFile, sep=';', header=0)
red['type']='red'
comb =red.append(white)

#----------------------- few graphs ----------------------------
plt.scatter(red['quality'], red['sulphates'])

np.random.seed(123)

redlabels = np.unique(red['quality'])

redcolors = np.random.rand(6,4)

for i in range(len(redcolors)):
    redy = red['alcohol'][red.quality == redlabels[i]]
    redx = red['volatile acidity'][red.quality == redlabels[i]]
    plt.scatter(redx, redy, c=redcolors[i])   
plt.title("Red Wine")
plt.xlim([0,1.3])
plt.ylim([7,15])
plt.xlabel("Volatile Acidity")
plt.ylabel("Alcohol")
plt.legend(redlabels, loc='best', bbox_to_anchor=(1.3, 1))
plt.show()

import seaborn as sns
corr = red.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.show()
#---------------------------------------------------------------

#----------- Preprocessing the data ============================#  
# quality is excluded 
X = comb.iloc[:,:-2]
Y = comb.iloc[:,-1]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

lb  = LabelEncoder().fit(Y)
y   = lb.transform(Y)

Xtr,Xte,ytr,yte = train_test_split(X,y, test_size = 0.33, 
                                     random_state = 234)
scaler = StandardScaler().fit(Xtr)
# if we want the value scaler.mean_, scaler.scale_ this is the std
Xtr = scaler.transform(Xtr)
Xte = scaler.transform(Xte) # use same scaler

                      
#========================== Keras ============================#                       
                      
from keras.models import Sequential
from keras.layers import Dense
import keras

in_dim = Xtr.shape[1] # 11

                  
#tbCallBack = keras.callbacks.TensorBoard(log_dir='./finishedCode/Graph', 
#                    histogram_freq=0, write_graph=True, write_images=True)

                 
                  
model = Sequential()
# Add an input layer 
model.add(Dense(12,activation='relu', input_shape=(in_dim,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#from keras.losses import 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
h=model.fit(Xtr, ytr,
          epochs=20, 
          batch_size=1, 
          #validation_data=(Xte, yte),
          validation_split = 0.30,
          #shuffle=True,
          verbose=2, 
          #callbacks=[tbCallBack] 
          )
# if use of tensorboard incomment collbacks=...
print("test accuracy : ", model.evaluate(Xte,yte))    

plt.plot(h.history['loss'],label='train loss')
plt.plot(h.history['val_loss'],label='validation loss')
plt.legend()
plt.grid()
plt.show()




#===================== TensorFlow ============================# 

import tensorflow as tf

in_dim  = Xtr.shape[1]
n_class = 2

x_  = tf.placeholder(tf.float32,[None,in_dim])
y_  = tf.placeholder(tf.float32,[None,n_class])

# weights 
w = {
   'h1': tf.Variable(tf.truncated_normal([in_dim, 12])),
   'h2': tf.Variable(tf.truncated_normal([12, 8])),       
   'out' : tf.Variable(tf.truncated_normal([8,n_class]))
}   
# biases
b = {
   'h1': tf.Variable(tf.truncated_normal([12])),
   'h2': tf.Variable(tf.truncated_normal([8])),       
   'out' : tf.Variable(tf.truncated_normal([n_class]))
}   

def mlp_model(x,w,b):
    layer1  = tf.nn.relu(tf.add(tf.matmul(x,w['h1']),b['h1']))    
    layer2  = tf.nn.relu(tf.add(tf.matmul(layer1,w['h2']),b['h2']))    
    layer_o = tf.nn.sigmoid(tf.add(tf.matmul(layer2,w['out']),b['out']))  
    return layer_o

# ----------- hot encoding ----------------------------
ytr_h = np.hstack((ytr.reshape(-1,1),1-ytr.reshape(-1,1)))    
yte_h = np.hstack((yte.reshape(-1,1),1-yte.reshape(-1,1))) 

model = mlp_model(x_,w,b)

eta = 0.01
loss_func   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                              logits=model, labels=y_))
train_steps = tf.train.AdamOptimizer(eta).minimize(loss_func)
corr_pred   = tf.equal(tf.argmax(model,1),tf.argmax(y_,1))
accuracy    = tf.reduce_mean(tf.cast(corr_pred,tf.float32))

#tf.train.batch(batch_size=20)

init = tf.global_variables_initializer()
        
sess = tf.Session()
sess.run(init)

epochs=200

#------- to visualise ...
logs_path = "C:\\Users\\saadik1\\Desktop\\pythonWork\\finishedCode\\tfGraph\\"
writer = tf.summary.FileWriter(logs_path, sess.graph)
# after the session is closed  go tot he directory above 'tfGraph'
# type tensorboard --logdir=tfGraph
# http address 0.0.0.0:6006 will display the graph

for epoch in range(epochs):
    
    sess.run(train_steps, feed_dict={x_: Xtr, y_: ytr_h })    
    
    loss    = sess.run(loss_func, feed_dict={x_: Xtr, y_: ytr_h })
    
    pred_tr = sess.run(model, feed_dict={x_: Xtr, y_: ytr_h })
    acc_tr  = sess.run(accuracy, feed_dict={x_: Xtr, y_:ytr_h} )    
    
    tePred  = sess.run(model, feed_dict={x_:Xte})
    acc_te  = sess.run(accuracy, feed_dict={x_: Xte, y_:yte_h} )    

    print("epoch = ", epoch,"loss =",loss," tr_acc= ", acc_tr, " test_acc= ", acc_te)

sess.close()

        



