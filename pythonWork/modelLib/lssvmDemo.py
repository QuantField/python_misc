

import lssvm as ls


import pandas as pd
import sklearn.metrics as sk


allData = pd.read_csv("C:\\KS_temp\\banana_data1000.csv")


trainData = allData.sample(frac=0.7, random_state = 564)        
testData  = allData.loc[~allData.index.isin(trainData.index)] 


y = trainData['Class']
x = trainData[['X','Y']]

net = ls.LSSVM( kern=ls.Polynomial(2,0.1), mu = 0.1)

print("---------- Diagnostics -------------")

print("----------Train Partition-----------")
net.train(x.as_matrix(),y.as_matrix())
print("----------Test Partition-----------")

yt = testData['Class']
xt = testData[['X','Y']]
(pred,prob,sign) = net.predict(xt.as_matrix())
net.errStats(yt,pred)

print("------Diagnostics after optimising -------------")
print("----------Train Partition-----------")
net.setOptimalRegularParam()
net. optimRetrain()
print("----------Test Partition-----------")
(pred,prob,sign) = net.predict(xt.as_matrix())
net.errStats(yt,pred)

print("------ Full optimisation RBF -------------")
print("------ Train partition ------------")
optNet = net.optRBF()
print("----------Test Partition-----------")
(pred,prob,sign) = optNet.predict(xt.as_matrix())
optNet.errStats(yt,pred)

print("")
import matplotlib.pyplot as plt
print("---ploting LOO residuals---")
print("---LOO residuals before full optim ---")
looR, press = net.looResiduals()
plt.plot(looR,'b.')
plt.show()
print("---LOO residuals after full optim ---")
looR2, press2 = optNet.looResiduals()
plt.plot(looR2,'b.')
plt.show()

#-------------- ROC curves -----------------
import diagnostics as diag
[auc1, fpr1, tpr1] = diag.AUROC(optNet.y,optNet.yhat)
[auc2, fpr2, tpr2] = diag.AUROC(yt,pred)
plt.plot(fpr1,tpr1,'b-',label='Train')
plt.hold()
plt.plot(fpr2,tpr2,'r-',label='Test')
plt.legend(loc='upper right', shadow=True)
plt.show()

print("------Diagnostics after optimising -------------")
print("----------Train Partition-----------")
net.setOptimalRegularParam()
net. optimRetrain()
print("----------Test Partition-----------")
(pred,prob,sign) = net.predict(xt.as_matrix())
net.errStats(yt,pred)

print("------ Full optimisation RBF -------------")
print("------ Train partition ------------")
optNet = net.optRBF()
print("----------Test Partition-----------")
(pred,prob,sign) = optNet.predict(xt.as_matrix())
optNet.errStats(yt,pred)

print("")
import matplotlib.pyplot as plt
print("---ploting LOO residuals---")
print("---LOO residuals before full optim ---")
looR, press = net.looResiduals()
plt.plot(looR,'b.')
plt.show()
print("---LOO residuals after full optim ---")
looR2, press2 = optNet.looResiduals()
plt.plot(looR2,'b.')
plt.show()






#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
#-------------------- Regression Example ------------------------------#
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
import math as mth
import numpy as np
x0 = np.linspace(-10,10,120)
fn= np.vectorize(lambda x: mth.sin(x)/x)
y0 = fn(x0)

#plt.plot(x0,y0,'r-')
x = x0
y = fn(x0)+np.random.normal(0,0.2, len(x0))
#plt.plot(x,y,'go')

net = ls.LSSVM( ls.RBF(), mu = 0.1, PrbType='reg')
x.shape = (len(x),1)
print("Training.....")
net.train(x,y)
plt.plot(x,y,'.g',label='Train')
plt.hold()
plt.plot(x,y0,'r-',label='Exact')
plt.hold()
plt.plot(net.x, net.yhat,'b-', label='Predict')
plt.hold()
plt.legend(loc='upper right', shadow=True)
plt.show()

print("optimatisation RBF and regular")
net2 = net.optRBF()
plt.plot(x,y,'g.',label='Train')
plt.hold()
plt.plot(x,y0,'r-',label='Exact')
plt.hold()
plt.plot(net2.x, net2.yhat,'b-', label='Predict')
plt.hold()
plt.legend(loc='upper right', shadow=True)
plt.show()

#--------------------------------------------------------------------------
#------------------------------- standarization ---------------------------
#--------------------------------------------------------------------------
y = trainData['Class']
x = trainData[['X','Y']]

def standarize(v):
    mean = v.mean()
    std  = v.std()
    return ( (v-mean)/std, mean, std)
    
x1, m1,s1 = standarize(x['X'])
x2, m2,s2 = standarize(x['Y'])
    

x['Xs'] = x1
x['Ys'] = x2

net = ls.LSSVM( kern=ls.Polynomial(2,0.1), mu = 0.1)

print("---------- Diagnostics -------------")

print("----------Train Partition-----------")
net.train(x[['Xs','Ys']].as_matrix(),y.as_matrix())
print("----------Test Partition-----------")

yt = testData['Class']
xt = testData[['X','Y']]
#-------- transforming the test set --------------- 
xt['Xs']=(xt['X']-m1)/s1
xt['Ys']=(xt['Y']-m2)/s2
(pred,prob,sign) = net.predict(xt[['Xs','Ys']].as_matrix())
net.errStats(yt,pred)

#----------------------- end of standarization example -----------------









