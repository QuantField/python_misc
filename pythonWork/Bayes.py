# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:30:53 2016

@author: saadik1
"""
d = {
'Gender'   :[ 'male','male','male','male','female','female','female','female'],
'height'   :[	6,	5.92,	5.58,	5.92,	5,	5.5,	5.42,	5.75],
'weight'   :[	180,	190,	170,	165,	100,	150,	130,	150],
'foot_size':[	12,	11,	12,	10,	6,	8,	7,	9]
}

fd = pd.DataFrame(d)

import scipy.stats as stats
#P_height = stats.norm(fd['Gender']==']['height'].mean()  ,fd['height'].std())




m =  fd[(fd['Gender']=='male')]
f =  fd[~fd.index.isin(m.index)]
        
P_male_prior   = len(m)/fd.shape[0]
P_female_prior = len(f)/fd.shape[0]
        

# assume probabilities are normal
# conditional probablility        
        
P_height_male   = stats.norm(m['height'].mean(), m['height'].std())        
P_height_female = stats.norm(f['height'].mean(), f['height'].std())               

P_weight_male   = stats.norm(m['weight'].mean(), m['weight'].std())        
P_weight_female = stats.norm(f['weight'].mean(), f['weight'].std()) 

P_footsize_male   = stats.norm(m['foot_size'].mean(), m['foot_size'].std())        
P_footsize_female = stats.norm(f['foot_size'].mean(), f['foot_size'].std())  


# Now we have to test this :
#Gender	height	weight foot_size
#sample	6	      130	8

P_male_posterior = P_male_prior*P_height_male.pdf(6)*P_weight_male.pdf(130)*\
                   P_footsize_male.pdf(8) 


P_female_posterior = P_female_prior*P_height_female.pdf(6)*P_weight_female.pdf(130)*\
                   P_footsize_female.pdf(8) 
                   
evidence = P_male_posterior+P_female_posterior

#normalised to give probabilities

P_male_posterior   = P_male_posterior/evidence  #1.15230663498e-05

P_female_posterior = P_female_posterior/evidence #0.999988476934

print (P_male_posterior,P_female_posterior)

# Analysis suggests instance is female



P(male|[6,130,8]) = P(male)*P(height =6|male)*P(weight=130|male)*P(footsize=8|male)


#------- Using the learning kit --------------------------



d = {
'Gender'   :[ 'male','male','male','male','female','female','female','female'],
'height'   :[	6,	5.92,	5.58,	5.92,	5,	5.5,	5.42,	5.75],
'weight'   :[	180,	190,	170,	165,	100,	150,	130,	150],
'foot_size':[	12,	11,	12,	10,	6,	8,	7,	9]
}

fd = pd.DataFrame(d)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

fd_data  = fd.drop(['Gender'],1) 
fd_target = fd['Gender'] 

nbmodel = gnb.fit(fd_data, fd_target)
 

test = {
'height'   : [6],
'weight'   : [130],
'foot_size': [8]
}
td = pd.DataFrame(test)
 
pred = nbmodel.predict(td)
pred_prob = nbmodel.predict_proba(td)
print(pred, pred_prob)

#------- Other example
import pandas as pd



allData = pd.read_csv("C:\\KS_temp\\banana_data1000.csv")


trainData = allData.sample(frac=0.7, random_state = 564)        
testData  = allData.loc[~allData.index.isin(trainData.index)] 


y = trainData['Class']
x = trainData[['X','Y']]

yt = testData['Class']
xt = testData[['X','Y']]
 
nbmodel = gnb.fit(x, y)
pred_train = nbmodel.predict(x)
acc1 = ((y != pred_train).sum())/len(y)
pred_test = nbmodel.predict(xt)
acc2 =((yt != pred_test).sum())/len(yt)
             
print("Train_Acc =", acc1)
print("Test_Acc =", acc2)
class1 = x[(y==1)]['Y']
class2 = x[(y!=1)]['Y']           
              

import matplotlib.pyplot as plt

plt.hist(class1, bins=40, alpha=0.5, label='+1')
plt.hist(class2, bins=40, alpha=0.5, label='-1')
plt.legend(loc='upper right')
plt.show()

plt.hist2d(class1, class2  )
plt.colorbar()
plt.show()



class1.hist(bins=40, alpha=0.3 )


