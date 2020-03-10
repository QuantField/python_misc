# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 21:33:56 2017

@author: ks_work
"""

#   hard wall eigen states and energies
import numpy as np

L = 2.0 #1.0

w = np.pi/L

import scipy as sp

import math

A = math.sqrt(2/L)

def psi(n,x):
    return (A*math.sin(n*w*x))


E0 = 0.5*w**2
def E(n):
    return ((n**2)*E0)
    
import scipy.integrate as integrate

integrate.quad(lambda x:psi(4,x)*psi(4,x), 0.0, L)


##  hard wall with a step potential:
#   c  = L/2
#   V = V0 if x in [c-e, c+e]
#   

V0  =  10 # 50.0
c   = 1.0 # 0.5
e   =  0.5 # 0.1

def V(x):
    if (x<= c-e or x>=c+e):
        return(V0)
    else:
        return(0.0)


n = 50;

H = np.zeros([n,n])

for i in range(n):
    for j in range(i,n):
        tmp =integrate.quad(lambda x:psi(i+1,x)*psi(j+1,x), c-e, c+e) 
        H[j,i]=H[i,j] = V0*tmp[0]
        if (i==j): 
            H[i,j]+=E(i+1)
        

eigenValues, eigenVectors = np.linalg.eig(H)

idx = eigenValues.argsort()

En           = eigenValues[idx]
Coef         = eigenVectors[:,idx]



import matplotlib.pyplot as plt

def linCombin(n,x):#n>=1
    S = 0.0
    cn = Coef[:,n-1]
    for i in range(len(cn)):
        S+= cn[i]*psi(i+1,x)
    return(S)
        
    
x = np.linspace(0,L,100)

#pot = np.array([V(t) for t in x])

level = 1
Enn   = En[level-1]
y = np.array([linCombin(level,t)**2 for t in x])

plt.figure(figsize=[7,7])
#plt.plot(x,pot)
plt.plot(x,y)
plt.grid()
plt.title("E ="+str(Enn))
    
#integrate.quad(lambda x:linCombin(10,x)**2, 0, L) 
     
    
######################################

# Matrix, finite difference for the hardwall equation

# Psi(0) = Psi(L)=0

N = 200
L = 1.0
dx = L/N

c = 2.0*(dx**2)
N_1 = N-1
A = np.zeros([N_1,N_1])

for i in range(N_1):
    A[i,i]=2.0
    if (i<N_1-1):
        A[i,i+1] = A[i+1,i] =-1.0
        
eigenValues, eigenVectors = np.linalg.eig(A)

idx = eigenValues.argsort()

# Eigens states are column vectors of eigenVectors
EigenEnergy = eigenValues[idx]/c # c from the calculation
Psi         = eigenVectors[:,idx]

# adding the boundary conditions psi(0)=psi(N)=0
ziltch = np.zeros([1,N_1])
Psi = np.concatenate((ziltch,Psi), axis=0)
Psi = np.concatenate((Psi,ziltch), axis=0)


import scipy.integrate as integrate
# Normalise wave functions(eigen states)
def normalise():
    for i in range(Psi.shape[1]):
        d = Psi[:,i]**2
        S = integrate.simps(d, dx = dx)
        Psi[:,i] = Psi[:,i]/S


plt.figure(figsize=[9,9])
plt.imshow(Psi, aspect='auto', interpolation='none' )


xa = np.linspace(0,L,N+1) # give a dx = L/N 

state = 3        # state = 1,2,3,...
plt.plot(xa,Psi[:,state-1])
plt.title("E ="+str(EigenEnergy[state-1]))
plt.grid()

#Exact Energy spectrum :
    
E0 = (np.pi/L)**2/2.0
    
Spec = [E0*n**2 for n in range(1,N+1)]

Now compare Spec with EigenEnergy

    
    
##########################################################
       Now with a step potential
##########################################################

import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

N = 400
L = 1.0
dx = L/N

c = 2.0*(dx**2)
N_1 = N-1
A = np.zeros([N_1,N_1])

e = 0.1; #0.1
xc = L/2.0;
xa = xc - e
xb = xc + e
V0 = 50

def Step(x):
    if (x<=xa or x>=xb):
        return( 0.0 )
    else :
        return( V0)
    

for i in range(N_1):
    A[i,i]=2.0 + c*Step(i*dx)
    if (i<N_1-1):
        A[i,i+1] = A[i+1,i] =-1.0
        
eigenValues, eigenVectors = np.linalg.eig(A)

idx = eigenValues.argsort()

# Eigens states are column vectors of eigenVectors
EigenEnergy = eigenValues[idx]/c # c from the calculation
Psi         = eigenVectors[:,idx]

# adding the boundary conditions psi(0)=psi(N)=0
ziltch = np.zeros([1,N_1])
Psi = np.concatenate((ziltch,Psi), axis=0)
Psi = np.concatenate((Psi,ziltch), axis=0)



# Normalise wave functions(eigen states)
def normalise():
    for i in range(Psi.shape[1]):
        d = Psi[:,i]**2
        S = integrate.simps(d, dx = dx)
        Psi[:,i] = Psi[:,i]/S


#plt.figure(figsize=[9,9])
#plt.imshow(Psi, aspect='auto', interpolation='none' )


xP = np.linspace(0,L,N+1) # give a dx = L/N 
scaleFactor = 120
yStep = np.array([Step(t)/(scaleFactor*V0)  for t in xP])

state = 1   # state = 1,2,3,...
ProbDensity = Psi[:,state-1]**2
plt.plot(xP,ProbDensity)
plt.plot(xP,yStep)
plt.title("E ="+str(EigenEnergy[state-1]))
plt.grid()

#Exact Energy spectrum :
    









