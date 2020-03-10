# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy as sc
from scipy.integrate import odeint
from scipy.optimize import brentq

n   = 500
Z   = 1.0
l   = 0
R   = np.logspace(-6,2.2,n)
R_1 = R[::-1] # we start from rmax coming down to zero

def bound(UQ,r,E,l):
    return [UQ[1], -(E +2.0*Z/r -l*(l+1.0)/r**2)*UQ[0]]

def shoot(E,l):
    sol = odeint(bound, initialCond, R_1, args=(E,l))
    t = sol[:,0]
    t =t[::-1]/max(np.abs(t))
    tPrime = (t[1]-t[0])/(R[1]-R[0])
    return  t[0] + tPrime*(0.0-R[0])
   
initialCond = [0.0, -1E-4]

#En = brentq(shoot,-1.2,-0.7,args =(l,), xtol=1e-17)
#print(En)


Nmax = 6
lmax = Nmax-1
maxsteps = 1000
en  = []
prb = []
dE  = 0.1
dE_change = dE
for l in range(Nmax):
    E0 = -1.2*Z**2
    Elevel = 0    
    u0 = shoot(E0,l)
    for i in range(maxsteps):        
        E1 = E0 + dE        
        if E1>0: break
        u1 = shoot(E1,l)
        if u0*u1<0:
            Ex = brentq(shoot,E0,E1,args =(l,), xtol=1e-17)                 
            print([Elevel+l,l,Ex])
            en.append([Elevel+l,l,Ex])
            Elevel +=1
            dE = dE/2             
        if Elevel>=Nmax-l: 
            dE = dE_change/1.5
            dE_change = dE
            print("new dE:",dE)
            break
        u0 = u1                   
        E0 +=dE
print("\n\n")
en.sort(key = lambda x: [np.round(x[2],4), x[1]])
for ss in en : print(ss)
            
            
    

















