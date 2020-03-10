"""
Created on Fri Nov 10 22:26:51 2017

Here we use logarithmic grid which is much better suited
as the most variation of the potential happends ver  close to 0

@author: ks_work
"""

import numpy as np
import math
from scipy.integrate import odeint
from scipy.optimize import brentq
from scipy.integrate import simps
import matplotlib.pyplot as plt 
"""--------------------------------------------------------------
   Z = 1, hydrogen atom
   -1/2 U" + (-Z/r + l(l+1)/2*r^2)U = E*U, where U(r) = r*Psi(r)
   This is assuming E is expressed in Hartree
   
   In Rayleig unit of energy
   
   -U'' + (-2Z/r + l(l+1)/r^2)U = E*U
   We go with the latter
   --------------------------------------------------------------
"""
""" Coulomb Potential  """
def V(r,Z):
    return -2.0*Z/r
""" Centripetal Potential"""
def Vcent(r,l):
    return l*(l+1.0)/(r**2.0)


"""-------------------------------------------------------
U = r*Psi
--
U'   = Q
Q'   = (E - V(r)-Vcent)*U
Initial condition at RMAX, r varies from 0 to RMAX
U(RMAX)   = 0
U(RMAX-h) = 1e-4  Q' = r*Psi' + Psi --> Q' = (0-U(RMAX-h)/h
----------------------------------------------------------
"""
def odeSystem(inVal,r,E,Z,l):     
    U,Q = inVal
    return [Q, (V(r,Z)+Vcent(r,l)-E)*U ]

def shoot(E,Z,l,r):
    sol    = odeint(odeSystem,iniValue, r, args=(E,Z,l))
    t      = sol[:,0]
    t      = t/max(np.abs(t))
    tPrime = (t[-1]-t[-2])/(Rrev[-1]-Rrev[-2])
    return t[-1] + tPrime*(0-R[0]) # first order apprx at 0
                    



""" Initialisations: """
n        = 500
R_MAX    = 60.0
R        = np.logspace(-6,np.log10(R_MAX),n)
Rrev     = R[::-1] # we start from rmax coming down to zero
Z        = 1.0
l        = 0.0
iniValue = [0.0, -1.0E-6]


#for simps to work R needs to be increasing
def normalise(U):
    NormFact  = simps(U**2 ,R)
    return U/math.sqrt(NormFact)


#En = brentq(shoot,-1.5,-0.8,args=(Z,l,Rrev))
#print("E = ",En)

"""
sol = odeint(odeSystem, iniValue, R, args=(En,Z,l))
Ur = sol[:,0]
Ur= normalise(Ur)
#psi = [Ur[i]/R[i] for i in range(len(R))]
#plt.plot(R,psi)
plt.plot(R,Ur**2)
plt.title("Probability density")
plt.xlabel("\r(bohr)")
plt.grid()
plt.show()
"""


Nsearch = 500
Nmax = 5
lmax = Nmax-1
EigenValues = []
E0 = -1.2*Z*Z
dE = 0.1*Z**2
#dE = 0.1
bag =[]
for l in range(Nmax): # or range(lmax+1)
    E0 = -1.2*Z**2
    dE = 0.05*Z**2     
    Elevel = 0
    shootE0 = shoot(E0,Z,l,Rrev)
    for n in range(Nsearch):
        shootE = shoot(E0+dE,Z,l,Rrev)
        #print("[",E0,E0+dE,"]")
        if shootE0*shootE <0:
            EigEn  = brentq(shoot,E0,E0+dE,args=(Z,l,Rrev), xtol=1e-17 )   
            Elevel += 1
            print("n = ",Elevel+l,"l = ",l," E = ", EigEn)            
            EigenValues.append([Elevel+l,l,EigEn])   
            if (Elevel>=Nmax-l): break
        #dE = dE/1.05
        shootE0 = shootE
        E0     +=dE
        if E0>-1E-4 : 
            break

    

# 
S = sorted(EigenValues, key= lambda t: (	np.round(t[2],4), t[1] ))
for t in S:
    print(t[:3])


""" 
bear in mind that the solution are reversed and so does R
R varies from Rmax...0
Sol          Rmax*Psi(Rmax) ... 0*Psi(0)
To use them properly we need to reverse the solution
plt.plot does that automatically
"""

"""
plt.figure(figsize=[8,8])
for g in S:
    # g[3] point to the function r*psi(r)
    plt.plot(R,g[3],label='n='+str(g[0])+' l='+str(g[1])) 
plt.xlabel("radius (bohr)")
plt.ylabel("r*psi(r)")
plt.legend()
plt.grid()
plt.show()
"""


"""
            #sol    = odeint(odeSystem, iniValue, Rrev, args=(EigEn,Z,l))
            #Ur     = sol[:,0]
            #Ur     = normalise(Ur[::-1])


            if Elevel>=Nmax-l: 
                #E0   = min(bag)
                #dE   = dE/2.0 
                #print("Emin =",E0," dE =",dE)                     
                #bag.remove(E0)         
                break
"""









