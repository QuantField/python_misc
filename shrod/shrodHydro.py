import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq
from scipy.integrate import simps
import matplotlib.pyplot as plt 
"""
   Z = 1, hydrogen atom
   -1/2 U" + (-Z/r + l(l+1)/2*r^2)U = E*U, where U(r) = r*Psi(r)
"""
""" Coulomb Potential  """
def V(r):
    if np.abs(r)<=1e-3:
        return -1.0E-3
    else :
        return -Z/r
""" Centripetal Potential"""
def Vcent(r):
    if np.abs(r)<=1e-3:
        return 1.0E-6
    else :
        return l*(l+1)/(2.0*r**2)

"""
Hydrogenoid is when the core holds Z protons and there is only one electron orbiting
the core
"""

Z = 1.0
l = 0.0
grid_len = 400
rmax = 20
r = np.linspace(0,rmax,grid_len)
r = r[::-1] # we start from rmax coming down to zero
"""
U = r*Psi

U'   = Q
Q'   = -2*(E - V(r)-Vcent)*U

U(rmax)   = 0
U(rmax-h) = 1e-4  Q' = r*Psi' + Psi --> Q' = U(rmax-h)/h
"""

h = np.abs(r[1]-r[0])
Uinf = 0.0
Qinf = 1e-6/h

iniValue = [Uinf,Qinf]




def odeSystem(inVal,r,E):     
    U,Q = inVal
    return [Q, -2*(E - V(r)-Vcent(r))*U ]

def shoot(E):
    sol = odeint(odeSystem,iniValue, r, args=(E,),mxstep=50000)
    t   = sol[:,0]
    return  t[-1]/max(np.abs(t))

En = brentq(shoot,-1,-0.4)
#En = brentq(shoot,-0.2,-0.1)
print(En)

sol = odeint(odeSystem, iniValue, r, args=(En,),mxstep=50000)

def normilizePob( U):
    NormFact  = simps(U**2,dx=h)
    return U/np.sqrt(NormFact)
    
# Normalisation
Ur = normilizePob(sol[:,0])

# getting the max principat quantum number n for a given Z
# This is for a hydrogenoid atom
# each principal number n is filled by 2*n**2 electrons
# this usuful in n-electron problem. not here as we deal with one electron only
"""
def get_n(z):
    Nmax=0
    Tot = 0
    while(Tot<z): 
      Nmax+=1
      Tot += 2*Nmax**2 # filling the states
    return Nmax
"""





energySpectrum  = np.linspace(-Z**2,-0.001,50)
dE              = energySpectrum[1]-energySpectrum[0]

enBounds = []
n0       = 0
for e in energySpectrum[:-1]: # all elements except the last one
    enext = e + dE
    if (shoot(e)*shoot(enext)<0):
        n0 +=1
        E0 = brentq(shoot,e,enext)
        print(E0)
        enBounds.append([n0,E0]) 


Nmax = 3

for l in range(Nmax):# l =0..Nmax-1
    



np.seterr(divide='ignore')
psi = [Ur[i]/r[i] for i in range(len(r))]
#psi[0]=0
#plt.plot(r,psi)
plt.plot(r,Ur**2)
plt.title("Probability density")
plt.xlabel("r(bohr)")
plt.grid()
plt.show()         