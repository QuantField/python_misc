# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:27:43 2017

@author: saadik1
"""

def calPi():
    import random as rd
    n = int(input("n = "))
    s = [(rd.random()**2 +rd.random()**2)<=1  for i in range(n)]
    PI = 4*sum(s)/n
    return PI
    