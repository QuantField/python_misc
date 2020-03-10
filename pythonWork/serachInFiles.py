# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:40:09 2017

@author: saadik1
"""

import os, sys

items = os.listdir()
# list .py files in current dirctory
files = [ obj for obj in items if (os.path.isfile(obj) and '.py' in obj)   ]
         
searchSting = 'W>=0.5'         
for file in files:
        row = 0
        for line in open(file,'r').readlines():
            row+=1
            if searchSting in line.upper():
                print("FILENAME :",file,"LINE=",row," --->", line)
