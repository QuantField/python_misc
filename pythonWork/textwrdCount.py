# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:49:21 2017

@author: saadik1
"""

import pandas as pd
f = open('congress.txt','r')
bbcTxt = f.readlines()
f.close()
import re 
txt = ''.join(bbcTxt)
L = re.split('\W+', txt)
counts = { r:L.count(r) for r in set(L)}
wordFreq = pd.DataFrame.from_dict(counts, orient='index')

wordFreq = wordFreq.sort_values(by=wordFreq.columns[0], ascending=False)







