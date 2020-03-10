
#--------- operation on columns -----------
import pandas as pd
Tab = pd.DataFrame({ 'V1':list('ABC')*4,
  'V2':list('DEFG')*3,
  'V3':list('DEFG')*3,
  'V3':list('IJKLMN')*2,
 }) 
   V1 V2 V3
0   A  D  I
1   B  E  J
2   C  F  K
3   A  G  L
4   B  D  M
5   C  E  N
6   A  F  I
7   B  G  J
8   C  D  K
9   A  E  L
10  B  F  M
11  C  G  N 
  
# concatenate two columns 
Tab['V4'] = Tab.apply(lambda row : row['V1']+row['V2'], axis = 1) 
   V1 V2 V3  V4
0   A  D  I  AD
1   B  E  J  BE
2   C  F  K  CF
3   A  G  L  AG
4   B  D  M  BD
5   C  E  N  CE
6   A  F  I  AF
7   B  G  J  BG
8   C  D  K  CD
9   A  E  L  AE
10  B  F  M  BF
11  C  G  N  CG

#--------------------- 
#-------- counting occurence of each word -----------------------------
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

#------- Search string 'APPLY' in a list of (.py) files under a given directory
import os, sys
items = os.listdir()
# list .py files in current dirctory
files = [ obj for obj in items if (os.path.isfile(obj) and '.py' in obj)   ]       
searchSting = 'APPLY'         
for file in files:
        row = 0
        for line in open(file,'r').readlines():
            row+=1
            if searchSting in line.upper():
                print("FILENAME :",file,"LINE=",row," --->", line)

------------------------------------------------------------------------------------------    
                  SAS                           |             Pandda 
------------------------------------------------------------------------------------------
data Tab;										|	Tab.set_value(Tab['V2']=='D','V3','II')
	set Tab;                                    | 
	if V2=='D' then V3 = 'II'                   | 
run;	                                        | 
--------------------------------------------------------------------------------------------
// create variable one_flag = {0,1}
// 1 if V1='A'
data Tab;                                          #using numpy 
                                                   import numpy as np 
	set Tab;                                       Tab['one_flag'] = np.apply_along_axis(lambda w:w=='A', 0, Tab['V1']) # result is true
    one_flag = (V1='A');                           import pandas as pd # using pandas  
run;	                                           Tab['one_flag'] = Tab.apply(lambda row : int(row['V1']=='A'), axis = 1)    










