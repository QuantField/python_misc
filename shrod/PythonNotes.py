
# coding: utf-8

# # Python good coding tips
# 

# In[6]:

color = ['red','blue','green','yellow','black']


# In Python 2.7  range(n) was inconvenient as it created a list. It has been replaced by xrange(n) which is an iterator, this was very efficient. With Python 3 xrange is simply range.

# In[7]:

for r in color:
    print(r)


# ## Looping backwards, traditional vs. new

# In[8]:

for i in range(len(color)-1,-1,-1):
    print(color[i])


# In[9]:

for r in reversed(color):
    print(r)


# ## Looping with indices

# In[10]:

for i,r in enumerate(color):
    print(i,"--->",r)


# ## Looping ove two collections

# In[11]:

color = ['red','blue','green','yellow','black']
town  = ['medina', 'cyprus','damas']
for h,k in zip(color,town):
    print(h,k)


# In[12]:

a = list(range(10))
b = list(range(10))
[i+j for i,j in zip(a,b)]


# ## Dictionaries

# In[24]:

Dict1 = {'electron':'fermion','proton':'fermion','photon':'boson'}


# In[25]:

for k in Dict1.keys():
    print(Dict1[k])


# In[26]:

# better way
for k,v in Dict1.items():
    print(k,v)


# ### Create dictionary from two collections

# In[29]:

color = ['red','blue','green']
town  = ['medina', 'cyprus','damas']
Dic   = dict(zip(town,color))
print(Dic)


# # Class example

# In[56]:

class poly:
    def __init__(self, *coefs):
        self.coefs = coefs
    def __add__(self,other):
        if len(self.coefs)==len(other.coefs):
            return poly([a+b for a,b in zip(self.coefs, other.coefs)])
        else:
            print("Polynomials must have same degree")
            raise Exception(classmethod)
    def __str__(self):
        return str(self.coefs)
    
p = poly(3,4,0,1)
q = poly(-1,3,2,5)
t = poly(-2,3,0)
print(p,q, p+q)
    


# # Generators
# Are simple way to create iterators

# In[62]:

def periodicTable_next():
    yield 'H'
    yield 'He'
    yield 'Na'
    yield 'Mg'
    
iterator = periodicTable_next() 
print(next(iterator))  # next applies to iterator
print(next(iterator))
print(next(iterator))
print(next(iterator))
# if we run one more time we will have an error


# In[63]:

for r in periodicTable_next():
    print(r)


# In[69]:

# generate Fibonacci sequence 0,1,1,2,3,5,...

def fibonacci():    
    a, b = 0,1
    while True:
        yield a
        a, b =  b, a+b
        
for f in fibonacci()  :
    if f>50 : break  
    print(f)


# In[68]:

def fibonacci(n):    
    a, b = 0,1
    while a<n:
        yield a
        a, b =  b, a+b
        
for f in fibonacci(50):
    print(f)
        


# ## Iterators 
# objects on which iter is applicable. the object becomes iterable

# In[75]:

class atoms:
    def __init__(self):
        self.elements = ['H','He','Na', 'Mg','Al', 'Si','P','Se']
        self.ind = -1
        
    def __iter__(self):  # must be defined
        return self
    
    def __next__(self): # also must be defined
        self.ind += 1
        if self.ind == len(self.elements):
            raise StopAsyncIteration # standard procedure, I thought it would loop
        return self.elements[self.ind]    
    
X = atoms()
itr = iter(X) # X is an iterator therefore I can apply iter
print(next(itr))
print(next(itr))

# this will create an error as the end of the list is reached
#print()            
#Y = atoms()    
#for r in Y:
#    print(r)                


# ## Decorators
# Help make the code look clean and uncluttered. Example here is a timing function

# In[82]:

import time

def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result= func(*args, **kwargs)
        end   = time.time()
        print(func.__name__+ " N = "+ str(args[0])+ " took  "+str(round(end-start,4)) + " sec")
        return result
    return wrapper

import numpy as np

@time_it
def invert(N):
    a = np.random.random([N,N])
    b = np.linalg.inv(a)
    
invert(1000)
invert(2000)
invert(3000)

