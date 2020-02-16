#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:43:00 2020

@author: suraj
"""

import numpy as np
np.random.seed(1)
from numpy.linalg import multi_dot
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.integrate import quad
from scipy.linalg import lu

font = {'family' : 'Times New Roman',
        'size'   : 12}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
n = 50
mu = 0.0
sigma2 = 0.5
sigma = np.sqrt(sigma2)
a = 1.00
xo = 1.0
max_iter = 5000
tolerance = 1e-6
lr = 0.005

xtrue = np.zeros(n)
zobs = np.zeros(n)

xtrue[0] = xo
noise = np.random.normal(mu,sigma,n)
zobs[0] = xtrue[0] + noise[0]
for i in range(1,n):
    xtrue[i] = a*xtrue[i-1]
    zobs[i] = xtrue[i] + noise[i]

ind1 = [i for i in range(n)]
o1 = zobs[ind1]
ind2 = np.hstack((0,[5*i+4 for i in range(10)]))
o2 = zobs[ind1]
ind3 = np.hstack((0,[10*i+9 for i in range(5)]))
o3 = zobs[ind2]

plt.plot(xtrue)
plt.plot(o1,'ro')
plt.show()

xold = 0.5
xda = np.zeros(n)
nobs = np.shape(o1)[0]
f = np.zeros(nobs)
lagr = np.zeros(nobs)

res = 1

for p in range(max_iter):   
    xda[0] = xold
    for i in range(1,n):
        xda[i] = a*xda[i-1]
    
    xtemp = xda[ind1]
    for k in range(nobs):
        f[k] = (1/sigma2)*(o1[k] - xtemp[k])
    
    lagr[-1] = f[-1]
    
    for k in range(nobs-2,-1,-1):   
        lagr[k] = a*lagr[k+1] + f[k]
    
    grad = -a*lagr[0]
    
    xnew = xold - lr*grad
    print(p, ' ', xold, ' ' , xnew, ' ', np.linalg.norm(xnew-xold))
    
    if np.linalg.norm(xnew-xold) < tolerance:
        break
    
    xold = xnew
