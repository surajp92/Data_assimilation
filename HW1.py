# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
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
def integrand(x,gamma):
    return x*gamma*np.exp(-x*gamma)

nobs = 5
nvar = 3

gamma = np.array([1/0.9, 1/0.7, 1/0.5, 1/0.3, 1/0.2])

level = np.array([1.0, 0.5, 0.2, 0.0])

H = np.zeros((nobs, nvar))

for i in range(nobs):
    for j in range(nvar):
        I = quad(integrand, level[j+1], level[j], args=(gamma[i]))
        H[i,j] = I[0]
        
xbar = np.array([0.9,0.85,0.875])

zbar = np.dot(H,xbar)        
        
mu = 0
std = np.std(xbar)

v = np.random.normal(mu,std,5)

z = zbar + v

# lu decomposition
M = lu(H)
L = M[1]
U = M[2]

# pseido-inverse for checking
xls_p = np.dot(np.linalg.pinv(H),z)

# LU decomposition
xls_lu = multi_dot([inv(np.dot(H.T,H)), H.T, z])

#g = multi_dot([inv(np.dot(L.T,L)), L.T, z])
#xls_lu = multi_dot([inv(U), g])

# QR decomposition
Q,R = np.linalg.qr(H)
xls_qr = np.dot(inv(R),np.dot(Q.T,z))

# SVD decomposition
U, s, Vh = np.linalg.svd(H)
S  = np.dot(np.diag(1/s),np.eye(nvar,nobs))
xls_svd = multi_dot([Vh.T, S, U.T, z])



