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

#%%
sigma = 10.0
rho = 28.0
beta = 8/3
dt = 0.01
ttrain = 2.0
tmax = 5.0
freq = 1
mu = 0.0
var = 0.09
std = np.sqrt(var)
max_iter = 50
tolerance = 1e-6
lr = 0.1

X_init = np.ones(3)
X_da_init = 1.1*np.ones(3)
nttrain = int(ttrain/dt)
ntmax = int(tmax/dt)
nobs = int(nttrain/freq)
t = np.linspace(0,ttrain,nttrain+1)

ind = [freq*i for i in range(1,int(nttrain/freq)+1)]
tobs = t[ind]

rk = var*np.eye(3)

f = np.zeros((nobs,3))
lagr = np.zeros((nobs,3))

xold = X_da_init

def plot_lorenz(X,t,Z,tz):
    color = ['r','b','g']
    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(8,6),sharex=True)
    axs = ax.flat
    
    for k in range(X.shape[1]):
        axs[k].plot(t,X[:,k],label=r'$x_'+str(k+1)+'$',color=color[k],
                   linewidth=2)
        axs[k].plot(tz,Z[:,k],'o',label=r'$z_'+str(k+1)+'$',color=color[k],
                    fillstyle='none',markersize=8)
        axs[k].set_xlim(0,t[-1])
        axs[k].set_ylim(np.min(X[:,k])-5,np.max(X[:,k])+5)
        axs[k].legend()
      
    plt.show()
       
def lorenz(X_init,sigma,rho,beta,dt,nt):
    X = np.zeros((nt+1,3))
    
    X[0,0] = X_init[1]; X[0,1] = X_init[1]; X[0,2] = X_init[2]
    
    for n in range(1,nt+1):
        X[n,0] = X[n-1,0] - dt*sigma*(X[n-1,0] - X[n-1,1])
        X[n,1] = X[n-1,1] + dt*(rho*X[n-1,0] - X[n-1,1] - X[n-1,0]*X[n-1,2])
        X[n,2] = X[n-1,2] + dt*(X[n-1,0]*X[n-1,1] - beta*X[n-1,2])
    
    return X

def dxk_model(sigma,rho,beta,dt,xk):
    dxk_m = np.zeros((3,3))
    dxk_m[0,0] = 1 - sigma*dt
    dxk_m[0,1] = -dt
    dxk_m[1,0] = rho*dt
    dxk_m[1,1] = 1 - dt
    dxk_m[1,2] = -dt*xk[0]
    dxk_m[2,0] = dt*xk[1]
    dxk_m[2,2] = 1-beta*dt
    
    return dxk_m
        
Xtrue = lorenz(X_init,sigma,rho,beta,dt,nttrain)

xobs = Xtrue[ind]
zobs = xobs + np.random.normal(mu,std,[nobs,3])

for p in range(max_iter):
    Xda = lorenz(xold,sigma,rho,beta,dt,nttrain)
    
    xdaobs = Xda[ind]
    for k in range(nobs):
        xk = xdaobs[k,:].reshape(-1,1)
        dxk_m = dxk_model(sigma,rho,beta,dt,xk)
        fk = np.linalg.multi_dot([dxk_m.T, np.linalg.inv(rk), xk])
        f[k,:] = fk.flatten()
        
    lagr[-1] = f[-1,:]
    
    for k in range(nobs-2,-1,-1):
        xk = xdaobs[k,:].reshape(-1,1)
        lkp = lagr[k+1].reshape(-1,1)
        dxk_m = dxk_model(sigma,rho,beta,dt,xk)
        lk = np.dot(dxk_m.T, lkp)
        lagr[k,:] = lk.flatten()
    
    x0 = xdaobs[0,:].reshape(-1,1)
    dx0_m = dxk_model(sigma,rho,beta,dt,x0)
    grad = -np.dot(dx0_m.T,lagr[0,:].reshape(-1,1))
    
    xnew = xold - lr*grad.flatten()/np.linalg.norm(grad)
    
    print(p, ' ', xold, ' ' , xnew, ' ', np.linalg.norm(xnew-xold))
    
    if np.linalg.norm(xnew-xold) < tolerance:
        break
    
    xold = xnew
    
plot_lorenz(Xtrue,t,zobs,tobs)

#X = lorenz(X_init,sigma,rho,beta,dt,ntmax)
#plot_lorenz(X,tmax,ntmax)





































