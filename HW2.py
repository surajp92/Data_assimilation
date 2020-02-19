#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:43:00 2020

@author: suraj
"""

import numpy as np
np.random.seed(22)
import matplotlib.pyplot as plt

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
oo = 3

xtrue = np.zeros(n)
zobs = np.zeros(n)

xtrue[0] = xo
noise = np.random.normal(mu,sigma,n)
zobs[0] = xtrue[0] + noise[0]
for i in range(1,n):
    xtrue[i] = a*xtrue[i-1]
    zobs[i] = xtrue[i] + noise[i]

if oo == 1: 
    ind = [i for i in range(n)]
    obs = zobs[ind]

elif oo == 2:
    ind = np.hstack((0,[5*i+4 for i in range(10)]))
    obs = zobs[ind]
    
elif oo == 3:
    ind = np.hstack((0,[10*i+9 for i in range(5)]))
    obs = zobs[ind]

xold = 0.5
xda = np.zeros(n)
nobs = np.shape(obs)[0]
f = np.zeros(nobs)
lagr = np.zeros(nobs)

res = 1

for p in range(max_iter):   
    xda[0] = xold
    for i in range(1,n):
        xda[i] = a*xda[i-1]
    
    xtemp = xda[ind]
    for k in range(nobs):
        f[k] = (1/sigma2)*(obs[k] - xtemp[k])
    
    lagr[-1] = f[-1]
    
    for k in range(nobs-2,-1,-1):   
        lagr[k] = a*lagr[k+1] + f[k]
    
    grad = -a*lagr[0]
    
    xnew = xold - lr*grad
    
    print(p, ' ', xold, ' ' , xnew, ' ', np.linalg.norm(grad))
    
    if np.linalg.norm(grad) < tolerance:
        break
    
    xold = xnew
        
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6),sharex=True)
ax.plot(ind,xtrue[ind],color='b',linewidth=3,label='True')
ax.plot(ind,xda[ind],'g--',linewidth=3,label='4D VAR')
ax.plot(ind,obs,'ro',fillstyle='none',markersize=12,markeredgewidth=3,label='Observations')
    
ax.legend()
plt.show()
    
#%%
n = 20

mu = 0.0
sigma2 = 0.5
sigma = np.sqrt(sigma2)

xo = 0.5

max_iter = 20
tolerance = 1e-6
lr = 0.0001

xtrue = np.zeros(n)
zobs = np.zeros(n)

xtrue[0] = xo
noise = np.random.normal(mu,sigma,n)
zobs[0] = xtrue[0] + noise[0]
for i in range(1,n):
    xtrue[i] = 4*xtrue[i-1]*(1 - xtrue[i-1])
    zobs[i] = xtrue[i] + noise[i]

ind1 = [i for i in range(n)]
o1 = zobs[ind1]

plt.plot(xtrue)
plt.plot(o1,'ro')
plt.show()

xold = 0.8
xda = np.zeros(n)
nobs = np.shape(o1)[0]
f = np.zeros(nobs)
lagr = np.zeros(nobs)

res = 1

for p in range(max_iter):   
    xda[0] = xold
    for i in range(1,n):
        xda[i] = 4*xda[i-1]*(1 - xda[i-1])
    
    xtemp = xda[ind1]
    for k in range(nobs):
        f[k] = (4.0 - 8.0*xtemp[k])*(1/sigma2)*(o1[k] - xtemp[k])
    
    lagr[-1] = f[-1]
    
    for k in range(nobs-2,-1,-1):   
        lagr[k] = (4.0 - 8.0*xtemp[k])*lagr[k+1] + f[k]
    
    grad = -(4.0 - 8.0*xtemp[0])*lagr[0]
    
    xnew = xold - lr*grad
    print(p, ' ', xold, ' ' , xnew, ' ', np.linalg.norm(xnew-xold))
    
    if np.linalg.norm(xnew-xold) < tolerance:
        break
    
    xold = xnew

#%% model = x**2
n = 20

mu = 0.0
sigma2 = 0.5
sigma = np.sqrt(sigma2)

xo = 0.5

max_iter = 100000
tolerance = 1e-5
lr = 0.001

xtrue = np.zeros(n)
zobs = np.zeros(n)

xtrue[0] = xo
noise = np.random.normal(mu,sigma,n)
zobs[0] = xtrue[0] + noise[0]
for i in range(1,n):
    xtrue[i] = xtrue[i-1]**2
    zobs[i] = xtrue[i] + noise[i]

ind1 = [i for i in range(n)]
o1 = zobs[ind1]

plt.plot(xtrue)
plt.plot(o1,'ro')
plt.show()

xold = 0.8
xda = np.zeros(n)
nobs = np.shape(o1)[0]
f = np.zeros(nobs)
lagr = np.zeros(nobs)

res = 1

for p in range(max_iter):   
    xda[0] = xold
    for i in range(1,n):
        xda[i] = xda[i-1]**2
    
    xtemp = xda[ind1]
    for k in range(nobs):
        f[k] = (2.0*xtemp[k])*(1/sigma2)*(o1[k] - xtemp[k])
    
    lagr[-1] = f[-1]
    
    for k in range(nobs-2,-1,-1):   
        lagr[k] = (2.0*xtemp[k])*lagr[k+1] + f[k]
    
    grad = -(2*xtemp[0])*lagr[0]
    
    xnew = xold - lr*grad
    print(p, ' ', xold, ' ' , xnew, ' ', np.linalg.norm(grad))
    
    if np.linalg.norm(grad) < tolerance:
        break
    
    xold = xnew

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6),sharex=True)
ax.plot(ind1,xtrue[ind1],color='b',linewidth=3,label='True')
ax.plot(ind1,xda[ind1],'g--',linewidth=3,label='4D VAR')
ax.plot(ind1,o1,'ro',fillstyle='none',markersize=12,markeredgewidth=3,label='Observations')
ax.legend()
plt.show()

#%%
sigma = 10.0
rho = 28.0
beta = 8/3

dt = 0.01
t_train = 0.4
t_max = 5.0
freq = 2

mu = 0.0
var = 0.09
std = np.sqrt(var)

max_iter = 10000
tolerance = 1e-6
lr = 0.001

X_init = np.ones(3)
X_da_init = 1.1*np.ones(3)
nttrain = int(t_train/dt)
ntmax = int(t_max/dt)
nobs = int(nttrain/freq)
ttrain = np.linspace(0,t_train,nttrain+1)
ttmax = np.linspace(0,t_max,ntmax+1)

ind = [freq*i for i in range(1,int(nttrain/freq)+1)]
tobs = ttrain[ind]

rk = var*np.eye(3)

f = np.zeros((nobs,3))
lagr = np.zeros((nobs,3))

xold = X_da_init

def plot_lorenz(ttrain,X,t,Z,tz,xda=[],tda=0):
    print(ttrain)
    color = ['k','red','blue']
    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(8,6),sharex=True)
    axs = ax.flat
    
    for k in range(X.shape[1]):
        axs[k].plot(t,X[:,k],label=r'$x_'+str(k+1)+'$',color=color[0],
                   linewidth=2)
        axs[k].plot(tz,Z[:,k],'o',label=r'$z_'+str(k+1)+'$',color=color[1],
                    fillstyle='none',markersize=8,markeredgewidth=2)
        if xda != []:
            axs[k].plot(tda,xda[:,k],'--',label=r'$z_'+str(k+1)+'$',color=color[2],
                    fillstyle='none',markersize=8)
            axs[k].axvspan(0, ttrain, alpha=0.5, color='orange')
        axs[k].set_xlim(0,t[-1])
        axs[k].set_ylim(np.min(X[:,k])-5,np.max(X[:,k])+5)
        axs[k].set_ylabel(r'$x_'+str(k+1)+'$')
        
        #axs[k].legend()
    
    axs[k].set_xlabel(r'$t$')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.125)
    
    line_labels = ['True','Observations','4D Var']#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=4, labelspacing=0.)
      
    plt.show()
       
def lorenz(X_init,sigma,rho,beta,dt,nt):
    X = np.zeros((nt+1,3))
    
    X[0,0] = X_init[0]
    X[0,1] = X_init[1]
    X[0,2] = X_init[2]
    
    for n in range(1,nt+1):
        X[n,0] = X[n-1,0] - dt*sigma*(X[n-1,0] - X[n-1,1])
        X[n,1] = X[n-1,1] + dt*(rho*X[n-1,0] - X[n-1,1] - X[n-1,0]*X[n-1,2])
        X[n,2] = X[n-1,2] + dt*(X[n-1,0]*X[n-1,1] - beta*X[n-1,2])
    
    return X

def dxk_model(sigma,rho,beta,dt,xk):
    dxk_m = np.zeros((3,3))
    dxk_m[0,0] = 1 - sigma*dt
    dxk_m[0,1] = sigma*dt
    dxk_m[0,2] = 0
    dxk_m[1,0] = rho*dt - dt*xk[2]
    dxk_m[1,1] = 1 - dt
    dxk_m[1,2] = -dt*xk[0]
    dxk_m[2,0] = dt*xk[1]
    dxk_m[2,1] = dt*xk[0]
    dxk_m[2,2] = 1-beta*dt
    
    return dxk_m
        
Xtrue = lorenz(X_init,sigma,rho,beta,dt,nttrain)

xobs = Xtrue[ind]

#noise = np.random.normal(mu,std,[nobs,1])*np.ones((nobs,3))

zobs = xobs + np.random.normal(mu,std,[nobs,3])

grado = 0.0
for p in range(max_iter):
    Xda = lorenz(xold,sigma,rho,beta,dt,nttrain)
    
    xdaobs = Xda[ind]
    
    for k in range(nobs):
        xk = xdaobs[k,:].reshape(-1,1)
        dxk_m = dxk_model(sigma,rho,beta,dt,xk)
        hk = zobs[k,:].reshape(-1,1) - xk
        fk = np.linalg.multi_dot([dxk_m.T, np.linalg.inv(rk), hk])
        f[k,:] = fk.flatten()
        
    lagr[-1] = f[-1,:]
    
    for k in range(nobs-2,-1,-1):
        xk = xdaobs[k,:].reshape(-1,1)
        lkp = lagr[k+1].reshape(-1,1)
        dxk_m = dxk_model(sigma,rho,beta,dt,xk)
        lk = np.dot(dxk_m.T, lkp)  
        lagr[k,:] = lk.flatten() + f[k,:]
    
    x0 = xdaobs[0,:].reshape(-1,1)
    dx0_m = dxk_model(sigma,rho,beta,dt,x0)
    
    gradn = -np.dot(dx0_m.T, lagr[0,:].reshape(-1,1))
    
    xnew = xold - lr*gradn.flatten()/np.linalg.norm(gradn)  #np.abs(grad.flatten())
    
    #print(p, ' ', xold, ' ' , xnew, ' ', np.linalg.norm(grad))
    print(p, ' ', np.linalg.norm(xnew-xold), np.linalg.norm(gradn-grado))
    
    if np.linalg.norm(xnew-xold) < tolerance:
        break
    
    grado = gradn
    xold = xnew
    
plot_lorenz(t_train,Xtrue,ttrain,zobs,tobs)

Xtrue_tmax = lorenz(X_init,sigma,rho,beta,dt,ntmax)
Xda_tmax = lorenz(xnew,sigma,rho,beta,dt,ntmax)

plot_lorenz(t_train,Xtrue_tmax,ttmax,zobs,tobs,Xda_tmax,ttmax)


































