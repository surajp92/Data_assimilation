#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:38:49 2020

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
def plot_lorenz(q,ttrain,X,t,Z,tz,xda=[],tda=0):
    print(ttrain)
    color = ['k','red','blue']
    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(8,5),sharex=True)
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
    fig.subplots_adjust(bottom=0.2)
    
    line_labels = ['True','Observations','4D Var']#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
      
    plt.show()
    fig.tight_layout()
    fig.savefig('problem4ts_' +str(q)+ '.pdf', bbox_inches='tight')
    
def plot_residual(res_history):
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5))
    ax.semilogy(res_history[:,0],res_history[:,1],'g-',label='')
    ax.set_xlabel('Itearation $(p)$')
    ax.set_ylabel('$f(x)$')
    fig.tight_layout()
    plt.show()
    fig.savefig('problem4_' +str(q)+ '.pdf') 

#%%       
def lorenz(X_init,sigma,rho,beta,dt,nt):
    X = np.zeros((nt+1,3))
    
    X[0,0] = X_init[0]
    X[0,1] = X_init[1]
    X[0,2] = X_init[2]
    
    for n in range(1,nt+1):
        x = X[n-1,0]
        y = X[n-1,1]
        z = X[n-1,2]
        X[n,0] = x - dt*sigma*(x - y)
        X[n,1] = y + dt*(rho*x - y - x*z)
        X[n,2] = z + dt*(x*y - beta*z)
    
    return X

def lorenz_rk4(X_init,sigma,rho,beta,dt,nt):
    X = np.zeros((nt+1,3))
    
    X[0,0] = X_init[0]
    X[0,1] = X_init[1]
    X[0,2] = X_init[2]
    
    for n in range(1,nt+1):
        x = X[n-1,0]
        y = X[n-1,1]
        z = X[n-1,2]
        X[n,0] = x + 0.5*dt*sigma*(2.0*(y - x) + dt*(rho*x - y - x*z) - sigma*dt*(y - x))
        
        X[n,1] = y + 0.5*dt*(rho*x - y - x*z + rho*(x + sigma*dt*(y - x)) - y - dt*(rho*x - y - x*z) - (x + sigma*dt*(y - x))*(z + dt*(x*y - beta*z)))
        
        X[n,2] = z + 0.5*dt*(x*y - beta*z + (x + dt*sigma*(y - x))*(y + dt*(rho*x - y - x*z)) - beta*z - dt*(x*y - beta*z)) 
    
    return X

def dxk_model_rk4(sigma,rho,beta,dt,xk):
    x = xk[0]
    y = xk[1]
    z = xk[2]
    dxk_m = np.zeros((3,3))
    dxk_m[0,0] = 1 + sigma*dt*(rho*dt + sigma*dt - z*dt -2)
    dxk_m[0,1] = -0.5*sigma*dt*(rho*dt - dt -2)
    dxk_m[0,2] = -0.5*sigma*dt**2*x
    dxk_m[1,0] = -0.5*dt*(rho*(-2 + dt + sigma*dt) + 2*dt*x*y + 2*z + \
                 (-1 - beta - sigma)*dt*z + sigma*dt**2*(-2*x*y + y**2 + beta*z))
    dxk_m[1,1] = 1 + 0.5*dt*(-2 + dt + rho*sigma*dt - dt*x*(x + sigma*dt*(-x + y)) - \
                 sigma*dt*(z + dt*(x*y - beta*z)))
    dxk_m[1,2] = 0.5*dt*(-x + dt*x - (1 - beta*dt)*(x + sigma*dt*(-x + y)))
    dxk_m[2,0] = 0.5*dt*(y - dt*y + dt*(x + sigma*dt*(-x + y))*(rho - z) + \
                 (1 - sigma*dt)*(y + dt*(rho*x - y - x*z)))
    dxk_m[2,1] = 0.5*dt*(x - dt*x + (1 - dt)*(x + sigma*dt*(-x + y)) + \
                 sigma*dt*(y + dt*(rho*x - y - x*z)))
    dxk_m[2,2] = 1 + 0.5*dt*(-2*beta + beta*dt - dt*x*(x + sigma*dt*(-x + y)))
    
    return dxk_m

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

def obj_function(zk,xk,rk,nobs):
    
    obj_fun = 0.0
    for n in range(nobs):
        diff = zk[n,:].reshape(-1,1) - xk[n,:].reshape(-1,1)
        #obj_fun = obj_fun + np.linalg.multi_dot([diff.T,np.linalg.inv(rk),diff])    
        obj_fun = obj_fun + diff.T @ np.linalg.inv(rk) @ diff    
        
    return 0.5*obj_fun

def fourdvar(zobs,xdaobs,dxk_h,rk,nobs):
    f = np.zeros((nobs,3))  
    lagr = np.zeros((nobs,3))

    for k in range(nobs):
        xk = xdaobs[k,:].reshape(-1,1)
        zk = zobs[k,:].reshape(-1,1)
        hk = zk - xk
        #fk = np.linalg.multi_dot([dxk_h.T, np.linalg.inv(rk), hk])
        fk = dxk_h.T @ np.linalg.inv(rk) @ hk
        f[k,:] = fk.reshape(1,3)
        
    lagr[-1] = f[-1,:]
    
    for k in range(nobs-2,-1,-1):
        xk = xdaobs[k,:].reshape(-1,1)
        lkp = lagr[k+1].reshape(-1,1)
        dxk_m = dxk_model(sigma,rho,beta,dt,xk)
        #lk = np.dot(dxk_m.T, lkp)  
        lk = dxk_m.T @ lkp  
        lagr[k,:] = lk.reshape(1,3) + f[k,:]
    
    x0 = xdaobs[0,:].reshape(-1,1)
    dx0_m = dxk_model(sigma,rho,beta,dt,x0)
    
    #grad = -np.dot(dx0_m.T, lagr[0,:].reshape(-1,1))
    grad = -dx0_m.T @ lagr[0,:].reshape(-1,1)
    
    return grad

def fourdvar_rk4(zobs,xdaobs,dxk_h,rk,nobs):
    f = np.zeros((nobs,3))  
    lagr = np.zeros((nobs,3))

    for k in range(nobs):
        xk = xdaobs[k,:].reshape(-1,1)
        zk = zobs[k,:].reshape(-1,1)
        hk = zk - xk
        #fk = np.linalg.multi_dot([dxk_h.T, np.linalg.inv(rk), hk])
        fk = dxk_h.T @ np.linalg.inv(rk) @ hk
        f[k,:] = fk.reshape(1,3)
        
    lagr[-1] = f[-1,:]
    
    for k in range(nobs-2,-1,-1):
        xk = xdaobs[k,:].reshape(-1,1)
        lkp = lagr[k+1].reshape(-1,1)
        dxk_m = dxk_model_rk4(sigma,rho,beta,dt,xk)
        #lk = np.dot(dxk_m.T, lkp)  
        lk = dxk_m.T @ lkp  
        lagr[k,:] = lk.reshape(1,3) + f[k,:]
    
    x0 = xdaobs[0,:].reshape(-1,1)
    dx0_m = dxk_model_rk4(sigma,rho,beta,dt,x0)
    
    #grad = -np.dot(dx0_m.T, lagr[0,:].reshape(-1,1))
    grad = -dx0_m.T @ lagr[0,:].reshape(-1,1)
    
    return grad
    
#%%
sigma = 10.0
rho = 28.0
beta = 8/3

dt = 0.01
t_train = 2.0
t_max = 5.0
freq = 2

q = 0
mu = 0.0
varl = np.array([0.09])
var = varl[q]
std = np.sqrt(var)

max_iter = 50
tolerance = 1e-3

mode = 0 # 0: Static, 1 Dynamics
lr = 0.0001
tau = 0.5
lr_init = 1.0

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

golden_section = True
line_search = False
dyn_alpha = False
rk4 = False

if rk4 == True:       
    Xtrue = lorenz_rk4(X_init,sigma,rho,beta,dt,nttrain)
else:
    Xtrue = lorenz(X_init,sigma,rho,beta,dt,nttrain)
    
xobs = Xtrue[ind]
zobs = xobs + np.random.normal(mu,std,[nobs,3])

res_history = np.zeros((max_iter,2))

dxk_h = np.eye(3,3)

rgs = 0.618034


#%%  golden section method
if golden_section == True:
    for q in range(max_iter):
        xda = lorenz(xold,sigma,rho,beta,dt,nttrain)
        xdaobs = xda[ind]    
        grad = fourdvar(zobs,xdaobs,dxk_h,rk,nobs)
        
        pk = -np.copy(grad)
        
        ags = 0.0
        bgs = 1.0
    
        Li = bgs - ags
        L1 = ags + rgs**2*Li
        L2 = ags + rgs*Li
        
        while Li >= 1e-4:
            xk1 = xdaobs + L1*pk.reshape(1,3)
            ofk1 = obj_function(zobs,xk1,rk,nobs)
            
            xk2 = xdaobs + L2*pk.reshape(1,3)
            ofk2 = obj_function(zobs,xk2,rk,nobs)
            
            if ofk1 > ofk2:
                ags = L1
                L1 = L2
                Li = bgs - ags
                L2 = ags + rgs*Li
            else:
                bgs = L2
                L2 = L1
                Li = bgs - ags
                L1 = ags + rgs**2*Li
                
        lr = 0.5*(ags + bgs)
            
        xnew = xold - lr*grad.flatten() #/np.linalg.norm(grad.flatten())
        
        print(q, ' ', lr, ' ', np.linalg.norm(grad))
        
        if q == 0:
            res0 = np.linalg.norm(grad)
    
        res_history[q,0] = q
        res_history[q,1] = np.linalg.norm(grad)/res0
        
        if np.linalg.norm(grad) < tolerance:
            break
    
        xold = np.copy(xnew)
    
    Xtrue_tmax1 = lorenz(X_init,sigma,rho,beta,dt,ntmax)
    Xda_tmax = lorenz(xnew,sigma,rho,beta,dt,ntmax)
    
    plot_lorenz(q,t_train,Xtrue_tmax1,ttmax,zobs,tobs,Xda_tmax,ttmax)
    plot_residual(res_history)

#%% dynamics learning rate method
if line_search == True:
    for q in range(10):
        xda = lorenz(xold,sigma,rho,beta,dt,nttrain)
        
        xdaobs = xda[ind]
            
        grad = fourdvar(zobs,xdaobs,dxk_h,rk,nobs)
        
        ofk = obj_function(zobs,xdaobs,rk,nobs)
        
        pk = -np.copy(grad)
        #lr = -0.5*ofk/(grad.T @ pk)
        #print(lr)
        lr = lr_init
        xkp = xdaobs + lr*pk.reshape(1,3)
        
        ofkp = obj_function(zobs,xkp,rk,nobs)
        
        while ofkp >= ofk - 0.5*lr*np.linalg.norm(grad):
            lr = tau*lr
            xkp = xdaobs + lr*pk.reshape(1,3)
            ofkp = obj_function(zobs,xkp,rk,nobs)
            print(lr, ' ', ofk, ' ', ofkp)
            
        xnew = xold - lr*grad.flatten() #/np.linalg.norm(grad.flatten())
        
        print(q, ' ', lr, ' ', np.linalg.norm(grad))
        
        if q == 0:
            res0 = np.linalg.norm(grad)
    
        res_history[q,0] = q
        res_history[q,1] = np.linalg.norm(grad)/res0
        
        if np.linalg.norm(grad) < tolerance:
            break
    
        xold = np.copy(xnew)
    
    Xtrue_tmax1 = lorenz(X_init,sigma,rho,beta,dt,ntmax)
    Xda_tmax = lorenz(xnew,sigma,rho,beta,dt,ntmax)
    
    plot_lorenz(q,t_train,Xtrue_tmax1,ttmax,zobs,tobs,Xda_tmax,ttmax)
    plot_residual(res_history)
    
#%%
if dyn_alpha == True:
    for q in range(max_iter):
        if rk4 == True:
            xda = lorenz_rk4(xold,sigma,rho,beta,dt,nttrain)
        else:
            xda = lorenz(xold,sigma,rho,beta,dt,nttrain)
            
        xdaobs = xda[ind]
        
        if rk4 == True:
            grad = fourdvar_rk4(zobs,xdaobs,dxk_h,rk,nobs)
        else:
            grad = fourdvar(zobs,xdaobs,dxk_h,rk,nobs)
            
        ofk = obj_function(zobs,xdaobs,rk,nobs)
        
        pk = -np.copy(grad)
        #lr = -0.5*ofk/(grad.T @ pk)
        #print(lr)
        #xkp = xdaobs + lr*pk.reshape(1,3)
        xkp = xdaobs + pk.reshape(1,3)
        
        ofkp = obj_function(zobs,xkp,rk,nobs)
        
        if mode == 0:
            lr = lr
        elif mode == 1:
            temp = grad.T @ pk
            #lr = -temp*lr**2/(2.0*(ofkp - lr*temp - ofk))
            lr = -temp/(2.0*(ofkp - temp - ofk))
            lr = lr[0]
        
        xnew = xold - lr*grad.flatten() #/np.linalg.norm(grad.flatten())
        
        print(q, ' ', lr, ' ', np.linalg.norm(grad))
        
        if q == 0:
            res0 = np.linalg.norm(grad)
    
        res_history[q,0] = q
        res_history[q,1] = np.linalg.norm(grad)/res0
        
        if np.linalg.norm(grad) < tolerance:
            break
    
        xold = np.copy(xnew)
    
    if rk4 == True:
        Xtrue_tmax1 = lorenz_rk4(X_init,sigma,rho,beta,dt,ntmax)
        Xda_tmax = lorenz_rk4(xnew,sigma,rho,beta,dt,ntmax)
    else:
        Xtrue_tmax1 = lorenz(X_init,sigma,rho,beta,dt,ntmax)
        Xda_tmax = lorenz(xnew,sigma,rho,beta,dt,ntmax)
        
    plot_lorenz(q,t_train,Xtrue_tmax1,ttmax,zobs,tobs,Xda_tmax,ttmax)
    plot_residual(res_history)