#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:07:20 2020

@author: suraj
"""
import numpy as np
np.random.seed(44)
import matplotlib.pyplot as plt

font = {'family' : 'Times New Roman',
        'size'   : 12}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%    
def plot(x,un,ue):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x,un[0,:],'b',label='Initial condition')
    ax.plot(x,ue,'k',label='Exact solution')
    ax.plot(x,un[-1,:],'r-.',label='Numerical solution')
    ax.legend()
    plt.show()
    fig.tight_layout()
    
def plot_final(x,un,uf,uw,istd,istart):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x,uw[-1,:],'k',lw=2,label='Wrong initial condition')
    ax.plot(x,uf[-1,:],'r',lw=4,label='4DVar initial condition')
    ax.plot(x,un[-1,:],'g--',lw=2,label='True initial condition')    
    ax.grid()
    ax.legend()
    plt.show()
    fig.tight_layout()
    fig.savefig('cgfinal_'+str(int(istart))+'_'+str(int(istd))+'.pdf')
    
def plot_da(x,uda0,udan,ut,z,istd,istart):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x,uda0,'k',label='Wrong initial condition')
    ax.plot(x,udan,'r',lw=4,label='4DVar initial condition')
    ax.plot(x,ut,'g--',lw=2,label='True initial condition')
#    ax.plot(x,z,'o',fillstyle='none',markersize=8,markeredgewidth=2,
#            color='b',label='Observations')
    ax.grid()
    ax.legend()
    plt.show()
    fig.tight_layout()
    fig.savefig('cginitial_'+str(int(istart))+'_'+str(int(istd))+'.pdf')
    
def plot_residual(res_history):
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5))
    ax.semilogy(res_history[:,0],res_history[:,1],'g-',label='')
    ax.set_xlabel('Itearation $(p)$')
    ax.set_ylabel('$f(x)$')
    ax.grid()
    fig.tight_layout()
    plt.show()
    fig.savefig('residual.pdf') 

#%%
def rhs_cs(nx,dx,a,u):
    r = np.zeros((nx+1))

    r[1:nx] = -a[1:nx]*(u[2:nx+1] - u[0:nx-1])/(2.0*dx)
        
    return r

def rk4(nx,dx,dt,a,u):
    r1 = rhs_cs(nx,dx,a,u)
    k1 = dt*r1
    
    r2 = rhs_cs(nx,dx,a,u+0.5*k1)
    k2 = dt*r2
    
    r3 = rhs_cs(nx,dx,a,u+0.5*k2)
    k3 = dt*r3
    
    r4 = rhs_cs(nx,dx,a,u+k3)
    k4 = dt*r4
    
    u = u + (k1 + 2.0*(k2 + k3) + k4)/6.0
    
    return u

def rk2(nx,dx,dt,a,u):
    r1 = rhs_cs(nx,dx,a,u)
    k1 = dt*r1
    
    r2 = rhs_cs(nx,dx,a,u+k1)
    k2 = dt*r2
    
    u = u + (k1 + k2)/2.0
    
    return u

def eu(nx,dx,dt,a,u):
    ap = np.where(a>0,a,0)
    an = np.where(a<0,a,0)
    
    v = np.copy(u)
    
    v[1:nx] = u[1:nx] - (ap[1:nx]*dt/dx)*(u[1:nx] - u[0:nx-1]) - \
              (an[1:nx]*dt/dx)*(u[2:nx+1] - u[1:nx])

    return v

#%%
def dxk_model_rk2(a,dt,dx,nx):
    
    dxk_m = np.zeros((nx+1,nx+1))
    i = 0
    dxk_m[i,i] = 1.0 
    
    i = 1
    c = a[i]*dt/(2.0*dx)
    dxk_m[i,i-1] = c
    dxk_m[i,i] = 1 -0.5*c**2
    dxk_m[i,i+1] = -c
    dxk_m[i,i+2] = 0.5*c**2
    
    for i in range(2,nx-1):
        c = a[i]*dt/(2.0*dx)
        dxk_m[i,i-1] = c
        dxk_m[i,i] = 1.0 - c**2
        dxk_m[i,i+1] = -c
        dxk_m[i,i-2] = 0.5*c**2
        dxk_m[i,i+2] = 0.5*c**2 
        
    i = nx-1
    c = a[i]*dt/(2.0*dx)
    dxk_m[i,i] = 1 -0.5*c**2
    dxk_m[i,i-1] = c
    dxk_m[i,i-2] = 0.5*c**2
    dxk_m[i,i+1] = -c
    
    i = nx
    dxk_m[i,i] = 1.0 
    
    return dxk_m

def dxk_model_eu(a,dt,dx,nx):
    ap = np.where(a>0,a,0)
    an = np.where(a<0,a,0)
    
    dxk_m = np.zeros((nx+1,nx+1))
    i = 0
    dxk_m[i,i] = 1.0 
    for i in range(1,nx):
        dxk_m[i,i-1] = ap[i+1]*dt/dx
        dxk_m[i,i] = 1.0 - ap[i+1]*dt/dx + an[i+1]*dt/dx
        dxk_m[i,i+1] = -an[i+1]*dt/dx
    i = nx
    dxk_m[i,i] = 1.0 
    
    return dxk_m

def dxk_obs(nx,u):
    dxh_h = 2.0*u*np.eye(nx+1)
    #dxh_h = np.eye(nx+1)
    return dxh_h

def obj_function(zk,xk,rk,nobs):
    obj_fun = 0.0
    for n in range(nobs):
        
        diff = zk[n,:].reshape(-1,1) - (xk[n,:].reshape(-1,1))**2
        obj_fun = obj_fun + diff.T @ np.linalg.inv(rk) @ diff
           
    return 0.5*obj_fun

def fourdvar(zobs,xdaobs,xda,rk,nobs,ind,nx,ischeme):
    f = np.zeros((nobs,nx+1))  
    
    nt = xda.shape[0]
    lagr = np.zeros((nt,nx+1))
    
    for k in range(nobs):
        xk = xdaobs[k,:].reshape(-1,1)
        zk = zobs[k,:].reshape(-1,1)
        hk = zk - xk**2
        dxk_h = dxk_obs(nx,xk)
        fk = dxk_h.T @ np.linalg.inv(rk) @ hk
        f[k,:] = fk.reshape(1,-1)
        
    lagr[-1] = f[-1,:]
    
    for k in range(nt-2,-1,-1):
        xk = xda[k,:].reshape(-1,1)
        lkp = lagr[k+1].reshape(-1,1)
        if ischeme == 1:
            dxk_m = dxk_model_eu(a,dt,dx,nx)
        if ischeme == 2:
            dxk_m = dxk_model_rk2(a,dt,dx,nx)
        lk = dxk_m.T @ lkp  
        lagr[k,:] = lk.reshape(1,-1)
        if k in ind:
            lagr[k,:] = lk.reshape(1,-1) + f[int(k/freq)-1,:]
    
    if ischeme == 1:
        dx0_m = dxk_model_eu(a,dt,dx,nx)
    if ischeme == 2:
        dx0_m = dxk_model_rk2(a,dt,dx,nx)

    grad = -dx0_m.T @ lagr[0,:].reshape(-1,1)
    
    return grad 

#%%
istd = 2 # std = 0.01, 0.1
istart = 2 # ic = 1.2sin(x), 1.0
ischeme = 2 # scheme = Euler, RK2

nx = 100
xl = 0
xr = 2*np.pi
dx = (xr-xl)/nx
x = np.linspace(xl, xr, nx+1)

tm = 2.0
dt = 0.01
nt = int(tm/dt)
t = np.linspace(0,tm,nt+1)

tda = 1.0
nda = int(tda/dt)
freq = 10
nobs = int(tda/(freq*dt))
ind = [freq*i for i in range(1,nobs+1)]
tobs = t[ind]

mu = 0.0
if istd == 1:
    std = 0.01
elif istd == 2:
    std = 0.1
    
var = std**2
rk = var*np.eye(nx+1,nx+1)

max_iter = 50
tolerance = 1e-2
mode = 1 # 0: Static, 1 Dynamics
lr = 0.01

res_history = np.zeros((max_iter,2))

#%%
un = np.zeros([nt+1, nx+1])
a = np.sin(x)
u = np.sin(x)
u[0] = 0.0
u[-1] = 0.0

un[0,:] = u
for k in range(1,nt+1):
    #un[k,:] = rk4(nx,dx,dt,a,un[k-1,:])
    if ischeme == 1:
        un[k,:] = eu(nx,dx,dt,a,un[k-1,:])
    if ischeme == 2:
        un[k,:] = rk2(nx,dx,dt,a,un[k-1,:])
    
ue = 2*np.exp(tm)*np.sin(x)/(1.0 + np.exp(2.0*tm) + np.cos(x)*(np.exp(2.0*tm) - 1.0))

plot(x,un,ue)

uobs = un[ind,:]
zobs = uobs**2 + np.random.normal(mu,std,[nobs,nx+1])
#zobs = uobs + np.random.normal(mu,std,[nobs,nx+1])

if istart == 1:
    udao = 1.2*np.sin(x)
elif istart == 2:
    udao = 1.0*np.ones(nx+1)
    
udao[0] = 0.0
udao[-1] = 0.0

uda = np.zeros([nda+1, nx+1])
udap = np.zeros([nda+1, nx+1])
udat = np.zeros([nda+1, nx+1])

uda0 = np.copy(udao)

uf = np.zeros([nt+1, nx+1])
uw= np.zeros([nt+1, nx+1])
#plot_da(x,uda0, uobs[-1,:], zobs[-1,:])

#%%
for q in range(max_iter):
    uda[0,:] = udao
    for k in range(1,nda+1):
        #uda[k,:] = rk4(nx,dx,dt,a,uda[k-1,:])
        if ischeme == 1:
            uda[k,:] = eu(nx,dx,dt,a,uda[k-1,:])
        if ischeme == 2:
            uda[k,:] = rk2(nx,dx,dt,a,uda[k-1,:])
    
    udaobs = uda[ind]
    
    ofk = obj_function(zobs,udaobs,rk,nobs)
    if q == 0:
        grad = fourdvar(zobs,udaobs,uda,rk,nobs,ind,nx,ischeme)
        pk = -np.copy(grad)
        resk = -np.copy(grad)
        
    lr = -0.5*ofk/(grad.T @ pk)
    
    udaop = udao + lr[0]*pk.flatten()
    udap[0,:] = udaop
    for k in range(1,nda+1):
        #un[k,:] = rk4(nx,dx,dt,a,un[k-1,:])
        if ischeme == 1:
            udap[k,:] = eu(nx,dx,dt,a,udap[k-1,:])
        if ischeme == 2:
            udap[k,:] = rk2(nx,dx,dt,a,udap[k-1,:])

    udaobsp = udap[ind]
    
    ofkp = obj_function(zobs,udaobsp,rk,nobs)
    
    if mode == 0:
        lr = lr
    elif mode == 1:
        temp = grad.T @ pk
        lr = -temp*lr**2/(2.0*(ofkp - lr*temp - ofk))
        lr = lr[0]
    
    udan = udao + lr*pk.flatten() #/np.linalg.norm(grad.flatten())
    
    udat[0,:] = udan
    for k in range(1,nda+1):
        #un[k,:] = rk4(nx,dx,dt,a,un[k-1,:])
        if ischeme == 1:
            udat[k,:] = eu(nx,dx,dt,a,udat[k-1,:])
        if ischeme == 2:
            udat[k,:] = rk2(nx,dx,dt,a,udat[k-1,:])
    
    udaobst = udat[ind]
    gradp = fourdvar(zobs,udaobst,udat,rk,nobs,ind,nx,ischeme)
    
    beta_pr = np.dot(gradp.T,(gradp - grad))/(np.dot(grad.T,grad))
    beta= np.max((0.0, beta_pr[0,0]))
    
    beta = np.dot(gradp.T,grad)/(np.dot(grad.T,grad))
    grad = np.copy(gradp)
    
    pk = -gradp + beta*pk
            
    print(q, ' ', lr, ' ', np.linalg.norm(udan-udao), ' ', np.linalg.norm(grad))
    
    if q == 0:
        res0 = np.linalg.norm(grad)

    res_history[q,0] = q
    res_history[q,1] = np.linalg.norm(grad)/res0
    
    if np.linalg.norm(grad) < tolerance or np.linalg.norm(udan-udao) < tolerance:
        break

    udao = np.copy(udan)

uf[0,:] = udao
for k in range(1,nt+1):
    #uda[k,:] = rk4(nx,dx,dt,a,uda[k-1,:])
    if ischeme == 1:
        uf[k,:] = eu(nx,dx,dt,a,uf[k-1,:])
    if ischeme == 2:
        uf[k,:] = rk2(nx,dx,dt,a,uf[k-1,:])

uw[0,:] = uda0
for k in range(1,nt+1):
    #uda[k,:] = rk4(nx,dx,dt,a,uda[k-1,:])
    if ischeme == 1:
        uw[k,:] = eu(nx,dx,dt,a,uw[k-1,:])
    if ischeme == 2:
        uw[k,:] = rk2(nx,dx,dt,a,uw[k-1,:])

plot_final(x,un,uf,uw,istd,istart)
plot_da(x,uda0, udan, un[0,:],zobs[0,:],istd,istart)
plot_residual(res_history)