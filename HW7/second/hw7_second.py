#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:12:14 2020

@author: suraj
"""

import numpy as np
#np.random.seed(1)
import matplotlib.pyplot as plt

font = {'family' : 'Times New Roman',
        'size'   : 12}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
def rhs(rho0,k,g,x):
    r = np.zeros(3)
    r[0] = x[1]
    r[1] = 0.5*rho0*np.exp(-x[0]/k)*x[1]**2*x[2] - g
    r[2] = 0.0
    
    return r
    
def rk2(rho0,k,g,dt,x):
    r1 = rhs(rho0,k,g,x)
    k1 = dt*r1
    
    r2 = rhs(rho0,k,g,x+k1)
    k2 = dt*r2
    
    x = x + (k1 + k2)/2.0
    
    return x

def euler(rho0,k,g,dt,x):
    r = rhs(rho0,k,g,x)
    
    x = x + dt*r
    
    return x

def jacobian_model(rho0,k,g,dt,x):
    Dm = np.zeros((3,3))
    
    Dm[0,0] = 1.0
    Dm[0,1] = dt
    Dm[0,2] = 0.0
    
    #temp = 0.5*rho0*np.exp(-x[0]/k)*x[1]*x[2]
    Dm[1,0] = -(rho0*dt/(2.0*k)) * np.exp(-x[0]/k) * (x[1]**2) * x[2]
    Dm[1,1] = 1.0 + (dt*rho0/2.0) * np.exp(-x[0]/k) * (2*x[1]) * x[2]
    Dm[1,2] = (dt*rho0/2.0) * np.exp(-x[0]/k) * (x[1]**2)
    
    Dm[2,0] = 0.0
    Dm[2,1] = 0.0
    Dm[2,2] = 1.0
    
    return Dm

def hessian_model(rho0,k,g,dt,x):
    D2_m1 = np.zeros((3,3))
    D2_m2 = np.zeros((3,3))
    D2_m3 = np.zeros((3,3))
        
    
    D2_m2[0,0] = (rho0*dt/(2*k**2)) * np.exp(-x[0]/k) * (x[1]**2) * x[2]
    D2_m2[0,1] = -(rho0*dt/(2*k)) * np.exp(-x[0]/k) * (2*x[1]) * x[2]
    D2_m2[0,2] = -(rho0*dt/(2*k)) * np.exp(-x[0]/k) * (x[1]**2) 

    D2_m2[1,0] = D2_m2[0,1]
    D2_m2[1,1] = (dt*rho0/2.0) * np.exp(-x[0]/k) * (2) * x[2]
    D2_m2[1,2] = (dt*rho0/2.0) * np.exp(-x[0]/k) * (2*x[1]) 
    
    D2_m2[2,0] = D2_m2[0,2]
    D2_m2[2,1] = D2_m2[1,2]
    D2_m2[2,2] = 0.0
    
    return D2_m1, D2_m2, D2_m3

def jacobian_observations(M,a,x):
    Dh = np.zeros((1,3))
    
    Dh[0,0] = (x[0]-a)/(np.sqrt(M**2 + (x[0]-a)**2))
    
    return Dh

def hessian_observations(M,a,x):
    D2_h1 = np.zeros((3,3))
    
    D2_h1[0,0] = -(x[0]-a)**2/(M**2 + (x[0]-a)**2)**(3/2) + 1.0/(np.sqrt(M**2 + (x[0]-a)**2))
     
    return D2_h1   
    
#%%
rho0 = 2.0
g = 32.2
k = 20000
M = 100000
a = 100000
mu = 0.0
std2_o = 1e4 #10000
std1_o = np.sqrt(std2_o)
dt = 1.0/64.0
tmax = 60
nt = int(tmax/dt)
t = np.linspace(0,tmax,nt+1)
freq = 32
zobs = np.zeros(int(nt/freq))
tobs = np.linspace(1,tmax,int(nt/freq))

P = np.array([[1e6,0,0],[0,4e6,0],[0,0,10]]) # initial covariance
q = 0.000
Q = q*np.eye(3)
#Q[0,0] = 0.1
#Q[1,1] = 0.01
#Q[2,2] = 0.001
R = std2_o #np.eye(3)
I = np.eye(3)

x =  np.array([300000,-20000,0.001])
xtrue = np.zeros((nt+1,3))
xtrue[0,:] = x

j = 0
for n in range(1,nt+1):
    xn = euler(rho0,k,g,dt,x)
    xtrue[n,:] = xn
    x = np.copy(xn)
    
    if n % freq == 0:
        # get observation    
        z = np.sqrt((M**2 + (x[0]-a)**2)) +  std1_o*np.random.randn(1)
        zobs[j] = z
        j = j+1

x =  np.array([300000,-20000,0.00003])    
xw = np.zeros((nt+1,3))
xw[0,:] = x
for n in range(1,nt+1):
    xn = euler(rho0,k,g,dt,x)
    xw[n,:] = xn
    x = np.copy(xn)
        
#%%
# initial condition
x =  np.array([300000,-20000,0.00003])
xda = np.zeros((nt+1,3))
xda[0,:] = x
    
s = 0
j = 0
for n in range(1,nt+1):
    D2_m1, D2_m2, D2_m3 =  hessian_model(rho0,k,g,dt,x)
    second_term = np.zeros(3)
    
    second_term[0] = np.trace(D2_m1 @ P) #np.trace(D2_m1 @ P)
    second_term[1] = np.trace(D2_m2 @ P) #np.trace(D2_m2 @ P)
    second_term[2] = np.trace(D2_m3 @ P) #np.trace(D2_m3 @ P)
    
    xn = euler(rho0,k,g,dt,x)
    
    xn = xn + 0.5*second_term
    
    # forecast covariance step
    Dm = jacobian_model(rho0,k,g,dt,x)
    P = Dm @ P @ Dm.T + Q
    P = 0.5* (P + P.T)
    
    if n % freq == 0:       
        # data assimilation step
        Dh = jacobian_observations(M,a,x)
        K = P @ Dh.T @ np.linalg.pinv((Dh @ P @ Dh.T + R))
        
        z = zobs[s]
        D2_h1 = hessian_observations(M,a,x)
        xn = xn + K.reshape(-1,) * ((z - np.sqrt((M**2 + (xn[0]-a)**2))) - 0.5*np.trace(D2_h1 @ P)) #D2_h1[0,0] #
        s = s+1
        
        P = (I - K @ Dh) @ P #@ (I - K @ Dh) + R* K @ K.T
        #print(z)
    
    xda[n,:] = xn
    x = np.copy(xn)    

#%%     
fig, ax = plt.subplots(3,1,sharex=True,figsize=(8,6))

ax[0].plot(tobs,zobs,'mo',lw=3)

for i in range(3):
    ax[i].plot(t,xtrue[:,i],'r-',lw=3)
    ax[i].plot(t,xw[:,i],'k-')
    ax[i].plot(t,xda[:,i],'g-')

ax[0].set_ylim([-75000,315000])        
ax[1].set_ylim([-25000,2000])
ax[2].set_ylim([-0.001,0.0015])        
ax[i].set_xlabel(r'$t$')
line_labels = ['Observations', 'True','Wrong', 'Extended Kalman']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
fig.tight_layout()
plt.show() 
fig.savefig('qs_'+str(q)+'_so_'+str(std2_o)+'.pdf')