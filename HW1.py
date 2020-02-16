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

#%% Problem 1
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
std = np.array([0.0,0.1,0.4,0.8,1.0,1.2])
error_norm = np.zeros((std.shape[0],4))

for n in range(std.shape[0]):
    v = np.random.normal(mu,std[n],5)
    
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
    
    error_norm[n,0] = np.linalg.norm(z-np.dot(H,xls_p))
    error_norm[n,1] = np.linalg.norm(z-np.dot(H,xls_lu))
    error_norm[n,2] = np.linalg.norm(z-np.dot(H,xls_qr))
    error_norm[n,3] = np.linalg.norm(z-np.dot(H,xls_svd))

fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(12,4))

ax[0].plot(std,error_norm[:,1],'ro-',label='LU')
ax[0].set_title('LU decomposition')
ax[0].set_xlabel('$\sigma^2$')
ax[0].set_ylabel('$||Z-Hx_{LS}||_2$')
ax[0].legend()

ax[1].plot(std,error_norm[:,2],'bv-',label='QR')
ax[1].set_title('QR decomposition')
ax[1].set_xlabel('$\sigma^2$')
ax[1].set_ylabel('$||Z-Hx_{LS}||_2$')
ax[1].legend()

ax[2].plot(std,error_norm[:,3],'gs-',label='SVD')
ax[2].set_title('SVD decomposition')
ax[2].set_xlabel('$\sigma^2$')
ax[2].set_ylabel('$||Z-Hx_{LS}||_2$')
ax[2].legend()

fig.tight_layout()
plt.show()
fig.savefig('probem1.pdf')

#%% Problem 2
nx = 4
ny = 4
xl = 3
yl = 3

dx = xl/(nx-1)
dy = yl/(ny-1)

nobs = 4
nvar = nx*ny

mu = 0
std = 1

v = np.random.normal(mu,std,nobs)

z = 75 + v

x = np.linspace(0,xl,nx)
y = np.linspace(0,yl,ny)

H = np.zeros((nobs,nvar))
Zxy = np.zeros((nobs,2))

for n in range(nobs):
    zx = np.random.rand(1)*xl
    zy = np.random.rand(1)*yl
    
    Zxy[n,0] = zx
    Zxy[n,1] = zy
    
    
    j = int(zx/dx) 
    i = int(zy/dy)

    a = (zx - x[j])/dx
    abar = (x[j+1] - zx)/dx
    b = (zy - y[i])/dy
    bbar = (y[i+1] - zy)/dy

    k = i*nx + j
    
    H[n,k] = abar*bbar
    H[n,k+1] = a*bbar
    H[n,k+nx] = abar*b
    H[n,k+nx+1] = a*b

# pseido-inverse for checking
xls_p = np.dot(np.linalg.pinv(H),z)

# LU decomposition
xls_lu = multi_dot([H.T, inv(np.dot(H,H.T)), z])

#g = multi_dot([inv(np.dot(L.T,L)), L.T, z])
#xls_lu = multi_dot([inv(U), g])

# QR decomposition
Q,R = np.linalg.qr(H.T)
xls_qr = np.dot(Q,np.dot(inv(R.T),z))

# SVD decomposition
U, s, Vh = np.linalg.svd(H)
S  = np.dot(np.eye(nvar,nobs),np.diag(1/s))
xls_svd = multi_dot([Vh.T, S, U.T, z])

error_norm = np.zeros((4))
error_norm[0] = np.linalg.norm(z-np.dot(H,xls_p))
error_norm[1] = np.linalg.norm(z-np.dot(H,xls_lu))
error_norm[2] = np.linalg.norm(z-np.dot(H,xls_qr))
error_norm[3] = np.linalg.norm(z-np.dot(H,xls_svd))

np.savetxt('problem2.csv',error_norm,delimiter='')


#%% Problem 3
nx = 101
ny = 101
xl = 1
yl = 1

nobsarray = np.array([20,2000])
nvar = nx*ny

mu = 0
std = 0.1

dx = xl/(nx-1)
dy = yl/(ny-1)

x = np.linspace(0,xl,nx)
y = np.linspace(0,yl,ny)
error_norm = np.zeros((2,5))

for p in range(2):
    nobs = nobsarray[p]
    error_norm[p,0] = nobs
    z = np.zeros(nobs)
    H = np.zeros((nobs,nvar))
    Zxy = np.zeros((nobs,2))
    
    for n in range(nobs):
        zx = np.random.rand(1)*xl
        zy = np.random.rand(1)*yl
        
        z[n] = 2*zx + 4*zy + zx*zy + np.random.normal(mu,std,1)
        
        Zxy[n,0] = zx
        Zxy[n,1] = zy
        
        
        j = int(zx/dx) 
        i = int(zy/dy)
    
        a = (zx - x[j])/dx
        abar = (x[j+1] - zx)/dx
        b = (zy - y[i])/dy
        bbar = (y[i+1] - zy)/dy
    
        k = i*nx + j
        
        H[n,k] = abar*bbar
        H[n,k+1] = a*bbar
        H[n,k+nx] = abar*b
        H[n,k+nx+1] = a*b
    
    # pseido-inverse for checking
    xls_p = np.dot(np.linalg.pinv(H),z)
    
    # LU decomposition
    xls_lu = multi_dot([H.T, inv(np.dot(H,H.T)), z])
    
    #g = multi_dot([inv(np.dot(L.T,L)), L.T, z])
    #xls_lu = multi_dot([inv(U), g])
    
    # QR decomposition
    Q,R = np.linalg.qr(H.T)
    xls_qr = np.dot(Q,np.dot(inv(R.T),z))
    
    # SVD decomposition
    U, s, Vh = np.linalg.svd(H)
    S  = np.dot(np.eye(nvar,nobs),np.diag(1/s))
    xls_svd = multi_dot([Vh.T, S, U.T, z])
    
    error_norm[p,1] = np.linalg.norm(z-np.dot(H,xls_p))
    error_norm[p,2] = np.linalg.norm(z-np.dot(H,xls_lu))
    error_norm[p,3] = np.linalg.norm(z-np.dot(H,xls_qr))
    error_norm[p,4] = np.linalg.norm(z-np.dot(H,xls_svd))

    X,Y = np.meshgrid(x,y)
    zfield = np.reshape(xls_p,[nx,ny])
    ztrue = 2*X + 4*Y + X*Y
    
    fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(12,10))
    ax = axs.flat
    cs0 = ax[0].contourf(X,Y,ztrue, 20, cmap = 'jet',vmin=np.min(ztrue),vmax=np.max(ztrue),extend='both')
    ax[0].set_aspect(1.0)
    ax[0].set_title('True $[z(x,y)=2x+4y+xy]$')
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$y$')
    
    cs = ax[1].contourf(X,Y,np.reshape(xls_lu,[nx,ny]), 20, cmap = 'jet',vmin=np.min(ztrue),vmax=np.max(ztrue))
    ax[1].set_aspect(1.0)
    ax[1].set_title('LU decomposition')
    ax[1].set_xlabel('$x$')
    ax[1].set_ylabel('$y$')
    
    cs = ax[2].contourf(X,Y,np.reshape(xls_qr,[nx,ny]), 20, cmap = 'jet',vmin=np.min(ztrue),vmax=np.max(ztrue))
    ax[2].set_aspect(1.0)
    ax[2].set_title('QR decomposition')
    ax[2].set_xlabel('$x$')
    ax[2].set_ylabel('$y$')
    
    cs = ax[3].contourf(X,Y,np.reshape(xls_svd,[nx,ny]), 20, cmap = 'jet',vmin=np.min(ztrue),vmax=np.max(ztrue))
    ax[3].set_aspect(1.0)
    ax[3].set_title('SVD decomposition')
    ax[3].set_xlabel('$x$')
    ax[3].set_ylabel('$y$')
    
    cbar_ax = fig.add_axes([0.2, -0.05, 0.6, 0.04])
    #cs.set_clim(np.min(ztrue), np.max(ztrue)) 
    cbar = fig.colorbar(cs0, cax=cbar_ax,ticks=np.linspace(np.min(ztrue), np.max(ztrue), 8),
                        orientation='horizontal')
    
    #fig.colorbar(cs)
    #ax[0].scatter(Zxy[:,0],Zxy[:,1],marker='*',s=80,color='red')
    fig.tight_layout()
    plt.show()
    fig.savefig('problem3_'+str(nobs)+'.pdf',bbox_inches = 'tight')

np.savetxt('problem3.csv',error_norm,delimiter='')

#%%  Problem 4
nx = 4
ny = 4
xl = 3
yl = 3

dx = xl/(nx-1)
dy = yl/(ny-1)

nobs = 18
nvar = nx*ny

mu = 0
std = 0.5

v = np.random.normal(mu,std,nobs)

z = 70 + v

z = z.reshape([-1,1])

x = np.linspace(0,xl,nx)
y = np.linspace(0,yl,ny)

H = np.zeros((nobs,nvar))
Zxy = np.zeros((nobs,2))

k = 0
for j in range(nx-1):
    for i in range(ny-1):
        for p in range(2):
            zx = np.random.rand(1)
            zy = np.random.rand(1)
            Zxy[k,0] = x[j] + zx
            Zxy[k,1] = y[i] + zx
            k = k + 1            

for n in range(nobs):    
    zx = Zxy[n,0]
    zy = Zxy[n,1]
    
    j = int(zx/dx) 
    i = int(zy/dy)

    a = (zx - x[j])/dx
    abar = (x[j+1] - zx)/dx
    b = (zy - y[i])/dy
    bbar = (y[i+1] - zy)/dy

    k = i*nx + j
    
    H[n,k] = abar*bbar
    H[n,k+1] = a*bbar
    H[n,k+nx] = abar*b
    H[n,k+nx+1] = a*b       

# pseido-inverse for checking
xls_p = np.dot(np.linalg.pinv(H),z)
xls_p = xls_p.reshape(-1,1)

maxit = 100000
tol = 1e-6


# Steepest desceny algorithm
b = np.dot(H.T,z).reshape(-1,1)
A = np.dot(H.T,H)
xo = np.zeros((nvar,1))
xn = np.zeros((nvar,1))
r = b - np.dot(A,xo)
res_history_sd = 0.5*np.dot((z-np.dot(H,xn)).T,(z-np.dot(H,xn)))

for k in range(maxit):
    alpha = np.dot(r.T,r)/multi_dot([r.T,A,r])
    xn = xo + alpha*r
    r = r - alpha*np.dot(A,r)
    print(k, ' ', np.dot(r.T,r)[0,0])
    res_history_sd = np.vstack((res_history_sd, 0.5*np.dot((z-np.dot(H,xn)).T,(z-np.dot(H,xn)))))
    if np.dot(r.T,r) < tol:
        break
    xo = xn
    
# Steepest-descent gradient method solution    
xsd = xn

# Conjugate Gradient algorithm
b = np.dot(H.T,z).reshape(-1,1)
A = np.dot(H.T,H)
xo = np.zeros((nvar,1))
xn = np.zeros((nvar,1))
r = b - np.dot(A,xo)
p = np.copy(r)
res_history_cg = 0.5*np.dot((z-np.dot(H,xn)).T,(z-np.dot(H,xn)))

for k in range(maxit):
    alpha = np.dot(p.T,r)/multi_dot([p.T,A,p])
    xn = xo + alpha*p
    r = r - alpha*np.dot(A,p)
    print(k, ' ', np.dot(r.T,r)[0,0])
    res_history_cg = np.vstack((res_history_cg, 0.5*np.dot((z-np.dot(H,xn)).T,(z-np.dot(H,xn)))))
    if np.dot(r.T,r) < tol:
        break
    beta = - multi_dot([r.T,A,p])/multi_dot([p.T,A,p])
    p = r + beta*p
    xo = xn
    

xcg = xn

# plotting
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5))
ax.plot(xls_p,'ro-',label='Least square (pinv)')
ax.plot(xsd,'bv--',label='Steepest descent')
ax.plot(xcg,'g*-.',label='Conjugate gradient')
ax.legend()
plt.show()

iteration_sd = np.linspace(1,res_history_sd.shape[0],res_history_sd.shape[0])
iteration_cg = np.linspace(1,res_history_cg.shape[0],res_history_cg.shape[0])

fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(11,5))
ax[0].semilogy(iteration_sd,res_history_sd,'ro-',label='Steepest descent')
ax[0].semilogy(iteration_cg,res_history_cg,'bv-',label='Conjugate gradient')
ax[0].legend()
ax[0].set_xlabel('Itearation $(k)$')
ax[0].set_ylabel('$f(x)$')
ax[1].semilogy(iteration_sd,res_history_sd,'ro-',label='Steepest descent')
ax[1].semilogy(iteration_cg,res_history_cg,'bv-',label='Conjugate gradient')
ax[1].legend()
ax[1].set_xlabel('Itearation $(k)$')
ax[1].set_ylabel('$f(x)$')
ax[1].set_xlim([1,50])
fig.tight_layout()
plt.show()
fig.savefig('problem4.pdf')