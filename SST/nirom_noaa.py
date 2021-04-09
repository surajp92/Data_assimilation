#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:55:27 2021

@author: suraj
"""
import random
random.seed(10)

import numpy as np
np.random.seed(10)

import tensorflow as tf
# tf.random.set_seed(0)

from numpy import linalg as LA

import time as tm

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.animation as animation

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from tensorflow.keras.regularizers import l2
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

# font = {'family' : 'Times New Roman',
#         'size'   : 16}    
# plt.rc('font', **font)

# #'weight' : 'bold'

# import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

import h5py
from tqdm import tqdm as tqdm

#%%
def POD(u,R): #Basis Construction
    n,ns = u.shape
    U,S,Vh = LA.svd(u, full_matrices=False)
    Phi = U[:,:R]  
    L = S**2
    #compute RIC (relative inportance index)
    RIC = sum(L[:R])/sum(L)*100   
    return Phi,L,RIC

def PODproj(u,Phi): #Projection
    a = np.dot(u.T,Phi)  # u = Phi * a.T
    return a

def PODrec(a,Phi): #Reconstruction    
    u = np.dot(Phi,a.T)    
    return u

def create_training_data_lstm(training_set, m, n, lookback):
    ytrain = [training_set[i+1] for i in range(lookback-1,m-1)]
    ytrain = np.array(ytrain)
    xtrain = np.zeros((m-lookback,lookback,n))
    for i in range(m-lookback):
        a = training_set[i]
        for j in range(1,lookback):
            a = np.vstack((a,training_set[i+j]))
        xtrain[i] = a

    return xtrain, ytrain

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


#%%
f = h5py.File('./sst_weekly.mat','r') # can be downloaded from https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu?usp=sharing
lat = np.array(f['lat'])
lon = np.array(f['lon'])
sst = np.array(f['sst'])
time = np.array(f['time'])

sst2 = np.zeros((len(sst[:,0]),len(lat[0,:]),len(lon[0,:])))
for i in tqdm(range(len(sst[:,0]))):
    sst2[i,:,:] = np.flipud((sst[i,:].reshape(len(lat[0,:]),len(lon[0,:]),order='F')))
    
#%%
sst_no_nan = np.nan_to_num(sst)
sst = sst.T

#%%
num_samples = sst.shape[1]

for i in range(num_samples):
    nan_array = np.isnan(sst[:,i])
    not_nan_array = ~ nan_array
    array2 = sst[:,i][not_nan_array]
    print(i, array2.shape[0])
    if i == 0:
        num_points = array2.shape[0]
        sst_masked = np.zeros((array2.shape[0],num_samples))
    sst_masked[:,i] = array2

#%%
ns = num_samples
nr = 5
num_samples_train = 854
lookback = 8

t = np.linspace(1,num_samples_train,num_samples_train)
sst_masked_small = sst_masked[:,:num_samples_train]
sst_average_small = np.sum(sst_masked_small,axis=1,keepdims=True)/num_samples_train

#%%
sst_masked_small_fluct = sst_masked_small - sst_average_small    

#%%
PHIw, L, RIC  = POD(sst_masked_small_fluct, nr)     

L_per = np.zeros(L.shape)
for n in range(L.shape[0]):
    L_per[n] = np.sum(L[:n],axis=0,keepdims=True)/np.sum(L,axis=0,keepdims=True)*100

#%%
k = np.linspace(1,num_samples_train,num_samples_train)
fig, axs = plt.subplots(1, 1, figsize=(7,5))#, constrained_layout=True)
axs.loglog(k,L_per, lw = 2, marker="o", linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
axs.set_xlim([1,ns])
axs.axvspan(0, nr, alpha=0.2, color='red')
fig.tight_layout()
plt.show()

#%%
at = PODproj(sst_masked_small_fluct, PHIw)

fig, ax = plt.subplots(nrows=3,ncols=2,figsize=(12,8),sharex=True)
ax = ax.flat
nrs = at.shape[1]

for i in range(nrs):
    ax[i].plot(t,at[:,i],'k',label=r'True Values')
#    ax[i].set_xlabel(r'$t$',fontsize=14)
    ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
    ax[-1].set_xlim([t[0],t[-1]])

ax[-2].set_xlabel(r'$t$',fontsize=14)    
ax[-1].set_xlabel(r'$t$',fontsize=14)    
fig.tight_layout()

fig.subplots_adjust(bottom=0.1)
line_labels = ["True"]#, "ML-Train", "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
plt.show()

#%%
tfluc = PODrec(at,PHIw)
T = tfluc + sst_average_small

#%%
aa = np.zeros(not_nan_array.shape[0])
aa[aa == 0] = 'nan'
aa[not_nan_array] = T[:,0]
trec = np.flipud((aa.reshape(len(lat[0,:]),len(lon[0,:]),order='F')))

fig,axs = plt.subplots(2,1, figsize=(10,8))

current_cmap = plt.cm.get_cmap('jet')
current_cmap.set_bad(color='white',alpha=1.0)

cs = axs[0].imshow(sst2[0,:,:],cmap='jet')
fig.colorbar(cs, ax=axs[0], orientation='vertical',shrink=1.0)

cs = axs[1].imshow(trec,cmap='jet')
fig.colorbar(cs, ax=axs[1], orientation='vertical',shrink=1)

fig.tight_layout()
plt.show()    

#%%
diff = trec - sst2[0,:,:]    
nan_array_2d = np.isnan(diff)
not_nan_array_2d = ~ nan_array_2d
diff_no_nan = diff[not_nan_array_2d]
l2_norm = np.linalg.norm(diff_no_nan)/np.sqrt(diff_no_nan.shape[0])

#%%
atrain = at[:num_samples_train,:]
m,n = atrain.shape

atrain_max = np.max(atrain, axis=0, keepdims=True)
atrain_min = np.min(atrain, axis=0, keepdims=True)

training_set = (2.0*atrain - (atrain_max + atrain_min))/(atrain_max - atrain_min)

#%%
data_sc, labels_sc = create_training_data_lstm(training_set, m, n, lookback)
xtrain, xvalid, ytrain, yvalid = train_test_split(data_sc, labels_sc, test_size=0.3 , shuffle= True)

#%%
Training = True
TF1 = True

if TF1:
    model_name = 'nirom_noaa.hd5'
else:
    model_name = 'nirom_noaa'
    
if Training:
    training_time_init = tm.time()
    
    input = Input(shape=(lookback,nr))
    a = LSTM(32, return_sequences=True)(input)
    
    x = LSTM(64, return_sequences=True,activation='relu', kernel_initializer='glorot_normal')(a) # main1 
    a = LSTM(64, return_sequences=True,activation='relu', kernel_initializer='glorot_normal')(a) # skip1
    
    x = LSTM(64, return_sequences=True,activation='relu', kernel_initializer='glorot_normal')(x) # main1
    
    b = Add()([a,x]) # main1 + skip1
    
    x = LSTM(64, return_sequences=True,activation='relu', kernel_initializer='glorot_normal')(b) # main2
    b = LSTM(64, return_sequences=True,activation='relu', kernel_initializer='glorot_normal')(b) # skip2
    
    x = LSTM(64, return_sequences=True,activation='relu', kernel_initializer='glorot_normal')(x) # main2
    
    c = Add()([b,x]) # main2 + skip2
    
    x = LSTM(64, return_sequences=False)(c)
    
    x = Dense(n, activation='linear')(x)
    model = Model(input, x)
    
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[coeff_determination])
    
    history = model.fit(xtrain, ytrain, epochs=600, batch_size=32, validation_data= (xvalid,yvalid))
    model.save(model_name)
    
    total_training_time = tm.time() - training_time_init
    print('Total training time=', total_training_time)
    cpu = open("a_cpu.txt", "w+")
    cpu.write('training time in seconds =')
    cpu.write(str(total_training_time))
    cpu.write('\n')
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    avg_mae = history.history['coeff_determination']
    val_avg_mae = history.history['val_coeff_determination']
    epochs = range(1, len(loss) + 1)
    
    plt.figure()
    plt.semilogy(epochs, loss, 'b', label='Training loss')
    plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.semilogy(epochs, avg_mae, 'b', label=f'Average $R_2$')
    plt.semilogy(epochs, val_avg_mae, 'r', label=f'Validation Average $R_2$')
    plt.title('Evaluation metric')
    plt.legend()
    plt.show()


model = load_model(model_name, custom_objects={'coeff_determination': coeff_determination})

#%%
sst_masked_all = sst_masked[:,:num_samples]
sst_average_all = np.sum(sst_masked_all,axis=1,keepdims=True)/num_samples_train

sst_masked_all_fluct = sst_masked_all - sst_average_small

at = PODproj(sst_masked_all_fluct, PHIw)
testing_set = (2.0*at - (atrain_max + atrain_min))/(atrain_max - atrain_min)


#%%
m,n = testing_set.shape
ytest = np.zeros((1,lookback,n))
ytest_ml = np.zeros((m,n))

# create input at t = 0 for the model testing
for i in range(lookback):
    ytest[0,i,:] = testing_set[i]
    ytest_ml[i] = testing_set[i]

#%%
testing_time_init = tm.time()

# predict results recursively using the model
for i in range(lookback,m):
    ytest_ml[i] = model.predict(ytest)
    ytest[0,:-1,:] = ytest[0,1:,:]
    ytest[0,-1,:] = ytest_ml[i] 

# unscaling
ytest_ml = 0.5*(ytest_ml*(atrain_max - atrain_min) + (atrain_max + atrain_min))
    
#%%
total_testing_time = tm.time() - testing_time_init
print('Total testing time=', total_testing_time)
cpu.write('testing time in seconds = ')
cpu.write(str(total_testing_time))
cpu.close()

#%%
t = np.linspace(1,num_samples,num_samples)
fig, ax = plt.subplots(nrows=nr,ncols=1,figsize=(12,8),sharex=True)
ax = ax.flat
nrs = at.shape[1]

for i in range(nrs):
    ax[i].plot(t,at[:,i],'k',label=r'True Values')
    ax[i].plot(t,ytest_ml[:,i],'b-',label=r'ML ')
    ax[i].set_xlabel(r'$t$',fontsize=14)
    ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
    ax[-1].set_xlim([t[0],t[-1]])
    ax[i].axvspan(0, t[num_samples_train], alpha=0.2, color='darkorange')

ax[-2].set_xlabel(r'$t$',fontsize=14)    
ax[-1].set_xlabel(r'$t$',fontsize=14)    
fig.tight_layout()

fig.subplots_adjust(bottom=0.1)
line_labels = ["True", "ML"]#, "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
plt.show()
fig.savefig('true_ml_0.png', dpi=200)    

#%%
utrue = np.copy(sst_masked)
ne, nt = utrue.shape[0], utrue.shape[1]

me = 500
oin = sorted(random.sample(range(ne), me)) 
roin = np.int32(np.linspace(0,me-1,me))

aa = np.zeros(not_nan_array.shape[0])
aa[aa == 0] = 'nan'
aa[not_nan_array] = T[:,0]
trec = np.flipud((aa.reshape(len(lat[0,:]),len(lon[0,:]),order='F')))

observations = np.zeros(ne)
observations[oin] = 1

obs_points = np.zeros(not_nan_array.shape[0])
obs_points[obs_points == 0] = 'nan'
obs_points[not_nan_array] = observations
obs_points = np.flipud((obs_points.reshape(len(lat[0,:]),len(lon[0,:]),order='F')))

#%%
fig,axs = plt.subplots(1,1, figsize=(10,8))

current_cmap = plt.cm.get_cmap('coolwarm')
current_cmap.set_bad(color='white',alpha=1.0)

cs = axs.imshow(sst2[0,:,:],cmap='coolwarm')
fig.colorbar(cs, ax=axs, orientation='vertical',shrink=0.5)

for i in range(obs_points.shape[0]):
    for j in range(obs_points.shape[1]):
        if obs_points[i,j] == 1:
            axs.plot(j,i,'kx', ms=4, fillstyle='none')

fig.tight_layout()
plt.show()   
fig.savefig('observation_sampling.png', dpi=200, bbox_inches='tight')

#%%
dh = np.zeros((me,ne))
dh[roin,oin] = 1.0

nf = 10
nb = int(nt/nf) 
oib = [nf*k for k in range(nb+1)]

mu_obs = 0.0
sd2_obs = 1.0
sd1_obs = np.sqrt(sd2_obs)

z = np.zeros((me,nb+1))
uobs = utrue[:,oib] + np.random.normal(mu_obs,sd1_obs,[ne,nb+1])
z = uobs[oin,:]

npe = 20
lambda_ = 1.1
ua = np.zeros((ne,nt))
ue = np.zeros((ne,npe))
Af = np.zeros((ne,npe))   # Af data

#%%
sst_masked_all_fluct = sst_masked_all - sst_average_small
atrue = PODproj(sst_masked_all_fluct, PHIw)
atrue_scaled = (2.0*atrue - (atrain_max + atrain_min))/(atrain_max - atrain_min) 

atest = np.zeros((npe,1,lookback,nr))
aml = np.zeros((npe,nt,nr))
aml_avg = np.zeros((nt,nr))

mu_ic = 0.0
sd2_ic = 0.01
sd1_ic = np.sqrt(sd2_ic)

ic_snapshots = np.random.randint(num_samples, size=npe)

#%%
# create input at t = 0 for the model testing
for ne in range(npe):
    for k in range(lookback):    
        print(ne, ic_snapshots[ne]+k)
        atest[ne,0,k,:] = atrue_scaled[ic_snapshots[ne]+k,:] 
        aml[ne,k,:] = atrue_scaled[ic_snapshots[ne]+k,:] 
    aml_avg[k,:] = np.average(aml[:,k,:], axis=0)
    
#%%
nt = num_samples
testing_time_init = tm.time()

kobs = 1

# predict results recursively using the model
for k in range(lookback,nt):
    print(k)
    for ne in range(npe):
        aml[ne,k,:] = model.predict(atest[ne])
        atest[ne,0,:-1,:] = atest[ne,0,1:,:]
        atest[ne,0,-1,:] = aml[ne,k,:] 
        
        aml_unscaled = 0.5*(aml[ne,k,:]*(atrain_max - atrain_min) + (atrain_max + atrain_min)) 
        tfluc = PODrec(aml_unscaled,PHIw)
        ue[:,ne] =  (tfluc + sst_average_small).flatten()
    
    # compute mean of the forecast fields
    ua[:,k] = np.average(ue,axis=1)         
    
    aml_avg[k,:] = np.average(aml[:,k,:], axis=0)
    
    # print(k, aml[0,k,:])
    
    if k%nf == 0:
        for ne in range(npe):
            aml_unscaled = 0.5*(aml[ne,k,:]*(atrain_max - atrain_min) + (atrain_max + atrain_min)) 
            tfluc = PODrec(aml_unscaled,PHIw)
            ue[:,ne] =  (tfluc + sst_average_small).flatten()
        
        # compute mean of the forecast fields
        uf = np.average(ue[:,:],axis=1)   
        
        # compute Af dat
        Af = ue - uf.reshape(-1,1)
        
        da = dh @ Af
        
        cc = da @ da.T/(npe-1)  
        
        diag = np.arange(me)
        cc[diag,diag] = cc[diag,diag] + sd2_obs 
        
        ci = np.linalg.pinv(cc)
        
        km = Af @ da.T @ ci/(npe-1)
                
        # analysis update    
        kmd = km @ (z[:,kobs] - uf[oin])
        ua[:,k] = uf[:] + kmd[:]
        
        # ensemble correction
        ha = dh @ Af
        
        ue = Af - 0.5*(km @ dh @ Af) + ua[:,k].reshape(-1,1)
        
        #multiplicative inflation: set lambda=1.0 for no inflation
        ue = ua[:,k].reshape(-1,1) + lambda_*(ue - ua[:,k].reshape(-1,1))
        
        for ne in range(npe):
            ufluc = ue[:,ne].reshape(-1,1) - sst_average_small
            at = PODproj(ufluc, PHIw)
            aml[ne,k,:] = (2.0*at - (atrain_max + atrain_min))/(atrain_max - atrain_min)
            atest[ne,-1,:] = aml[ne,k,:] 
        
        aml_avg[k,:] = np.average(aml[:,k,:], axis=0)
            
        kobs = kobs + 1

#%%
total_testing_time = tm.time() - testing_time_init
print('Total testing time=', total_testing_time)
cpu = open("a_cpu.txt", "w+")
cpu.write('testing time in seconds = ')
cpu.write(str(total_testing_time))
cpu.close()

#%%  unscaling
ytest_ml_denkf = 0.5*(aml_avg*(atrain_max - atrain_min) + (atrain_max + atrain_min))

#%%
t = np.linspace(1,num_samples,num_samples)
fig, ax = plt.subplots(nrows=nr,ncols=1,figsize=(12,8),sharex=True)
ax = ax.flat
nrs = at.shape[1]

for i in range(nrs):
    ax[i].plot(t,atrue[:,i],'k',label=r'True Values')
    ax[i].plot(t,ytest_ml_denkf[:,i],'b-',label=r'ML ')
    ax[i].set_xlabel(r'$t$',fontsize=14)
    ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
    ax[-1].set_xlim([t[0],t[-1]])
    ax[i].axvspan(0, t[num_samples_train], alpha=0.2, color='darkorange')

ax[-2].set_xlabel(r'$t$',fontsize=14)    
ax[-1].set_xlabel(r'$t$',fontsize=14)    
fig.tight_layout()

fig.subplots_adjust(bottom=0.1)
line_labels = ["True", "ML"]#, "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
plt.show()
fig.savefig(f'true_ml_{npe}.png', dpi=200)    

#%%
T_fom = sst_masked_all

tfluc = PODrec(atrue,PHIw)
T_true = tfluc + sst_average_small

tfluc = PODrec(ytest_ml,PHIw)
T_ml = tfluc + sst_average_small

tfluc = PODrec(ytest_ml_denkf,PHIw)
T_ml_denkf = tfluc + sst_average_small

#%%
rmse_fom_true = np.linalg.norm(T_true - T_fom, axis=0)
rmse_fom_ml = np.linalg.norm(T_ml - T_fom, axis=0)
rmse_fom_ml_denkf = np.linalg.norm(T_ml_denkf - T_fom, axis=0)

rmse_true_ml = np.linalg.norm(T_ml - T_true, axis=0)
rmse_true_ml_denkf = np.linalg.norm(T_ml_denkf - T_true, axis=0)

fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(12,5),sharex=True)

ax[0].plot(t, rmse_fom_true, label='True')
ax[0].plot(t, rmse_fom_ml, label='ML')
ax[0].plot(t, rmse_fom_ml_denkf, label='ML-DEnKF')

ax[1].plot(t, rmse_true_ml, label='ML')
ax[1].plot(t, rmse_true_ml_denkf, label='ML-DEnKF')

for i in range(2):
    ax[i].legend()
    ax[i].set_xlabel('$t$')
    ax[i].set_ylabel('$||\epsilon||$')
    
plt.show()
fig.savefig('error_l2.png', dpi=200)