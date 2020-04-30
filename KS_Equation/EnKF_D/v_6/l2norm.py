#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:05:47 2020

@author: suraj
"""

import numpy as np

data = np.load('l2norm.npz')

l16 = data['l2_16'] 
l32 = data['l2_32'] 
l64 = data['l2_64'] 

l2norm = np.array([l16,l32,l64])
l2norm = np.reshape(l2norm,[-1,1])

np.savetxt('l2norm.csv',l2norm)