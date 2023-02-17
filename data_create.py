#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:29:44 2022

@author: administrateur
"""
import numpy as np
import matplotlib.pyplot as plt

# We read the data
fid = open('./data/uspa_mob15_1.dat','r');
signal = np.fromfile(fid, dtype='>f')
fid.close()

# We normalize the signal
signal=(signal-np.mean(signal))/np.std(signal)

# We define the size N of each realization
N = 2**15
N_samples=  392*2

# We reshape the data -> 392*2 realizations of size N
data=np.reshape(signal,(N_samples,N));
Nreal=data.shape[0]

# The sampling frequency of the signal is 25000 Hz
tau=1/25000 # Sampling period

# We will be interested in the "derivative"
incrs=np.diff(data,axis=1)

# incrs is your dataset. So you have 392 signals of size 2**(16)-1
# You can define you training dataset as containing 314 signals ~80% 
# Your validation dataset containing 40 signals
# Your test dataset containing 38 signals. Actually, you don't need it

plt.figure()
plt.plot(incrs[1,:])
plt.show()

plt.figure()
plt.hist(signal, bins=30)
plt.show()

np.save('./data/data_incrs.npy', incrs)
np.save('./data/data.npy', data)

    




