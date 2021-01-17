# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:29:18 2021

@author: henry
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from statistics import median

# %% moving average function

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# %% import data
cwd = os.getcwd()

# for i in np.arange(15,16,1):
with open(cwd+'/data/'+str(0)+'_30_ep_CNN','rb') as f:
    data = pickle.load(f)

# plot average reward per 10 iterations
# data_fixer = [sum(data[j:j+1000])/1000 for j,jj in enumerate(data) if j % 1000 == 0]

data_fixer = moving_average(data,100)

plt.figure()
plt.plot(data_fixer[:])
plt.title('moving avg 100')
    # plt.xlabel('10th iterations instance')
    # plt.ylabel('average reward over the latest 10 iterations')
    # plt.title('final iteration was ' +str(10*i))




min_data = [min(data[j:j+1000]) for j,jj in enumerate(data) if j % 1000 == 0]


plt.figure(2)
plt.plot(min_data[:-1])
plt.title('min')


median_data = [median(data[j:j+1000]) for j,jj in enumerate(data) if j%1000 == 0]

plt.figure(3)
plt.plot(median_data)
plt.title('median')








































