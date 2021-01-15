# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:29:18 2021

@author: henry
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# %% moving average function

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# %% import data
cwd = os.getcwd()

for i in np.arange(15,16,1):
    with open(cwd+'/data/'+str(i),'rb') as f:
        data = pickle.load(f)
    
    # plot average reward per 10 iterations
    # data_fixer = [sum(data[j:j+10])/10 for j,jj in enumerate(data) if j % 10 == 0]
    
    data_fixer = moving_average(data,500)
    
    plt.figure(i)
    plt.plot(data_fixer[:])
    # plt.xlabel('10th iterations instance')
    # plt.ylabel('average reward over the latest 10 iterations')
    # plt.title('final iteration was ' +str(10*i))




    # min_data = [min(data[j:j+10]) for j,jj in enumerate(data) if j % 10 == 0]
    
    
    # plt.figure(10*i)
    # plt.plot(min_data[:-1])
    # plt.xlabel('10th iterations instance')
    # plt.ylabel('min reward over the latest 10 iterations')
    # plt.title('final iteration was ' +str(100*i))














































