# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 13:36:36 2022

@author: ch5b2
"""
import numpy as np
import pickle as pk
import os


vary_bat = True
# vary_bat = False

ep_greed = 0.7
g_discount = 0.7

if vary_bat == True:
    bat_vec = ['medium','high','vhigh'] #'low',
    # bat_vec2 = ['low','medium','high']
else:
    bat_vec = ['medium']

seeds = [1]#,3,4]
cwd = os.getcwd()

all_brt_freqs = {}
for ind_b,brt in enumerate(bat_vec):
    for ind_s,seed in enumerate(seeds):
        with open(cwd+'/drl_results/RNN/seed_'+str(seed)+'_'\
                  +str(ep_greed)+'_'+'visit_freq_large'+\
                  '_'+str(g_discount)\
                +'_tanh_mse'+\
                '_'+brt,'rb') as f:
            freq_visits = pk.load(f)
            
        if ind_s == 0: #3 swarms, 1000 epochs
            brt_freqs = sum((freq_visits[29000]-freq_visits[28000])[-2:])/1000/len(seeds)/3
        else:
            brt_freqs += sum((freq_visits[29000]-freq_visits[28000])[-2:])/1000/len(seeds)/3
    
    all_brt_freqs[brt] = brt_freqs