# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:01:17 2021

@author: henry
"""
import os 
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import string

# proof of concept replotter

mpl.style.use('default')
mpl.rc('font',family='Times New Roman')
csfont = {'fontname':'Times New Roman'}
markersize = dict(markersize=2)

cwd = os.getcwd()

save_loc = cwd+'/ml_plots/'
data_loc = cwd+'/data/'

data_source = 'mnist'
# data_source = 'fmnist'

# %% plot 1 fl vs pfl extreme nonidd comparison
# static ratio - 1,1 at taus1=taus2=2
ratio = 1 #, taus1,taus2 = 1,2,2
## reload data
total_fl_accs, total_pfl_accs = [], []
for iid_style in ['extreme','mild']: #'iid' crashed on frankie for some reason...; wtf ,'extreme'
    
    with open(data_loc+'fl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source,'rb') as f:
        fl_acc = pk.load(f)
    
    with open(data_loc+'hn_pfl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source,'rb') as f:
        pfl_acc = pk.load(f)
    
    total_fl_accs.append(fl_acc)
    total_pfl_accs.append(pfl_acc)


## subplots method with iid, mild, extreme 
plt.figure(1)
f,ax = plt.subplots(1,3,figsize=(10,6))

ind = 0
ax[ind].plot(total_fl_accs[ind],label='fl_acc',marker='x',color='forestgreen',linestyle='dashed')
ax[ind].plot(total_pfl_accs[ind],label='pfl acc',marker='o',color='darkblue',linestyle='dashed')

ind = 1
ax[ind].plot(total_fl_accs[ind],label='fl_acc',marker='x',color='forestgreen',linestyle='dashed')
ax[ind].plot(total_pfl_accs[ind],label='pfl acc',marker='o',color='darkblue',linestyle='dashed')



# %% plot 2 ratio plots with taus1 = 1


# %% plot 3 ratio plots with taus2 = 1

