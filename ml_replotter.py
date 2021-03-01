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
data_source = 'fmnist'

# %% plot 1 fl vs pfl extreme nonidd comparison
# static ratio - 1,1 at taus1=taus2=2
ratio = 1 #, taus1,taus2 = 1,2,2
## reload data
total_fl_accs, total_pfl_accs = [], []
swarm_period, global_period = 2,2
for iid_style in ['extreme','mild','iid']: #crashed on frankie for some reason...; wtf ,'extreme'
    
    with open(data_loc+'fl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source \
              +'_'+str(swarm_period)+'_'+str(global_period),'rb') as f:
        fl_acc = pk.load(f)
    
    with open(data_loc+'hn_pfl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source \
               +'_'+str(swarm_period)+'_'+str(global_period),'rb') as f:
        pfl_acc = pk.load(f)
    
    total_fl_accs.append(fl_acc)
    total_pfl_accs.append(pfl_acc)


## subplots method with iid, mild, extreme 
plt.figure(1)
f,ax = plt.subplots(1,3,figsize=(10,4),dpi=100,sharey=True)

ind = 0 #[1:20]; [:19]
ax[ind].plot(total_fl_accs[ind],label='fl_acc',marker='x',color='forestgreen',linestyle='dashed')
ax[ind].plot(total_pfl_accs[ind],label='pfl acc',marker='o',color='darkblue',linestyle='dashed')
ax[ind].set_title('extreme non-i.i.d')
# ax[ind].legend()

ind = 1
ax[ind].plot(total_fl_accs[ind],label='fl_acc',marker='x',color='forestgreen',linestyle='dashed')
ax[ind].plot(total_pfl_accs[ind],label='pfl acc',marker='o',color='darkblue',linestyle='dashed')
ax[ind].set_title('moderate non-i.i.d')
# ax[ind].legend()

ind = 2
ax[ind].plot(total_fl_accs[ind],label='fl_acc',marker='x',color='forestgreen',linestyle='dashed')
ax[ind].plot(total_pfl_accs[ind],label='pfl acc',marker='o',color='darkblue',linestyle='dashed')
ax[ind].set_title('i.i.d')
# ax[ind].legend()

for i in range(3):
    if i == 0:
        ax[i].set_ylabel('Accuracy(%)',fontsize=12,**csfont)
    
    ax[i].set_xlabel('Global Aggregation ($k_s^{\mathsf{G}}$)', fontsize=11, **csfont)
    ax[i].set_axisbelow(True)
    ax[i].grid(True)

h,l = ax[0].get_legend_handles_labels()
kw = dict(ncol=2,loc = 'lower center',frameon=False)
#(x, y, width, height)
leg1 = ax[0].legend(h[:],l[:],bbox_to_anchor=(0.8,1.05,1.5,0.2),\
                       mode='expand',fontsize='large',**kw)
    
ax[0].add_artist(leg1)
plt.subplots_adjust(top=0.8,wspace=0.05,hspace=0.15)

# plt.savefig(str(data_source)+'2_2_ratio_1.png',dpi=500)
# %% plot 2 ratio plots with taus1 = 1
# static ratio - 1,1 at taus1=taus2=2

total_fl_ratios = []
total_pfl_ratios = []

for ratio in [1,2]:#,4]:
    # ratio = 1 #, taus1,taus2 = 1,2,2
    ## reload data
    total_fl_accs, total_pfl_accs = [], []
    swarm_period = 1
    global_period = ratio*swarm_period
    for iid_style in ['extreme']:#,'mild']: #,'iid']: #crashed on frankie for some reason...; wtf ,'extreme'
        
        with open(data_loc+'2fl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source \
                  +'_'+str(swarm_period)+'_'+str(global_period),'rb') as f:
            fl_acc = pk.load(f)
        
        with open(data_loc+'hn_pfl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source \
                   +'_'+str(swarm_period)+'_'+str(global_period),'rb') as f:
            pfl_acc = pk.load(f)
        
        total_fl_accs.append(fl_acc)
        total_pfl_accs.append(pfl_acc)
    
    total_fl_ratios.append(total_fl_accs)
    total_pfl_ratios.append(total_pfl_accs)
    
plt.figure(2)
f2,ax2 = plt.subplots(1,2,figsize=(10,4),dpi=100,sharey=True)

ind = 0
for i in range(2): #3
    if i == 0:
        temp_indexes = np.arange(0,120,step=1) #total_fl_ratios[i][0]
    elif i == 1:
        temp_indexes = np.arange(0,120,step=2)
    else:
        temp_indexes = np.arange(0,120,step=4)
    
    ax2[ind].plot(temp_indexes,total_fl_ratios[i][0],label='fl_acc',\
        marker='x',color='forestgreen',linestyle='dashed')
    ax2[ind].plot(temp_indexes,total_pfl_ratios[i][0],label='pfl acc',\
        marker='o',color='darkblue',linestyle='dashed')
    
            
ax2[ind].set_title('extreme non-i.i.d')


# ind = 1
# for i in range(2): #3
#     if i == 0:
#         temp_indexes = np.arange(0,120,step=1) #total_fl_ratios[i][0]
#     elif i == 1:
#         temp_indexes = np.arange(0,120,step=2)
#     else:
#         temp_indexes = np.arange(0,120,step=4)
    
#     ax2[ind].plot(temp_indexes,total_fl_ratios[i][ind],label='fl_acc',\
#         marker='x',color='forestgreen',linestyle='dashed')
#     ax2[ind].plot(temp_indexes,total_pfl_ratios[i][ind],label='pfl acc',\
#         marker='o',color='darkblue',linestyle='dashed')
    
            
# ax2[ind].set_title('moderate non-i.i.d')
# ax[ind].set_title('moderate non-i.i.d')
# ax[ind].legend()

    
    # for i in range(3):
    #     if i == 0:
    #         ax[i].set_ylabel('Accuracy(%)',fontsize=12,**csfont)
        
    #     ax[i].set_xlabel('Global Aggregation ($k_s^{\mathsf{G}}$)', fontsize=11, **csfont)
    #     ax[i].set_axisbelow(True)
    #     ax[i].grid(True)
    
    # h,l = ax[0].get_legend_handles_labels()
    # kw = dict(ncol=2,loc = 'lower center',frameon=False)
    # #(x, y, width, height)
    # leg1 = ax[0].legend(h[:],l[:],bbox_to_anchor=(0.8,1.05,1.5,0.2),\
    #                        mode='expand',fontsize='large',**kw)
        
    # ax[0].add_artist(leg1)
    # plt.subplots_adjust(top=0.8,wspace=0.05,hspace=0.15)


# %% plot 3 ratio plots with taus2 = 1


total_fl_ratios = []
total_pfl_ratios = []

for ratio in [1,2,4]:
    # ratio = 1 #, taus1,taus2 = 1,2,2
    ## reload data
    total_fl_accs, total_pfl_accs = [], []
    global_period = 1
    swarm_period = ratio*global_period
    for iid_style in ['extreme','mild']: #,'iid']: #crashed on frankie for some reason...; wtf ,'extreme'
        
        with open(data_loc+'fl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source \
                  +'_'+str(swarm_period)+'_'+str(global_period),'rb') as f:
            fl_acc = pk.load(f)
        
        with open(data_loc+'hn_pfl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source \
                   +'_'+str(swarm_period)+'_'+str(global_period),'rb') as f:
            pfl_acc = pk.load(f)
        
        total_fl_accs.append(fl_acc)
        total_pfl_accs.append(pfl_acc)
    
    total_fl_ratios.append(total_fl_accs)
    total_pfl_ratios.append(total_pfl_accs)
    
plt.figure(2)
f2,ax2 = plt.subplots(1,2,figsize=(10,4),dpi=100,sharey=True)

ind = 0
for i in range(3):
    if i == 0:
        temp_indexes = np.arange(0,120,step=1) #total_fl_ratios[i][0]
    elif i == 1:
        temp_indexes = np.arange(0,120,step=2)
    else:
        temp_indexes = np.arange(0,120,step=4)
    
    ax2[ind].plot(temp_indexes,total_fl_ratios[i][0],label='fl_acc',\
        marker='x',color='forestgreen',linestyle='dashed')
    ax2[ind].plot(temp_indexes,total_pfl_ratios[i][0],label='pfl acc',\
        marker='o',color='darkblue',linestyle='dashed')
    
            
ax2[ind].set_title('extreme non-i.i.d')


ind = 1
for i in range(3):
    if i == 0:
        temp_indexes = np.arange(0,120,step=1) #total_fl_ratios[i][0]
    elif i == 1:
        temp_indexes = np.arange(0,120,step=2)
    else:
        temp_indexes = np.arange(0,120,step=4)
    
    ax2[ind].plot(temp_indexes,total_fl_ratios[i][ind],label='fl_acc',\
        marker='x',color='forestgreen',linestyle='dashed')
    ax2[ind].plot(temp_indexes,total_pfl_ratios[i][ind],label='pfl acc',\
        marker='o',color='darkblue',linestyle='dashed')
    
            
ax2[ind].set_title('moderate non-i.i.d')
# ax[ind].set_title('moderate non-i.i.d')
# ax[ind].legend()

    