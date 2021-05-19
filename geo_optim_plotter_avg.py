# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:25:29 2021

@author: henry
"""
import pickle as pk 
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

mpl.style.use('default')
mpl.rc('font',family='Times New Roman')
# mpl.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

csfont = {'fontname':'Times New Roman'}
markersize = dict(markersize=2)

# %% bar plots
cwd = os.getcwd()

with open(cwd+'/geo_optim_chars/results_theta_var2','rb') as f:
    theta_var = pk.load(f)

width = 0.7
thetas = [0.01, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7, 0.8, 0.9] #,0.95]
categories = len(thetas)-1

master_seeds = [1,3,5,6,7,8,10,13,16,17]

# rho, varrho, alphas, frequencies, 14 instances
# avg rho workers; avg rho coords
# avg varrho, avg alphas
# avg freqs

##  alphas[i], rho[i], varrho[i], worker_freq[i]
# alphas_all = [] #{i: [] for i in range(categories)}
# worker_rho_all = [] #{i: [] for i in range(categories)}
# coord_rho_all = [] #{i: [] for i in range(categories)} 
# rho_all = []
# varrho_all = [] #{i: [] for i in range(categories)} 
# worker_freqs_all = [] #{i: [] for i in range(categories)} 
# data_raw = []

## dictionaries with numpy arrays embedded within
alphas_all = {i:[] for i in master_seeds}
worker_rho_all = {i:[] for i in master_seeds}
coord_rho_all = {i:[] for i in master_seeds}
rho_all = {i:[] for i in master_seeds}
varrho_all = {i:[] for i in master_seeds}
worker_freqs_all = {i:[] for i in master_seeds}
data_raw = {i:[] for i in master_seeds}


## take the data
subfolder = 'avg' #'greed'
for theta in thetas:
    for seed in master_seeds:
        
        with open(cwd+'/geo_optim_chars/avg/seed_'+str(seed)+'_'+str(theta)+'rho','rb') as f:
            rho_all[seed].append(pk.load(f))
    
        with open(cwd+'/geo_optim_chars/avg/seed_'+str(seed)+'_'+str(theta)+'varrho','rb') as f:
            varrho_all[seed].append(pk.load(f))
            
        with open(cwd+'/geo_optim_chars/avg/seed_'+str(seed)+'_'+str(theta)+'alphas','rb') as f:
            alphas_all[seed].append(pk.load(f))
    
        with open(cwd+'/geo_optim_chars/avg/seed_'+str(seed)+'_'+str(theta)+'D_j','rb') as f:
            data_raw[seed].append(pk.load(f))
    
        with open(cwd+'/geo_optim_chars/avg/seed_'+str(seed)+'_'+str(theta)+'worker_freq','rb') as f:
            worker_freqs_all[seed].append(pk.load(f))
        
        # print(worker_freqs_all)
        # input('yadda')

# %% take avgs
rho_avgs = {i:[np.mean(j) for j in rho_all[i]] for i in master_seeds}
varrho_avgs = {i:[np.mean(j) for j in varrho_all[i]] for i in master_seeds}
freqs_avgs = {i:[np.mean(j) for j in worker_freqs_all[i]] for i in master_seeds}

data_raw_avgs = {i:[] for i in master_seeds}
# data_raw_avgs = []
for seed in master_seeds:
    for i,j in enumerate(data_raw[seed]):
        data_raw_avgs[seed].append(np.mean(np.multiply(j[:2],\
                np.sum(alphas_all[seed][i],1))))

# for i,j in enumerate(data_raw): #[j:2] cause only the workers
    # data_raw_avgs.append(np.mean(np.multiply(j[:2], np.sum(alphas_all[i],1))))

# %% avg of each seed results at each theta instance
# for theta in thetas:
rho_avgs2 = np.zeros(len(rho_avgs[1]))
varrho_avgs2 = np.zeros(len(varrho_avgs[1]))
freqs_avgs2 = np.zeros(len(freqs_avgs[1]))
data_raw_avgs2 = np.zeros(len(data_raw_avgs[1]))

for seed in master_seeds:
    rho_avgs2 += np.array(rho_avgs[seed])/len(master_seeds)
    varrho_avgs2 += np.array(varrho_avgs[seed])/len(master_seeds)
    freqs_avgs2 += np.array(freqs_avgs[seed])/len(master_seeds)
    data_raw_avgs2 += np.array(data_raw_avgs[seed])/len(master_seeds)


# %%
f,axs = plt.subplots(1,4,figsize=(18,3))

# freqs_avgs[4] = 12967653


x_locs = np.arange(len(thetas))
# for i in range(categories):
axs[0].bar(x_locs,rho_avgs2,color='forestgreen',edgecolor='black')
# ax1.bar()
axs[0].set_title(r'(a)',y=-0.3)
axs[0].set_xlabel(r'ML Performance Weight ($1-\theta$)')
axs[0].set_ylabel(r'Avg. Device to UAV Offloading Ratio ($\mathbf{\rho}$)')
# axs[0].set_xticks(thetas )
# axs[0].grid(True)

axs[1].bar(x_locs,varrho_avgs2,color='royalblue',edgecolor='black')
axs[1].set_title(r'(b)',y=-0.3)
axs[1].set_xlabel(r'ML Performance Weight ($1-\theta$)')
axs[1].set_ylabel(r' Avg. Coordinator Transfer Ratio ($\mathbf{\varrho}$)')
# axs[1].set_xticks(thetas )
# axs[1].grid(True)

axs[2].bar(x_locs,freqs_avgs2,color='darkblue',edgecolor='black')
axs[2].set_title('(c)',y=-0.3) #' Avg. Worker CPU Frequencies',y=-0.3)
axs[2].set_xlabel(r'ML Performance Weight ($1-\theta$)')
axs[2].set_ylabel(r'Avg. Worker CPU Frequency ($\mathbf{g}$)')
# axs[2].set_xticks(thetas )
# axs[2].grid(True)

axs[3].bar(x_locs,data_raw_avgs2,color='sienna',edgecolor='black')
axs[3].set_title('(d)',y=-0.3) #' Avg. of Total Processed Data',y=-0.3)
axs[3].set_xlabel(r'ML Performance Weight ($1-\theta$)')
axs[3].set_ylabel(r'Avg. Processed Data')
# axs[3].set_xticks(thetas )
# axs[3].grid(True)

loc = mpl.ticker.MultipleLocator(base=1.0)
# axs[0].xaxis.set_major_locator(loc)
# axs[1].xaxis.set_major_locator(loc)
thetas = [0] + thetas

for i in range(4):
    axs[i].set_axisbelow(True)
    axs[i].grid(True)
    axs[i].set_xticklabels(thetas)
    axs[i].xaxis.set_major_locator(loc)
    
#     ax_ticks = [0, '-20%','-15%','-10%','-5%','Max']
# axs[0].set_xticklabels(ax_ticks)
# axs[1].set_xticklabels(ax_ticks) 
# 14 clusters of bars
plt.savefig(cwd+'/geo_optim_chars/optimizer_avgs.pdf',dpi=1000,bbox_inches='tight')
# axs.bar(x_locs)

# rects1 = axs[0].bar(x_locs - 2.5*width/categories,branch_track1,width/categories,\
#                 label='smart w/ offload',color='forestgreen',edgecolor='black')
# rects2 = axs[0].bar(x_locs - 1.5*width/categories,random_track1,width/categories,\
#                 label='random w/ offload',color='darkblue',edgecolor='black')
# rects3 = axs[0].bar(x_locs - 0.5*width/categories,greedy_track1,width/categories,\
#                 label='heuristic w/ offload',color='sienna',edgecolor='black')
    
# rects4 = axs[0].bar(x_locs + 0.5*width/categories,branch_no_track1,width/categories,\
#                 label='smart sampling',color='limegreen',edgecolor='black')
# rects5 = axs[0].bar(x_locs + 1.5*width/categories,random_no_track1,width/categories,\
#                 label='random sampling',color='royalblue',edgecolor='black')
# rects6 = axs[0].bar(x_locs + 2.5*width/categories,greedy_no_track1,width/categories,\
#                 label='heuristic sampling',color='sandybrown',edgecolor='black')

# ## adding text for labels and title
# axs[0].set_ylabel('Cumulative Processed Data')

# axs[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
# axs.set_yticklabels(list(range(0,81,10)))
# axs.yaxis.set_offset_position('left')

# # axs.set_xticks([0,30,30,35,35,40,40,45,45,50,50])
# for i in range(2):
#     axs[i].set_xlabel('Model Accuracy(%)')    
    
#     axs[i].set_axisbelow(True)
#     axs[i].grid(True)
    
# loc = mpl.ticker.MultipleLocator(base=1.0)
# axs[0].xaxis.set_major_locator(loc)
# axs[1].xaxis.set_major_locator(loc)































