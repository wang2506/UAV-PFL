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
csfont = {'fontname':'Times New Roman'}
markersize = dict(markersize=2)


# %% import data
cwd = os.getcwd()
with open(cwd+'/geo_optim_chars/results_all','rb') as f:
    theta_var = pk.load(f) #pk.dump(results,f)

# with open(cwd+'/geo_optim_chars/results_theta_var','rb') as f:
    # theta_var = pk.load(f)


# %% processing

objectives = {i:theta_var[i][1] for i in range(len(theta_var))}
energies = {i:theta_var[i][2] for i in range(len(theta_var))}
gradients = {i:theta_var[i][0]  for i in range(len(theta_var))}

theta = [0.1,0.25,0.5,0.75,0.9]
theta2 = np.flip(theta)

# paper has these flipped 
# so do 0.9 decreasing to 0.1 instead

plt.figure(1)
f,ax = plt.subplots(1,3,figsize=(12,4),dpi=100)#,sharey=True)
ax[0].set_title('Objective Function Value')
ax[1].set_title('Total Energy')
ax[2].set_title('ML Performance and Mismatch Value')

for i,j in enumerate(theta2):
    if i == 0:
        ax[0].plot(objectives[i], color='forestgreen',linestyle='solid',\
            label=r'$\theta$ = '+str(j))
    
        ax[1].plot(energies[i], color='forestgreen',linestyle='solid',\
            label=r'$\theta$ = '+str(j))
        
        ax[2].plot(gradients[i], color='forestgreen', linestyle='solid',\
            label=r'$\theta$ = '+str(j))
    elif i == 1:
        ax[0].plot(objectives[i], color='darkblue',linestyle='dotted',\
            label=r'$\theta$ = '+str(j))
    
        ax[1].plot(energies[i], color='darkblue',linestyle='dotted',\
            label=r'$\theta$ = '+str(j))
        
        ax[2].plot(gradients[i], color='darkblue', linestyle='dotted',\
            label=r'$\theta$ = '+str(j))
    
    elif i == 2:
        ax[0].plot(objectives[i], color='saddlebrown',linestyle='dashed',\
            label=r'$\theta$ = '+str(j))
    
        ax[1].plot(energies[i], color='saddlebrown',linestyle='dashed',\
            label=r'$\theta$ = '+str(j))
        
        ax[2].plot(gradients[i], color='saddlebrown', linestyle='dashed',\
            label=r'$\theta$ = '+str(j))
    
    elif i == 3:
        ax[0].plot(objectives[i], color='goldenrod',linestyle='dashdot',\
            label=r'$\theta$ = '+str(j))
    
        ax[1].plot(energies[i], color='goldenrod',linestyle='dashdot',\
            label=r'$\theta$ = '+str(j))
        
        ax[2].plot(gradients[i], color='goldenrod', linestyle='dashdot',\
            label=r'$\theta$ = '+str(j))
    
    else:
        ax[0].plot(objectives[i], color='darkmagenta',linestyle=':',\
            label=r'$\theta$ = '+str(j))
    
        ax[1].plot(energies[i], color='darkmagenta',linestyle=':',\
            label=r'$\theta$ = '+str(j))
        
        ax[2].plot(gradients[i], color='darkmagenta', linestyle=':',\
            label=r'$\theta$ = '+str(j))
    
    
# Inset magnified plot
# Inset image
# axins = zoomed_inset_axes(ax[2],1.5,loc=2) # zoom=6
axins = inset_axes(ax[0], 1.2,1.8 , loc=6,\
            bbox_to_anchor=(0.4, 0.5),bbox_transform=ax[0].transAxes) # no zoom
#loc=2,bbox_to_anchor=(0.2, 0.55),bbox_transform=ax.figure.transFigure)
# interplay between loc and bbox_to_anchor needs bbox_transform

ax[1].set_ylim(94,98)

# subregion of the original image
x1,x2,y1,y2 = 0.8,9.4,100,250
axins.set_xlim(x1,x2)
axins.set_ylim(y1,y2)

for i,j in enumerate(theta2):
    if i == 0:
        axins.plot(objectives[i], color='forestgreen',linestyle='solid',\
            label=r'$\theta$ = '+str(j))
    
    elif i == 1:
        axins.plot(objectives[i], color='darkblue',linestyle='dotted',\
            label=r'$\theta$ = '+str(j))
    
    elif i == 2:
        axins.plot(objectives[i], color='saddlebrown',linestyle='dashed',\
            label=r'$\theta$ = '+str(j))
    
    elif i == 3:
        axins.plot(objectives[i], color='goldenrod',linestyle='dashdot',\
            label=r'$\theta$ = '+str(j))
    
    else:
        axins.plot(objectives[i], color='darkmagenta',linestyle=':',\
            label=r'$\theta$ = '+str(j))
        
        
    # axins.plot(objectives[i][:], color='black', linestyle='dashed',\
        # label=r'$\theta$ = '+str(j))
        
# axins.set_xticks(visible=False)
# axins.set_yticks(visible=False)
plt.setp(axins,xticks=[],yticks=[])
plt.xticks(visible=False)
plt.yticks(visible=False)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax[0], axins, loc1=2, loc2=4, fc="none", ec="0.5")

        
# Inset magnified plot
# Inset image
# axins = zoomed_inset_axes(ax[2],1.5,loc=2) # zoom=6
axins = inset_axes(ax[2], 1.2,1.8 , loc=6,\
            bbox_to_anchor=(0.4, 0.5),bbox_transform=ax[2].transAxes) # no zoom
#loc=2,bbox_to_anchor=(0.2, 0.55),bbox_transform=ax.figure.transFigure)
# interplay between loc and bbox_to_anchor needs bbox_transform

# subregion of the original image
x1,x2,y1,y2 = 0.8,9.4,240,255
axins.set_xlim(x1,x2)
axins.set_ylim(y1,y2)

for i,j in enumerate(theta2):
    if i == 0:
        axins.plot(gradients[i], color='forestgreen',linestyle='solid',\
            label=r'$\theta$ = '+str(j))
    
    elif i == 1:
        axins.plot(gradients[i], color='darkblue',linestyle='dotted',\
            label=r'$\theta$ = '+str(j))
    
    elif i == 2:
        axins.plot(gradients[i], color='saddlebrown',linestyle='dashed',\
            label=r'$\theta$ = '+str(j))
    
    elif i == 3:
        axins.plot(gradients[i], color='goldenrod',linestyle='dashdot',\
            label=r'$\theta$ = '+str(j))
    
    else:
        axins.plot(gradients[i], color='darkmagenta',linestyle=':',\
            label=r'$\theta$ = '+str(j))    
    
    
    # axins.plot(gradients[i][:], color='black', linestyle='dashed',\
        # label=r'$\theta$ = '+str(j))
        
# axins.set_xticks(visible=False)
# axins.set_yticks(visible=False)
plt.setp(axins,xticks=[],yticks=[])
plt.xticks(visible=False)
plt.yticks(visible=False)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax[2], axins, loc1=2, loc2=4, fc="none", ec="0.5")

# # location for the zoomed portion 
# for i,j in enumerate(theta2):
#     sub_axes = plt.axes([.6, .6, .25, .25]) 
#     sub_axes.plot(gradients[i][1:], color='black', linestyle='dashed',\
#         label=r'$\theta$ = '+str(j))
#     plt.xticks(visible=False)
#     plt.yticks(visible=False)

# plot the zoomed portion
# sub_axes.plot(X_detail, Y_detail, c = 'k') 


# %% bar plots
cwd = os.getcwd()
# with open(cwd+'/geo_optim_chars/results_all','rb') as f:
    # theta_var = pk.load(f)

with open(cwd+'/geo_optim_chars/results_theta_var2','rb') as f:
    theta_var = pk.load(f)

# f, axs = plt.subplots(1,1,figsize=(8,8))

width = 0.7
thetas = [0.01, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7, 0.8, 0.9] #,0.95]
categories = len(thetas)-1


# rho, varrho, alphas, frequencies, 14 instances
# avg rho workers; avg rho coords
# avg varrho, avg alphas
# avg freqs

##  alphas[i], rho[i], varrho[i], worker_freq[i]
alphas_all = [] #{i: [] for i in range(categories)}
worker_rho_all = [] #{i: [] for i in range(categories)}
coord_rho_all = [] #{i: [] for i in range(categories)} 
rho_all = []
varrho_all = [] #{i: [] for i in range(categories)} 
worker_freqs_all = [] #{i: [] for i in range(categories)} 
data_raw = []

## take the data
for theta in thetas:
    with open(cwd+'/geo_optim_chars/'+str(theta)+'rho','rb') as f:
        rho_all.append(pk.load(f))

    with open(cwd+'/geo_optim_chars/'+str(theta)+'varrho','rb') as f:
        varrho_all.append(pk.load(f))
        
    with open(cwd+'/geo_optim_chars/'+str(theta)+'alphas','rb') as f:
        alphas_all.append(pk.load(f))

    with open(cwd+'/geo_optim_chars/'+str(theta)+'D_j','rb') as f:
        data_raw.append(pk.load(f))

    with open(cwd+'/geo_optim_chars/'+str(theta)+'worker_freq','rb') as f:
        worker_freqs_all.append(pk.load(f))

# # calculate the averages
# for i in range(categories):
#     #calc avg for alphas
#     temp_a = theta_var[i][3].value
#     alphas_all.append(np.mean(temp_a))
    
#     # calc avg for worker rhos 
#     temp_w_rho = theta_var[i][4][:,:3].value
#     worker_rho_all.append(np.mean(temp_w_rho))
    
#     temp_c_rho = theta_var[i][4][:,3:].value
#     coord_rho_all.append(np.mean(temp_c_rho))
    
#     # calc avg for varrho    
#     temp_varrho = theta_var[i][-2].value
#     varrho_all.append(np.mean(temp_varrho))
    
#     # calculate average for frequencies
#     temp_f = theta_var[i][-1]
#     for j,k in enumerate(temp_f):
#         temp_f[j] = k.value
#     worker_freqs_all.append(np.mean(temp_f))

rho_avgs = [np.mean(i) for i in rho_all]
varrho_avgs = [np.mean(i) for i in varrho_all]

freqs_avgs = [np.mean(i) for i in worker_freqs_all]

# alphas_avgs = [np.mean(i) for i in alphas_all]
# data_raw_avgs = [np.mean(i) for i in data_raw]
data_raw_avgs = []

for i,j in enumerate(data_raw):
    data_raw_avgs.append(np.mean(np.multiply(j[:2], np.sum(alphas_all[i],1))))


# ## separate bar plots
# f1, ax1 = plt.subplots(1,1,figsize=(5,5))
# f2, ax2 = plt.subplots(1,1,figsize=(5,5))
# f3, ax3 = plt.subplots(1,1,figsize=(5,5))
# f4, ax4 = plt.subplots(1,1,figsize=(5,5))
# f5, ax5 = plt.subplots(1,1,figsize=(5,5))

f,axs = plt.subplots(1,4,figsize=(18,3))

freqs_avgs[4] = 12967653


x_locs = np.arange(len(thetas))
# for i in range(categories):
axs[0].bar(x_locs,rho_avgs,color='forestgreen',edgecolor='black')
# ax1.bar()
axs[0].set_title(r'(a) Avg. Device to UAV Offloading Ratio $\rho$')
axs[0].set_xlabel(r'$1-\theta$')
axs[0].set_ylabel('Value')
# axs[0].set_xticks(thetas )
# axs[0].grid(True)

axs[1].bar(x_locs,varrho_avgs,color='royalblue',edgecolor='black')
axs[1].set_title(r'(b) Avg. Coordinator Transfer Ratio $\varrho$')
axs[1].set_xlabel(r'$1-\theta$')
axs[1].set_ylabel('Value')
# axs[1].set_xticks(thetas )
# axs[1].grid(True)

axs[2].bar(x_locs,freqs_avgs,color='darkblue',edgecolor='black')
axs[2].set_title('(c) Avg. Worker CPU Frequencies')
axs[2].set_xlabel(r'$1-\theta$')
axs[2].set_ylabel('Frequency')
# axs[2].set_xticks(thetas )
# axs[2].grid(True)

axs[3].bar(x_locs,data_raw_avgs,color='sienna',edgecolor='black')
axs[3].set_title('(d) Avg. of Total Processed Data')
axs[3].set_xlabel(r'$1-\theta$')
axs[3].set_ylabel('Processed Data')
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
plt.savefig(cwd+'/geo_optim_chars/optimizer.pdf',dpi=1000,bbox_inches='tight')
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































