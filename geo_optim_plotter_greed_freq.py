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
# with open(cwd+'/geo_optim_chars/results_all','wb') as f:
#     pk.dump(results,f)

# import defaults
theta_vect = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
big_defaults_obj = []
big_defaults_energy = []
big_defaults_acc = []

big_greed_rhos = []
big_greed_energy = []
big_greed_acc = []

for theta in theta_vect:
    # defaults
    with open(cwd+'/geo_optim_chars/greed/default_'+str(theta)+'_obj','rb') as f:
        big_defaults_obj.append(pk.load(f))

    with open(cwd+'/geo_optim_chars/greed/default_'+str(theta)+'_energy','rb') as f:
        big_defaults_energy.append(pk.load(f))

    with open(cwd+'/geo_optim_chars/greed/default_'+str(theta)+'_acc','rb') as f:
        big_defaults_acc.append(pk.load(f))

    # greed rhos
    with open(cwd+'/geo_optim_chars/greed/freq_max_'+str(theta)+'_obj','rb') as f:
        big_greed_rhos.append(pk.load(f))

    with open(cwd+'/geo_optim_chars/greed/freq_max_'+str(theta)+'_energy','rb') as f:
        big_greed_energy.append(pk.load(f))

    with open(cwd+'/geo_optim_chars/greed/freq_max_'+str(theta)+'_acc','rb') as f:
        big_greed_acc.append(pk.load(f))



# %% determine percentages for all three metrics and for all theta changes
# x is theta; y is bar - percentage diff
# 3 bar clumped per x index


obj_percents = []
energy_percents = []
acc_percents = []

for ind,theta in enumerate(theta_vect):
    # determine obj percent diff first
    obj_temp_per = 100*np.abs(big_defaults_obj[ind][-1] - big_greed_rhos[ind][-1])   \
        /big_defaults_obj[ind][-1]
    obj_percents.append(np.round(obj_temp_per,2))
    
    
    # determine energy percent diff
    nrg_temp_per = 100*np.abs(big_defaults_energy[ind][-1] - big_greed_energy[ind][-1])   \
        /big_defaults_energy[ind][-1]
    energy_percents.append(np.round(nrg_temp_per,2))
    
    
    # determine acc percent diff
    acc_temp_per = 100*np.abs(big_defaults_acc[ind][-1] - big_greed_acc[ind][-1])   \
        /big_defaults_acc[ind][-1]
    acc_percents.append(np.round(acc_temp_per,2))


f, axs = plt.subplots(1,1,figsize=(4,4))
# axs2 = axs.twinx() # Create another axes that shares the same x-axis as ax
width = 0.9
cats= 3
x = np.arange(len(theta_vect))

axs.bar(x - 1 * width/cats, obj_percents, width=width/cats, \
        color='sienna',edgecolor='black',label='object function (%) diff')
axs.bar(x, energy_percents, width=width/cats, \
        color='forestgreen',edgecolor='black',label='energy (%) diff')
axs.bar(x + 1 * width/cats, acc_percents, width=width/cats, \
        color='royalblue',edgecolor='black',label='learning (%) diff')

# axs.set_yscale('log')
# axs2.set_ylim([0,5])
axs.set_axisbelow(True)
axs.grid(True)


axs.set_title('Percent of resource consumption greedy (max freq) vs our method')
axs.legend()
axs.set_ylabel('Percent difference (%)')
axs.set_xlabel(r'1 - $\theta$')

from copy import deepcopy

# theta_vect_mpl = deepcopy(theta_vect)
# theta_vect_mpl.append(0)
# theta_vect_mpl = np.roll(theta_vect_mpl,1)

axs.set_xticks(list(range(len(theta_vect))))
# axs.set_xticklabels(theta_vect_mpl)
axs.set_xticklabels(theta_vect)
# axs2.legend()

import os
cwd = os.getcwd()
# plt.savefig(cwd+'/geo_optim_chars/greed1_percent_diff.pdf',dpi=1000,bbox_inches='tight')




























