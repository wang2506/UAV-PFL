# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:25:29 2021

@author: henry
"""
## gop_greed_both - geo_optim_plotter_greed_both
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

big_greed_rhos2 = []
big_greed_energy2 = []
big_greed_acc2 = []

for theta in theta_vect:
    # defaults
    with open(cwd+'/geo_optim_chars/greed/tau_adjust_default_'+str(theta)+'_obj','rb') as f:
        big_defaults_obj.append(pk.load(f))

    with open(cwd+'/geo_optim_chars/greed/tau_adjust_default_'+str(theta)+'_energy','rb') as f:
        big_defaults_energy.append(pk.load(f))

    with open(cwd+'/geo_optim_chars/greed/tau_adjust_default_'+str(theta)+'_acc','rb') as f:
        big_defaults_acc.append(pk.load(f))

    # greed rhos
    # tau_adjusted_
    with open(cwd+'/geo_optim_chars/greed/tau_adjusted_rho_max_'+str(theta)+'_obj','rb') as f:
        big_greed_rhos.append(pk.load(f))

    with open(cwd+'/geo_optim_chars/greed/tau_adjusted_rho_max_'+str(theta)+'_energy','rb') as f:
        big_greed_energy.append(pk.load(f))

    with open(cwd+'/geo_optim_chars/greed/tau_adjusted_rho_max_'+str(theta)+'_acc','rb') as f:
        big_greed_acc.append(pk.load(f))

    # greed alphas
    with open(cwd+'/geo_optim_chars/greed/tau_adjusted_alphas_'+str(theta)+'_obj','rb') as f:
        big_greed_rhos2.append(pk.load(f))

    with open(cwd+'/geo_optim_chars/greed/tau_adjusted_alphas_'+str(theta)+'_energy','rb') as f:
        big_greed_energy2.append(pk.load(f))

    with open(cwd+'/geo_optim_chars/greed/tau_adjusted_alphas_'+str(theta)+'_acc','rb') as f:
        big_greed_acc2.append(pk.load(f))



# %% determine percentages for all three metrics and for all theta changes
# x is theta; y is bar - percentage diff
# 3 bar clumped per x index

obj_percents = []
energy_percents = []
acc_percents = []

obj_percents2 = []
energy_percents2 = []
acc_percents2 = []

for ind,theta in enumerate(theta_vect):
    # determine obj percent diff first
    obj_temp_per = 100*np.abs(big_defaults_obj[ind][-1] - big_greed_rhos[ind][-1])   \
        /big_greed_rhos[ind][-1] #big_defaults_obj[ind][-1]
    obj_percents.append(np.round(obj_temp_per,2))
    
    obj_temp_per2 = 100*np.abs(big_defaults_obj[ind][-1] - big_greed_rhos2[ind][-1])   \
        /big_greed_rhos2[ind][-1] #big_defaults_obj[ind][-1]
    obj_percents2.append(np.round(obj_temp_per2,2))    
    
    
    # determine energy percent diff
    nrg_temp_per = 100*np.abs(big_defaults_energy[ind][-1] - big_greed_energy[ind][-1])   \
        / big_greed_energy[ind][-1] #big_defaults_energy[ind][-1]
    energy_percents.append(np.round(nrg_temp_per,2))
    
    nrg_temp_per2 = 100*np.abs(big_defaults_energy[ind][-1] - big_greed_energy2[ind][-1])   \
        / big_greed_energy2[ind][-1] #big_defaults_energy[ind][-1]
    energy_percents2.append(np.round(nrg_temp_per2,2))


f, axs = plt.subplots(1,1,figsize=(3.5,1.7))
# axs2 = axs.twinx() # Create another axes that shares the same x-axis as ax
width = 0.82
cats= 4
x = np.arange(len(theta_vect))
 #Function ; Consumption
axs.bar(x - 1.5* width/cats, obj_percents, width=width/cats, \
        color='darkblue',edgecolor='black',label='Obj. Func. (Ours vs. G.O.)')#'Objective Greedy Offloading')
axs.bar(x - 0.5* width/cats, obj_percents2, width=width/cats, \
        color='cornflowerblue',edgecolor='black',label='Obj. Func. (Ours vs. M.P.)')#'Objective Greedy Processing')
    
axs.bar(x+0.5 * width/cats, energy_percents, width=width/cats, \
        color='darkgreen',edgecolor='black',label='Energy (Ours vs. G.O.)')#'Energy Greedy Offloading')
axs.bar(x+ 1.5* width/cats, energy_percents2, width=width/cats, \
        color='limegreen',edgecolor='black',label='Energy (Ours vs. M.P.)')#'Energy Greedy Processing')


# axs.set_yscale('log')
# axs2.set_ylim([0,5])
axs.set_axisbelow(True)
axs.grid(True)


# axs.set_title('Percent of resource consumption greedy (max freq) vs our method')
axs.legend(fontsize=8)
axs.set_ylabel('Percentage Decrease (%)',fontsize=9)
axs.set_xlabel(r'ML Performance Weight (1 - $\theta$)',fontsize=9)
# axs.set_xlabel(r'Learning Objective Weight ($\theta$)',fontsize=10) #for chris proposal

from copy import deepcopy

# theta_vect_mpl = deepcopy(theta_vect)
# theta_vect_mpl.append(0)
# theta_vect_mpl = np.roll(theta_vect_mpl,1)

axs.set_xticks(list(range(len(theta_vect))))
# axs.set_xticklabels(theta_vect_mpl)
axs.set_xticklabels(theta_vect)
# axs2.legend()

axs.tick_params(axis='both', which='major', labelsize=9)


h,l = axs.get_legend_handles_labels()
kw = dict(ncol=2,loc = 'lower center',frameon=False)
kw2 = dict(ncol=2,loc = 'lower center',frameon=False)
#(x, y, width, height) #h[0:2:1],l[0:2:1]; h[2::1],l[2::1],
leg1 = axs.legend(h[1::2],l[1::2],bbox_to_anchor=(-0.2,1.01,1.3,0.2),\
                mode='expand',fontsize=9,**kw)
leg2 = axs.legend(h[0::2],l[0::2],bbox_to_anchor=(-0.2,1.13,1.3,0.2),\
                        mode='expand',fontsize=9,**kw)
axs.add_artist(leg1)

import os
cwd = os.getcwd()
plt.savefig(cwd+'/geo_optim_chars/greed1_percent_diff_test.pdf',dpi=1000,bbox_inches='tight')
# plt.savefig(cwd+'/geo_optim_chars/greed1_percent_diff.pdf',dpi=1000,bbox_inches='tight')
