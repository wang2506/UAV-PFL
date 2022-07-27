# -*- coding: utf-8 -*-
"""
@author: henry
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from statistics import median,mean
import matplotlib.patches as patches

mpl.style.use('default')
mpl.rc('font',family='Times New Roman')
# mpl.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

csfont = {'fontname':'Times New Roman'}
 
# %% moving average function

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# %% import data
cwd = os.getcwd()
lwd = 2
# for i in np.arange(15,16,1):
# with open(cwd+'/data/'+str(0)+'_30_ep_CNN','rb') as f:
#     data = pickle.load(f)
ep_start = 0.6
# seed = 2
seed = 1
# seed_vec = [1,3,4]
# seed_vec = [1]
seed_vec = [4]
# gamma = 0.8 #gamma = 0.7 by default
plt.figure(1)

# f1,ax1 = plt.subplots(1,2,figsize=(10,4))#(9.6,4)) #10,4
f1,ax1 = plt.subplots(1,3,figsize=(9,3.3)) #(9,4)

colors = ['darkblue','darkgreen','purple']
bat_state = 'medium'

# vary_ep = True
vary_ep = False

# vary_g = True
vary_g = False

vary_bat = True
# vary_bat = False

if vary_ep == True:
    ep_vec = [0.6,0.7,0.8]
else:
    ep_vec = [0.7]

if vary_g == True:
    g_vec = [0.6,0.7,0.8]
else:
    g_vec = [0.7]

if vary_bat == True:
    # bat_vec = ['medium','vhigh','vhigh3']#'high','vhigh'] #'low',
    # bat_vec = ['medium','vhigh','vhigh2']
    # bat_vec = ['medium','high','vhigh']
    # bat_vec = ['low','high','vhigh3']
    # bat_vec = ['low','medium','high']
    bat_vec = ['low','medium','hlow']
    bat_vec2 = ['low','medium','high']
else:
    bat_vec = ['medium']

# vary_cap = True
# #vary_cap = False

# if vary_cap == True:
#     cap_vec = ['low','medium','high']
# else:
#     cap_vec = ['low']

# # vary_pen = True
# vary_pen = False

# if vary_pen == True:
#     pen_vec = ['low','medium','high']
# else:
#     pen_vec = ['high']

mv_window = 1000 #1000
tests = True
# tests = False

for ind_ep,ep_start in enumerate(ep_vec):#[0.7]):
    for ind_g,gamma in enumerate(g_vec):#[0.7]):
        for ind_bat,bat_state in enumerate(bat_vec):
        # ind_bat = 0 
        # for ind_cap,cap in enumerate(cap_vec):
        # for ind_pen,pen in enumerate(pen_vec):
            for nn_style in ['RNN']:#['MLP','RNN']:
            
                if nn_style == 'MLP':
                    with open(cwd+'/data/new10_'+str(ep_start)+'_rewardtest_large_'+str(gamma)\
                        +'_dynamic','rb') as f:
                        data = pickle.load(f)
                    data_fixer = moving_average(data,1000)
                    ax1[0].plot(data_fixer,label='MLP' \
                        ,linestyle='dotted', \
                            color = 'magenta',linewidth=lwd)                    
                elif nn_style == 'RNN':
                    datas = 0
                    for seed in seed_vec:
                        with open(cwd+'/drl_results/RNN/seed_'+str(seed)+'_'+str(ep_start)+'_'+'reward'\
                                +'test_large'+'_'+str(gamma)+'_tanh_mse'\
                                +'_'+bat_state, \
                                'rb') as f:
                            data = pickle.load(f)
                        # # with open(cwd+'/drl_results/RNN/'+'cap_'+cap\
                        # with open(cwd+'/drl_results/RNN/'+'pen_'+pen\
                        #           +'/seed_'+str(seed)+'_'+str(ep_start)+'_'+'reward'\
                        #         +'test_large'+'_'+str(gamma)+'_tanh_mse'\
                        #         +'_debug2',\
                        #             # +'_'+bat_state, \
                        #         'rb') as f:
                        #     data = pickle.load(f)
                        datas += np.array(data)/len(seed_vec)
                    data_fixer = moving_average(datas,mv_window)
                    
                    if vary_ep == True:
                        ax1[0].plot(data_fixer,label='Ours '+r'$\epsilon=$'+str(ep_start) \
                            ,linestyle='solid', \
                                color = colors[ind_ep],linewidth=lwd)
                    elif vary_g == True:
                        ax1[0].plot(data_fixer,label='Ours '+r'$\gamma=$'+str(gamma) \
                            ,linestyle='solid', \
                                color = colors[ind_g],linewidth=lwd)
                    elif vary_bat == True:
                        ax1[0].plot(data_fixer,label='RT '+bat_vec2[ind_bat].capitalize() \
                            ,linestyle='solid', \
                                color = colors[ind_bat],linewidth=lwd)
                    # elif vary_cap == True:
                    #     ax1[0].plot(data_fixer,label='Capacity '+cap.capitalize() \
                    #         ,linestyle='solid', \
                    #             color = colors[ind_cap],linewidth=lwd)                        
                    # elif vary_pen == False: #True:
                    #     ax1[0].plot(data_fixer,label='Penalty '+pen.capitalize() \
                    #         ,linestyle='solid', \
                    #             color = colors[ind_pen],linewidth=lwd)

ax1[0].set_title(r'(a)',fontsize=20,y=-0.35) # Reward Over Time
ax1[0].grid(True)
ax1[0].set_xlabel('Epoch',fontsize=20)
ax1[0].set_ylabel('Reward',fontsize=20)

# plots for battery
for ind_ep,ep_start in enumerate(ep_vec):#[0.7]):
    for ind_g,gamma in enumerate(g_vec):#[0.7]):
        for ind_bat,bat_state in enumerate(bat_vec):
        # ind_bat = 0 
        # for ind_cap,cap in enumerate(cap_vec):
        # for ind_pen,pen in enumerate(pen_vec):
            for nn_style in ['RNN']:#['MLP','RNN']:
                if nn_style == 'MLP':
                    with open(cwd+'/data/new10_'+str(ep_start)+'_batterytest_large_'+str(gamma)\
                        +'_dynamic','rb') as f:
                        data_b = pickle.load(f)
                    data_b2 = [mean(i) for i in data_b]
                    data_b2 = moving_average(data_b2,1000)
                    ax1[1].plot(data_b2,label='MLP' \
                        ,linestyle='dotted', \
                            color = 'magenta',linewidth=lwd)                    
                elif nn_style == 'RNN':
                    datas_b = 0
                    for seed in seed_vec:
                        with open(cwd+'/drl_results/RNN/seed_'+str(seed)+'_'+str(ep_start)\
                            +'_batterytest_large_'+str(gamma)+'_tanh_mse'\
                                +'_'+bat_state, \
                                'rb') as f:
                            data_b = pickle.load(f)
                        # # with open(cwd+'/drl_results/RNN/'+'cap_'+cap\
                        # with open(cwd+'/drl_results/RNN/'+'pen_'+pen\
                        #           +'/seed_'+str(seed)+'_'+str(ep_start)+'_batterytest_large_'\
                        #         +str(gamma)+'_tanh_mse'\
                        #         +'_debug2',\
                        #             # +'_'+bat_state, \
                        #         'rb') as f:
                        #     data_b = pickle.load(f)                              
                        datas_b += np.array(data_b)/len(seed_vec)
                    data_b2 = [mean(i) for i in datas_b]
                    data_b2 = moving_average(data_b2,mv_window)

                    if vary_ep == True:
                        ax1[1].plot(data_b2,label='Ours '+r'$\epsilon=$'+str(ep_start) \
                            ,linestyle='solid', \
                                color = colors[ind_ep],linewidth=lwd)
                    elif vary_g == True:
                        ax1[1].plot(data_b2,label='Ours '+r'$\gamma=$'+str(gamma) \
                            ,linestyle='solid', \
                                color = colors[ind_g],linewidth=lwd)
                    elif vary_bat == True:
                        ax1[1].plot(data_b2,label='Ours Recharge '+bat_vec2[ind_bat].capitalize() \
                            ,linestyle='solid', \
                                color = colors[ind_bat],linewidth=lwd)
                    # elif vary_cap == True:
                    #     ax1[1].plot(data_b2,label='Capacity '+cap.capitalize() \
                    #         ,linestyle='solid', \
                    #             color = colors[ind_cap],linewidth=lwd)       
                    # elif vary_pen == False: #True:
                    #     ax1[1].plot(data_b2,label='Penalty '+pen.capitalize()  \
                    #         ,linestyle='solid', \
                    #             color = colors[ind_pen],linewidth=lwd)

ax1[1].set_title('(b)',fontsize=20,y=-0.35) # Battery Over Time',fontsize=15,y=-0.24)
ax1[1].grid(True)
ax1[1].set_xlabel('Epoch',fontsize=20)
if tests == True:
    ax1[1].set_ylabel('Avg Battery Level ',fontsize=20)
else:
    ax1[1].set_ylabel('Avg Battery Level (kJ)',fontsize=20)

for ind_ep,ep_start in enumerate(ep_vec):#[0.7]):
    for ind_g,gamma in enumerate(g_vec):#[0.7]):
        for ind_bat,bat_state in enumerate(bat_vec):
        # ind_bat = 0 
        # for ind_cap,cap in enumerate(cap_vec):
        # for ind_pen,pen in enumerate(pen_vec):
            for nn_style in ['RNN']:#['MLP','RNN']:
                if nn_style == 'MLP':
                    with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(ep_start)\
                    +'_ml_reward_onlytest_large_'+str(gamma), \
                    'rb') as f:
                        data_ml = pickle.load(f)
                    data_ml2 = moving_average(data_ml,1000)
                    ax1[2].plot(data_ml2,label='MLP' \
                        ,linestyle='dotted', \
                            color = 'magenta',linewidth=lwd)                    
                elif nn_style == 'RNN':
                    datas_ml = 0
                    for seed in seed_vec:
                        with open(cwd+'/drl_results/RNN/seed_'+str(seed)+'_'+str(ep_start)\
                            +'_ml_reward_onlytest_large_'+str(gamma)+'_tanh_mse'\
                                +'_'+bat_state, \
                                'rb') as f:
                            data_ml = pickle.load(f)  
                        # # with open(cwd+'/drl_results/RNN/'+'cap_'+cap\
                        # with open(cwd+'/drl_results/RNN/'+'pen_'+pen\
                        #           +'/seed_'+str(seed)+'_'+str(ep_start)+'_ml_reward_onlytest_large_'\
                        #         +str(gamma)+'_tanh_mse'\
                        #         +'_debug2',\
                        #             # +'_'+bat_state, \
                        #         'rb') as f:
                        #     data_ml = pickle.load(f)                            
                        datas_ml += np.array(data_ml)/len(seed_vec)
                    data_ml2 = moving_average(datas_ml,mv_window)

                    if vary_ep == True:
                        ax1[2].plot(data_ml2,label='Ours '+r'$\epsilon=$'+str(ep_start) \
                            ,linestyle='solid', \
                                color = colors[ind_ep],linewidth=lwd)
                    elif vary_g == True:
                        ax1[2].plot(data_ml2,label='Ours '+r'$\gamma=$'+str(gamma) \
                            ,linestyle='solid', \
                                color = colors[ind_g],linewidth=lwd)
                    elif vary_bat == True:
                        ax1[2].plot(data_ml2,label='Ours Recharge '+bat_vec2[ind_bat].capitalize() \
                            ,linestyle='solid', \
                                color = colors[ind_bat],linewidth=lwd)
                    # elif vary_cap == True:
                    #     ax1[2].plot(data_ml2,label='Capacity '+cap.capitalize() \
                    #         ,linestyle='solid', \
                    #             color = colors[ind_cap],linewidth=lwd)     
                    # elif vary_pen == False: #True:
                    #     ax1[2].plot(data_ml2,label='Penalty '+pen.capitalize()  \
                    #         ,linestyle='solid', \
                    #             color = colors[ind_pen],linewidth=lwd)

# ax1[1].set_title('b)',fontsize=20,y=-0.32) # Battery Over Time',fontsize=15,y=-0.24)
ax1[2].set_title('(c)',fontsize=20,y=-0.35) # Battery Over Time',fontsize=15,y=-0.24)
ax1[2].grid(True)
ax1[2].set_xlabel('Epoch',fontsize=20)
ax1[2].set_ylabel('Learning Objective',fontsize=20)


# %% plotting the greedy DRL
seed = 1

for ep_start in [0.7]:#[0.6,0.8]:
    for gamma in [0.7]:#[0.7,0.8]:
        for g_ind,greed_style in enumerate([0,1,2]):
            if greed_style != 2:                
                with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(ep_start) \
                        +'_rewardtest_large_'+str(gamma) \
                        +'_greedy_'+str(greed_style), \
                        'rb') as f:
                    g_reward = pickle.load(f)
                with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(ep_start) \
                        +'_batterytest_large_'+str(gamma) \
                        +'_greedy_'+str(greed_style), \
                        'rb') as f:
                    g_bat = pickle.load(f)                    
                
                with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(ep_start) \
                        +'_ml_reward_onlytest_large_'+str(gamma) \
                        +'_greedy_'+str(greed_style), \
                        'rb') as f:
                    g_ml = pickle.load(f)                      
            else:
                with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(ep_start) \
                        +'_rewardtest_large_'+str(gamma) \
                        +'_greedy_'+str(greed_style)+'_rng_thresh_0.2', \
                        'rb') as f:
                    g_reward = pickle.load(f)
                with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(ep_start) \
                        +'_batterytest_large_'+str(gamma) \
                        +'_greedy_'+str(greed_style)+'_rng_thresh_0.2', \
                        'rb') as f:
                    g_bat = pickle.load(f)                 

                with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(ep_start) \
                        +'_ml_reward_onlytest_large_'+str(gamma) \
                        +'_greedy_'+str(greed_style)+'_rng_thresh_0.2', \
                        'rb') as f:
                    g_ml = pickle.load(f)    
                    
            # fig,ax_g = plt.subplots(1,2,figsize=(10,4))
            g_reward2 = moving_average(g_reward,1000)
            g_bat2 = [mean(i) for i in g_bat]
            g_bat2 = moving_average(g_bat2,1000)            
            g_ml2 = moving_average(g_ml,1000)
            
        #     if greed_style == 0:
        #         # print('temp')
        #         # ax1[0].plot(g_reward2,label='S.H.',color='saddlebrown',\
        #         #     linewidth = lwd) #Sequential Heuristic

        #         # ax1[1].plot(g_bat2,label='S.H.',color='saddlebrown',\
        #         #     linewidth = lwd) #Sequential Heuristic
                    
        #         ax1[2].plot(g_ml2,label='S.H.',color='saddlebrown',\
        #             linewidth = lwd)
                    
        #     elif greed_style == 1:
        #         # ax1[0].plot(g_reward2,label='G.M.D.',color='slategrey',\
        #         #     linewidth = lwd) #Greedy Minimum Distance

        #         # ax1[1].plot(g_bat2,label='G.M.D.',color='slategrey',\
        #         #     linewidth = lwd)
                    
        #         ax1[2].plot(g_ml2,label='G.M.D.',color='slategrey',\
        #             linewidth = lwd)                    
        #     else:
        #         # ax1[0].plot(g_reward2,label='T.M.D.',color='darkorange',\
        #         #     linewidth = lwd) # T.M.D. - threshold minimum distance

        #         # ax1[1].plot(g_bat2,label='T.M.D.',color='darkorange',\
        #         #     linewidth = lwd) # T.M.D.
                
        #         ax1[2].plot(g_ml2,label='T.M.D.',color='darkorange',\
        #             linewidth = lwd)                                     
        # # plt.savefig(cwd+'/drl_plots/freq_no_recharge.pdf',dpi=1000, bbox_inches='tight')


# ax1[0].set_ylim([300,800])

h,l = ax1[0].get_legend_handles_labels()
kw = dict(ncol=3,loc = 'lower center',frameon=False)
kw2 = dict(ncol=3,loc = 'lower center',frameon=False)
leg1 = ax1[0].legend(h[:3],l[:3],bbox_to_anchor=(-0.42,0.95,4.6,0.2),\
                fontsize=18,**kw) # mode='expand',
leg2 = ax1[0].legend(h[3:],l[3:],bbox_to_anchor=(-0.42,1.05,4.6,0.2),\
                fontsize=18,**kw2)
ax1[0].add_artist(leg1)
# ax1[0].add_artist(leg2)

if vary_ep == True:
    ax1[1].set_yticklabels(['25','30','35','40','45'])
elif vary_g == True:
    ax1[1].set_yticklabels(['10','20','30','40']) #['15','20','25','30','35','40','45'])
# elif vary_bat == True:
#     ax1[1].set_yticklabels(['20','25','30','35','40','45'])

f1.tight_layout()#pad=1.5)
f1.subplots_adjust(wspace=0.6)

ax1[0].tick_params(axis='both', which='major', labelsize=18)
ax1[1].tick_params(axis='both', which='major', labelsize=18)
ax1[2].tick_params(axis='both', which='major', labelsize=18)

# if vary_ep == True:
#     f1.savefig(cwd+'/drl_plots/RNN_drl_ovr_comp_epsilon.pdf',dpi=1000, bbox_inches='tight')
# elif vary_g == True:
#     f1.savefig(cwd+'/drl_plots/RNN_drl_ovr_comp_gamma.pdf',dpi=1000, bbox_inches='tight')
# elif vary_bat == True:
#     f1.savefig(cwd+'/drl_plots/RNN_drl_ovr_comp_bat.pdf',dpi=1000, bbox_inches='tight')

# %% replot the paper

seed = 1
# %% visit frequency plotter
for ep_start in [0.7]:#[0.6,0.8]:
    for gamma in [0.7]:#[0.7,0.8]:
        # with open(cwd+'/data/new10'+str(ep_start)+'_visit_freq_large_'\
        #           +str(gamma)+'_dynamic','rb') as f:
        #     freqs = pickle.load(f)

        with open(cwd+'/drl_results/RNN/seed_'+str(seed)+'_'\
                  +str(ep_start)+'_visit_freq_large_'\
                  +str(gamma)+'_tanh_mse_dynamic','rb') as f:
            freqs = pickle.load(f)

        # The dynamic model drift is linear
        
        # extract freqs in intervals of 1000
        ## because timestep%100, last batch is at timestep 9800
        # final_freq = freqs[9800] - freqs[8800]
        # mid_freq2 = freqs[6800] - freqs[5800]
        # mid_freq = freqs[3800] - freqs[2800]
        # init_freq = freqs[1800] - freqs[800]
        
        # timestep%100, only after first 100 has passed, 29800 is the final one
        final_freq = freqs[29800] - freqs[28800]
        # mid_freq2 = freqs[18800] - freqs[17800]
        # mid_freq2 = freqs[24800] - freqs[23800]
        mid_freq = freqs[15800] - freqs[14800]
        init_freq = freqs[1800] - freqs[800]
        
        x = np.arange(len(final_freq))  # the label locations
        width = 0.6 #0.8  # the width of the bars
        
        # setup double axes
        cats = 3 #4 #cats = categories
        plt.figure(3)
        fig, ax = plt.subplots(1,1,figsize=(4,2))
        rects_init = ax.bar(x - 1.02*width/cats, init_freq/1000, width=width/cats, \
            label='1000-2000',edgecolor='black',color='goldenrod')
        rects_mid1 = ax.bar(x - 0*width/cats, mid_freq/1000, width=width/cats, \
            label='15000-16000',edgecolor='black',color='forestgreen')
        # rects_mid2 = ax.bar(x + 0.5*width/cats, mid_freq2, width=width/cats, \
        #     label='5000-6000',edgecolor='black',color='royalblue')
        rects_final = ax.bar(x + 1.02*width/cats, final_freq/1000, width=width/cats, \
            label='29000-30000',edgecolor='black',color='darkmagenta')

        # plt.title('Hist frequences with recharge stations')
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_ylabel('Visit Rate (Visits/Epoch)',fontsize=11)
        #' Rate (Freq/1k Epochs)',fontsize=11) #visit frequency
        ax.legend(fontsize=4)
        
        loc = mpl.ticker.MultipleLocator(base=1.0)
        ax.xaxis.set_major_locator(loc)
        
        ax_ticks = ['C:'+str(i) for i in range(9)]
        ax_ticks += ['R:'+str(i+1) for i in range(2)]
        ax.set_xticklabels(ax_ticks) #this thing always drops the index 0 for some reason
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        h,l = ax.get_legend_handles_labels()
        kw = dict(ncol=4,loc = 'lower center',frameon=False)
        # kw2 = dict(ncol=3,loc = 'lower center',frameon=False)
        #(x, y, width, height)
        leg1 = ax.legend(h[:],l[:],bbox_to_anchor=(-0.05,0.98,1.05,0.2),\
                        mode='expand',fontsize=9,**kw)
        # leg2 = ax1[0].legend(h[0::2],l[0::2],bbox_to_anchor=(0.1,1.11,1.8,0.2),\
                                # mode='expand',fontsize='large',**kw)
        ax.add_artist(leg1)

        rect = patches.Rectangle((1.62,0),0.78,500/1000,linewidth=1.5,edgecolor='red',facecolor='none')
        rect2 = patches.Rectangle((2.62,0),0.78,300/1000,linewidth=1.5,edgecolor='black',facecolor='none')
        
        ax.add_patch(rect)
        ax.add_patch(rect2)
        
        # ax.text(0.5,620,'Max ',fontsize=9)
        # ax.text(5.5,620,'Min Model Drift',fontsize=9)
        ax.text(1.2,520/1000,'Max MD', fontsize=9)
        ax.text(2.5,320/1000,'Min MD',fontsize=9)
        ax.text(2.5,1150/1000, 'Epoch Range:',fontsize=11)
        
        # plt.savefig(cwd+'/drl_plots/RNN_freq_recharge.pdf',dpi=1000, bbox_inches='tight')






