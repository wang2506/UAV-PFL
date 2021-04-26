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

# %% moving average function

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# %% import data
cwd = os.getcwd()

# for i in np.arange(15,16,1):
# with open(cwd+'/data/'+str(0)+'_30_ep_CNN','rb') as f:
#     data = pickle.load(f)
ep_start = 0.6
# gamma = 0.8 #gamma = 0.7 by default
plt.figure(1)

f1,ax1 = plt.subplots(1,2,figsize=(10,4))

for ep_start in [0.7]:#[0.6,0.8]:
    for gamma in [0.6,0.7,0.8]:#[0.7,0.8]:
        
        with open(cwd+'/data/new10_'+str(ep_start)+'_rewardtest_large_'+str(gamma),'rb') as f:
            data = pickle.load(f)
    
        data_fixer = moving_average(data,1000)
        
        if ep_start == 0.6:
            if gamma == 0.6:
                ax1[0].plot(data_fixer,label=r'$\gamma$ =' + str(gamma)\
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='dotted')
            elif gamma == 0.7:
                ax1[0].plot(data_fixer,label=r'$\gamma$ = '+str(gamma) \
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='solid', color = 'forestgreen')
            else:
                ax1[0].plot(data_fixer,label=r'$\gamma$ = '+str(gamma) \
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='solid', color = 'darkblue')
        elif ep_start == 0.7:
            if gamma == 0.6:
                ax1[0].plot(data_fixer,label=r'$\gamma$ =' +str(gamma)\
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='dotted')
            elif gamma == 0.7:
                ax1[0].plot(data_fixer,label=r'$\gamma$ = '+str(gamma) \
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='dashed', color = 'forestgreen')
            else:
                ax1[0].plot(data_fixer,label=r'$\gamma$ = '+str(gamma) \
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='dashed', color = 'darkblue')
        else:
            if gamma == 0.6:
                ax1[0].plot(data_fixer,label=r'$\gamma$ =' +str(gamma)\
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='dashdot')
            elif gamma == 0.7:
                ax1[0].plot(data_fixer,label=r'$\gamma$ = '+str(gamma) \
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='dashdot',color = 'forestgreen')
            else:
                ax1[0].plot(data_fixer,label=r'$\gamma$ = '+str(gamma) \
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='dashdot',color = 'darkblue')

ax1[0].set_title(r'Moving Average 2000 Epochs')
ax1[0].grid(True)
ax1[0].set_xlabel('Epoch')
ax1[0].set_ylabel('Reward')
# ax1[0].legend()

# plt.xlabel('10th iterations instance')
# plt.ylabel('average reward over the latest 10 iterations')
# plt.title('final iteration was ' +str(10*i))

# with open(cwd+'/data/0_'+str(ep_start)+'_'+'reward'\
#           +'test_large'+'_'+str(gamma),'rb') as f:
#     data = pickle.load(f)

# plot average reward per 10 iterations
# data_fixer = [sum(data[j:j+1000])/1000 for j,jj in enumerate(data) if j % 1000 == 0]
# plots for battery
for ep_start in [0.7]:#[0.6,0.8]:
    for gamma in [0.6,0.7,0.8]:#[0.7,0.8]:
        with open(cwd+'/data/new10_'+str(ep_start)+'_batterytest_large_'+str(gamma),'rb') as f:
            data_b = pickle.load(f)
        
        data_b2 = [mean(i) for i in data_b]
        data_b2 = moving_average(data_b2,1000)
    
        if ep_start == 0.6:
            if gamma == 0.6:
                ax1[1].plot(data_b2,label=r'$\gamma$ =' + str(gamma)\
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='dotted')
            elif gamma == 0.7:
                ax1[1].plot(data_b2,label=r'$\gamma$ = '+str(gamma) \
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='solid', color = 'forestgreen')
            else:
                ax1[1].plot(data_b2,label=r'$\gamma$ = '+str(gamma) \
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='solid', color = 'darkblue')
        elif ep_start == 0.7:
            if gamma == 0.6:
                ax1[1].plot(data_b2,label=r'$\gamma$ =' + str(gamma)\
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='dotted')
            elif gamma == 0.7:
                ax1[1].plot(data_b2,label=r'$\gamma$ = '+str(gamma) \
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='dashed', color = 'forestgreen')
            else:
                ax1[1].plot(data_b2,label=r'$\gamma$ = '+str(gamma) \
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='dashed', color = 'darkblue')
        else:
            if gamma == 0.6:
                ax1[1].plot(data_b2,label=r'$\gamma$ =' + str(gamma)\
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='dashdot')
            elif gamma == 0.7:
                ax1[1].plot(data_b2,label=r'$\gamma$ = '+str(gamma) \
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='dashdot',color = 'forestgreen')
            else:
                ax1[1].plot(data_b2,label=r'$\gamma$ = '+str(gamma) \
                    + r' $\epsilon$ = ' + str(ep_start),linestyle='dashdot',color = 'darkblue')    
    
    
ax1[1].set_title('Moving Average 1000 Epochs')
ax1[1].grid(True)
ax1[1].set_xlabel('Epoch')
ax1[1].set_ylabel('Average Battery Level (kJ)')
ax1[1].set_yticklabels(['36','36','38','40','42','44','46'])
# ax1[1].legend()

# h,l = ax1[0].get_legend_handles_labels()
# kw = dict(ncol=4,loc = 'lower center',frameon=False)
# # kw2 = dict(ncol=3,loc = 'lower center',frameon=False)
# #(x, y, width, height)
# leg1 = ax1[0].legend(h[:],l[:],bbox_to_anchor=(0.0,1.03,2,0.2),\
#                 mode='expand',fontsize='large',**kw)
# # leg2 = ax1[0].legend(h[0::2],l[0::2],bbox_to_anchor=(0.1,1.11,1.8,0.2),\
#                         # mode='expand',fontsize='large',**kw)
# ax1[0].add_artist(leg1)


# plt.savefig(cwd+'/drl_plots/DRL.pdf',dpi=1000, bbox_inches='tight')

# # with open(cwd+'/data/0_'+str(ep_start)+'_'+'battery'\
# #           +'test_large'+'_'+str(gamma),'rb') as f:
# #     data_b = pickle.load(f)


# data_b2_1 = [i[0] for i in data_b]
# data_b2_2 = [i[1] for i in data_b]

# drift = [ 8.63901818,  9.91918686, 13.47041835, 10.47986286, 17.13048751,
        # 5.11130624, 21.95293591,  0.68468983]

    # plt.figure(2)
    # plt.plot(data_b2[:10000])
    # plt.title('moving avg 1000, avg battery levels,'+str(ep_start))

# plt.figure(3)
# plt.plot(data_b2_1)
# plt.title('battery level 1 '+str(ep_start))

# plt.figure(4)
# plt.plot(data_b2_2)
# plt.title('battery level 2 '+str(ep_start))

# # visits
# with open(cwd+'/data/10'+str(ep_start)+'_visit_freq_large_0.8','rb') as f:
#     data_freq = pickle.load(f)

# # plot histogram trends every 1k iterations
# g_timesteps = 10000

# for gt in range(g_timesteps/1000):
#     f_timestep = gt * 1000
    
#     data_freq 
    
#     plt.figure(10+f_timestep)
#     plt.hist()
    
    

# data_b2 = moving_average(data_b2,1000)

# plt.figure(3)
# plt.plot(data_b2[:10000])
# plt.title('moving avg 1000, avg battery levels,'+str(ep_start))



# min_data = [min(data[j:j+1000]) for j,jj in enumerate(data) if j % 1000 == 0]


# plt.figure(2)
# plt.plot(min_data[:-1])
# plt.title('min')


# median_data = [median(data[j:j+1000]) for j,jj in enumerate(data) if j%1000 == 0]

# plt.figure(3)
# plt.plot(median_data)
# plt.title('median')

# %% replot the paper


# %% visit frequency plotter
for ep_start in [0.7]:#[0.6,0.8]:
    for gamma in [0.7]:#[0.7,0.8]:
        with open(cwd+'/data/new10'+str(ep_start)+'_visit_freq_large_'\
                  +str(gamma)+'_dynamic','rb') as f:
            freqs = pickle.load(f)

        # The dynamic model drift is linear
        
        # extract freqs in intervals of 1000
        ## because timestep%100, last batch is at timestep 9800
        final_freq = freqs[9800] - freqs[8800]
        
        mid_freq2 = freqs[6800] - freqs[5800]
        
        mid_freq = freqs[3800] - freqs[2800]
        
        init_freq = freqs[1800] - freqs[800]
        
        x = np.arange(len(final_freq))  # the label locations
        width = 0.8  # the width of the bars
        
        # setup double axes
        cats = 4 #cats = categories
        plt.figure(3)
        fig, ax = plt.subplots(1,1,figsize=(4,2))
        rects_init = ax.bar(x - 1.5*width/cats, init_freq, width=width/cats, \
            label='Initial 1k',edgecolor='black',color='goldenrod')
        rects_mid1 = ax.bar(x - 0.5*width/cats, mid_freq, width=width/cats, \
            label='Middle 3k',edgecolor='black',color='forestgreen')
        rects_mid2 = ax.bar(x + 0.5*width/cats, mid_freq2, width=width/cats, \
            label='Middle 6k',edgecolor='black',color='royalblue')
        rects_final = ax.bar(x + 1.5*width/cats, final_freq, width=width/cats, \
            label='Final 10k',edgecolor='black',color='darkmagenta')

        # plt.title('Hist frequences with recharge stations')
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_ylabel('Visit Frequency',fontsize=9)
        ax.legend(fontsize=4)
        
        loc = mpl.ticker.MultipleLocator(base=1.0)
        ax.xaxis.set_major_locator(loc)
        
        ax_ticks = ['C '+str(i) for i in range(9)]
        ax_ticks += ['R '+str(i+1) for i in range(2)]
        ax.set_xticklabels(ax_ticks) #this thing always drops the index 0 for some reason
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        
        h,l = ax.get_legend_handles_labels()
        kw = dict(ncol=4,loc = 'lower center',frameon=False)
        # kw2 = dict(ncol=3,loc = 'lower center',frameon=False)
        #(x, y, width, height)
        leg1 = ax.legend(h[:],l[:],bbox_to_anchor=(-0.2,1.03,1.3,0.2),\
                        mode='expand',fontsize=9,**kw)
        # leg2 = ax1[0].legend(h[0::2],l[0::2],bbox_to_anchor=(0.1,1.11,1.8,0.2),\
                                # mode='expand',fontsize='large',**kw)
        ax.add_artist(leg1)


        plt.savefig(cwd+'/drl_plots/freq_recharge.pdf',dpi=1000, bbox_inches='tight')
        
        # # setup double axes
        # plt.figure(4)
        # fig,ax = plt.subplots()
        # # 2 recharge stations
        # y = np.arange(len(final_freq[:8]))
        # rects_init = ax.bar(y - 1.5*width/cats, init_freq[:8], width=width/cats, \
        #     label='initial',edgecolor='black',color='goldenrod')
        # rects_mid1 = ax.bar(y - 0.5*width/cats, mid_freq[:8], width=width/cats, \
        #     label='middle',edgecolor='black',color='forestgreen')
        # rects_mid2 = ax.bar(y + 0.5*width/cats, mid_freq2[:8], width=width/cats, \
        #     label='middle 2',edgecolor='black',color='royalblue')
        # rects_final = ax.bar(y + 1.5*width/cats, final_freq[:8], width=width/cats, \
        #     label='final',edgecolor='black',color='darkmagenta')
        
        # plt.title('hist frequencies without recharge stations')

        # ax.set_axisbelow(True)
        # ax.grid(True)
        # ax.set_ylabel('frequency over the last 1000 epochs')

        # loc = mpl.ticker.MultipleLocator(base=1.0)
        # ax.xaxis.set_major_locator(loc)
        
        # ax_ticks = ['C '+str(i) for i in range(9)]
        # # ax_ticks += ['R '+str(i+1) for i in range(2)]
        # ax.set_xticklabels(ax_ticks) #this thing always drops the index 0 for some reason

        # # plt.savefig(cwd+'/drl_plots/freq_no_recharge.pdf',dpi=1000, bbox_inches='tight')


# %% plotting the greedy DRL

# for ep_start in [0.7]:#[0.6,0.8]:
#     for gamma in [0.7]:#[0.7,0.8]:
#         for g_ind,greed_style in enumerate([0,1]):
            
#             with open(cwd+'/data/new10_'+str(ep_start)+'_rewardtest_large_'\
#                       +str(gamma)+'_greedy_'+str(greed_style),'rb') as f:
#                 g_reward = pickle.load(f)
    
#             with open(cwd+'/data/new10_'+str(ep_start)+'_batterytest_large_'\
#                       +str(gamma)+'_greedy_'+str(greed_style),'rb') as f:
#                 g_bat = pickle.load(f)


#             # fig,ax_g = plt.subplots(1,2,figsize=(10,4))
#             g_reward2 = moving_average(g_reward,1000)
#             g_bat2 = [mean(i) for i in g_bat]
#             g_bat2 = moving_average(g_bat2,1000)            
            
#             if greed_style == 0:
#                 ax1[0].plot(g_reward2,label='greedy graph',color='chocolate')

#                 ax1[1].plot(g_bat2,label='greedy graph',color='chocolate')
#             elif greed_style == 1:
#                 ax1[0].plot(g_reward2,label='greedy dist',color='magenta')

#                 ax1[1].plot(g_bat2,label='greedy dist',color='magenta')                
        
#         # plt.savefig(cwd+'/drl_plots/freq_no_recharge.pdf',dpi=1000, bbox_inches='tight')

# h,l = ax1[0].get_legend_handles_labels()
# kw = dict(ncol=5,loc = 'lower center',frameon=False)
# # kw2 = dict(ncol=3,loc = 'lower center',frameon=False)
# #(x, y, width, height)
# leg1 = ax1[0].legend(h[:],l[:],bbox_to_anchor=(-0.15,1.03,2.5,0.2),\
#                 mode='expand',fontsize='large',**kw)
# # leg2 = ax1[0].legend(h[0::2],l[0::2],bbox_to_anchor=(0.1,1.11,1.8,0.2),\
#                         # mode='expand',fontsize='large',**kw)
# ax1[0].add_artist(leg1)

# # f1.savefig(cwd+'/drl_plots/drl_ovr_comp.pdf',dpi=1000, bbox_inches='tight')










