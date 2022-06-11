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
# gamma = 0.8 #gamma = 0.7 by default
plt.figure(1)

# f1,ax1 = plt.subplots(1,2,figsize=(10,4))#(9.6,4)) #10,4
f1,ax1 = plt.subplots(1,3,figsize=(9,4))

for ep_start in [0.7]:#[0.6,0.8]:
    for gamma in [0.6,0.7]:#[0.7,0.8]:
        
        # with open(cwd+'/data/new10_'+str(ep_start)+'_rewardtest_large_'+str(gamma)\
        #     +'_dynamic','rb') as f:
        #     data = pickle.load(f)
    
        with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(ep_start)+'_'+'reward'\
                +'test_large'+'_'+str(gamma), \
                'rb') as f:
            data = pickle.load(f)
    
        data_fixer = moving_average(data,1000)
        
        if ep_start == 0.6:
            if gamma == 0.6:
                ax1[0].plot(data_fixer,label=r'DRL: $\gamma$ =' + str(gamma)\
                   ,linestyle='dotted',\
                        linewidth=lwd,color = 'darkgreen') # + r' $\epsilon$ = ' + str(ep_start)
            elif gamma == 0.7:
                ax1[0].plot(data_fixer,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='solid', \
                        color = 'magenta',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)
            else:
                ax1[0].plot(data_fixer,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='solid', \
                        color = 'darkgreen',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)
        elif ep_start == 0.7:
            if gamma == 0.6:
                ax1[0].plot(data_fixer,label=r'DRL: $\gamma$ =' +str(gamma)\
                    ,linestyle='dotted',\
                        linewidth=lwd,color = 'darkgreen') #+ r' $\epsilon$ = ' + str(ep_start)
            elif gamma == 0.7:
                ax1[0].plot(data_fixer,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='dashed', \
                        color = 'magenta',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)
            else:
                ax1[0].plot(data_fixer,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='dashed', \
                        color = 'darkgreen',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)
        else:
            if gamma == 0.6:
                ax1[0].plot(data_fixer,label=r'DRL: $\gamma$ =' +str(gamma)\
                    ,linestyle='dashdot',\
                        linewidth=lwd,color = 'darkgreen') #+ r' $\epsilon$ = ' + str(ep_start)
            elif gamma == 0.7:
                ax1[0].plot(data_fixer,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='dashdot', \
                        color = 'magenta',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)
            else:
                ax1[0].plot(data_fixer,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='dashdot', \
                        color = 'darkgreen',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)

# ax1[0].set_title(r'a)',fontsize=20,y=-0.32) # Reward Over Time
ax1[0].set_title(r'(a)',fontsize=20,y=-0.35) # Reward Over Time
ax1[0].grid(True)
ax1[0].set_xlabel('Epoch',fontsize=20)
ax1[0].set_ylabel('Reward',fontsize=20)
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
    for gamma in [0.6,0.7]:#[0.7,0.8]:
        # with open(cwd+'/data/new10_'+str(ep_start)+'_batterytest_large_'+str(gamma)\
        #           +'_dynamic','rb') as f:
        #     data_b = pickle.load(f)
        
        with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(ep_start)\
                +'_batterytest_large_'+str(gamma), \
                'rb') as f:
            data_b = pickle.load(f)
        
        data_b2 = [mean(i) for i in data_b]
        data_b2 = moving_average(data_b2,1000)
    
        if ep_start == 0.6:
            if gamma == 0.6:
                ax1[1].plot(data_b2,label=r'DRL: $\gamma$ =' + str(gamma)\
                    ,linestyle='dotted',\
                        linewidth=lwd,color = 'darkgreen') #+ r' $\epsilon$ = ' + str(ep_start)
            elif gamma == 0.7:
                ax1[1].plot(data_b2,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='solid', \
                        color = 'magenta',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)
            else:
                ax1[1].plot(data_b2,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='solid', \
                        color = 'darkblue',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)
        elif ep_start == 0.7:
            if gamma == 0.6:
                ax1[1].plot(data_b2,label=r'DRL: $\gamma$ =' + str(gamma)\
                    ,linestyle='dotted',\
                        linewidth=lwd,color = 'darkgreen') #+ r' $\epsilon$ = ' + str(ep_start)
            elif gamma == 0.7:
                ax1[1].plot(data_b2,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='dashed', \
                        color = 'magenta',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)
            else:
                ax1[1].plot(data_b2,label=r'DRL: $\gamma$ = '+str(gamma) \
                   ,linestyle='dashed', \
                        color = 'darkblue',linewidth=lwd) # + r' $\epsilon$ = ' + str(ep_start),
        else:
            if gamma == 0.6:
                ax1[1].plot(data_b2,label=r'DRL: $\gamma$ =' + str(gamma)\
                    ,linestyle='dashdot',\
                        linewidth=lwd,color = 'darkgreen') #+ r' $\epsilon$ = ' + str(ep_start)
            elif gamma == 0.7:
                ax1[1].plot(data_b2,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='dashdot', \
                        color = 'magenta',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)
            else:
                ax1[1].plot(data_b2,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='dashdot', \
                        color = 'darkblue',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)
    
    
# ax1[1].set_title('b)',fontsize=20,y=-0.32) # Battery Over Time',fontsize=15,y=-0.24)
ax1[1].set_title('(b)',fontsize=20,y=-0.35) # Battery Over Time',fontsize=15,y=-0.24)
ax1[1].grid(True)
ax1[1].set_xlabel('Epoch',fontsize=20)
ax1[1].set_ylabel('Avg Battery Level (kJ)',fontsize=20)

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

for ep_start in [0.7]:#[0.6,0.8]:
    for gamma in [0.6,0.7]:#[0.7,0.8]:
        # with open(cwd+'/data/new10_'+str(ep_start)+'_batterytest_large_'+str(gamma)\
        #           +'_dynamic','rb') as f:
        #     data_b = pickle.load(f)
        with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(ep_start)\
                +'_ml_reward_onlytest_large_'+str(gamma), \
                'rb') as f:
            data_ml = pickle.load(f)
        
        # data_ml2 = [mean(i) for i in data_ml]
        data_ml2 = moving_average(data_ml,1000)
    
        if ep_start == 0.6:
            if gamma == 0.6:
                ax1[2].plot(data_ml2,label=r'DRL: $\gamma$ =' + str(gamma)\
                    ,linestyle='dotted',\
                        linewidth=lwd,color = 'darkgreen') #+ r' $\epsilon$ = ' + str(ep_start)
            elif gamma == 0.7:
                ax1[2].plot(data_ml2,label=r'DRL: $\gamma$ = '+str(gamma) \
                   ,linestyle='solid', \
                        color = 'magenta',linewidth=lwd)  #+ r' $\epsilon$ = ' + str(ep_start)
            else:
                ax1[2].plot(data_ml2,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='solid', \
                        color = 'darkblue',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)
        elif ep_start == 0.7:
            if gamma == 0.6:
                ax1[2].plot(data_ml2,label=r'DRL: $\gamma$ =' + str(gamma)\
                    ,linestyle='dotted',\
                        linewidth=lwd,color = 'darkgreen') #+ r' $\epsilon$ = ' + str(ep_start)
            elif gamma == 0.7:
                ax1[2].plot(data_ml2,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='dashed', \
                        color = 'magenta',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)
            else:
                ax1[2].plot(data_ml2,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='dashed', \
                        color = 'darkblue',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)
        else:
            if gamma == 0.6:
                ax1[2].plot(data_ml2,label=r'DRL: $\gamma$ =' + str(gamma)\
                    ,linestyle='dashdot',\
                        linewidth=lwd,color = 'darkgreen') #+ r' $\epsilon$ = ' + str(ep_start)
            elif gamma == 0.7:
                ax1[2].plot(data_ml2,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='dashdot', \
                        color = 'magenta',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)
            else:
                ax1[2].plot(data_ml2,label=r'DRL: $\gamma$ = '+str(gamma) \
                    ,linestyle='dashdot', \
                        color = 'darkblue',linewidth=lwd) #+ r' $\epsilon$ = ' + str(ep_start)

# ax1[1].set_title('b)',fontsize=20,y=-0.32) # Battery Over Time',fontsize=15,y=-0.24)
ax1[2].set_title('(c)',fontsize=20,y=-0.35) # Battery Over Time',fontsize=15,y=-0.24)
ax1[2].grid(True)
ax1[2].set_xlabel('Epoch',fontsize=20)
ax1[2].set_ylabel('Learning Objective',fontsize=20)

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
            label='0-1000',edgecolor='black',color='goldenrod')
        rects_mid1 = ax.bar(x - 0.5*width/cats, mid_freq, width=width/cats, \
            label='2000-3000',edgecolor='black',color='forestgreen')
        rects_mid2 = ax.bar(x + 0.5*width/cats, mid_freq2, width=width/cats, \
            label='5000-6000',edgecolor='black',color='royalblue')
        rects_final = ax.bar(x + 1.5*width/cats, final_freq, width=width/cats, \
            label='9000-10000',edgecolor='black',color='darkmagenta')

        # plt.title('Hist frequences with recharge stations')
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_ylabel('Visit Rate (Freq/1k Epochs)',fontsize=11) #visit frequency
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
        leg1 = ax.legend(h[:],l[:],bbox_to_anchor=(-0.2,1.03,1.3,0.2),\
                        mode='expand',fontsize=9,**kw)
        # leg2 = ax1[0].legend(h[0::2],l[0::2],bbox_to_anchor=(0.1,1.11,1.8,0.2),\
                                # mode='expand',fontsize='large',**kw)
        ax.add_artist(leg1)

        rect = patches.Rectangle((1.54,0),0.94,600,linewidth=1.5,edgecolor='red',facecolor='none')
        rect2 = patches.Rectangle((2.54,0),0.94,300,linewidth=1.5,edgecolor='black',facecolor='none')
        
        ax.add_patch(rect)
        ax.add_patch(rect2)
        
        # ax.text(0.5,620,'Max ',fontsize=9)
        # ax.text(5.5,620,'Min Model Drift',fontsize=9)
        ax.text(1.2,620,'Max MD', fontsize=9)
        ax.text(2.55,320,'Min MD',fontsize=9)
        
        
        # plt.savefig(cwd+'/drl_plots/freq_recharge.pdf',dpi=1000, bbox_inches='tight')
        
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

for ep_start in [0.7]:#[0.6,0.8]:
    for gamma in [0.7]:#[0.7,0.8]:
        for g_ind,greed_style in enumerate([0,1,2]):
            if greed_style != 2:
            
                # with open(cwd+'/data/new10_'+str(ep_start)+'_rewardtest_large_'\
                #           +str(gamma)+'_greedy_'+str(greed_style),'rb') as f:
                #     g_reward = pickle.load(f)
        
                # with open(cwd+'/data/new10_'+str(ep_start)+'_batterytest_large_'\
                #           +str(gamma)+'_greedy_'+str(greed_style),'rb') as f:
                #     g_bat = pickle.load(f)
                
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
                # with open(cwd+'/data/new10_'+str(ep_start)+'_rewardtest_large_'\
                #           +str(gamma)+'_greedy_'+str(greed_style)+'_rng_thresh_0.2','rb') as f:
                #     g_reward = pickle.load(f)
        
                # with open(cwd+'/data/new10_'+str(ep_start)+'_batterytest_large_'\
                #           +str(gamma)+'_greedy_'+str(greed_style)+'_rng_thresh_0.2','rb') as f:
                #     g_bat = pickle.load(f)

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
            
            if greed_style == 0:
                # print('temp')
                ax1[0].plot(g_reward2,label='S.H.',color='saddlebrown',\
                    linewidth = lwd) #Sequential Heuristic

                ax1[1].plot(g_bat2,label='S.H.',color='saddlebrown',\
                    linewidth = lwd) #Sequential Heuristic
                    
                ax1[2].plot(g_ml2,label='S.H.',color='saddlebrown',\
                    linewidth = lwd)
                    
            elif greed_style == 1:
                ax1[0].plot(g_reward2,label='G.M.D.',color='slategrey',\
                    linewidth = lwd) #Greedy Minimum Distance

                ax1[1].plot(g_bat2,label='G.M.D.',color='slategrey',\
                    linewidth = lwd)
                    
                ax1[2].plot(g_ml2,label='G.M.D.',color='slategrey',\
                    linewidth = lwd)                    
            else:
                ax1[0].plot(g_reward2,label='T.M.D.',color='darkorange',\
                    linewidth = lwd) # T.M.D. - threshold minimum distance

                ax1[1].plot(g_bat2,label='T.M.D.',color='darkorange',\
                    linewidth = lwd) # T.M.D.
                
                ax1[2].plot(g_ml2,label='T.M.D.',color='darkorange',\
                    linewidth = lwd)                                     
        # plt.savefig(cwd+'/drl_plots/freq_no_recharge.pdf',dpi=1000, bbox_inches='tight')

h,l = ax1[0].get_legend_handles_labels()
# kw = dict(ncol=2,loc = 'lower center',frameon=False)
# kw2 = dict(ncol=3,loc = 'lower center',frameon=False)
kw = dict(ncol=5,loc = 'lower center',frameon=False)
#(x, y, width, height)
# leg1 = ax1[0].legend(h[:],l[:],bbox_to_anchor=(-0.25,1.03,2.5,0.2),\
                # mode='expand',fontsize=18,**kw)
# leg1 = ax1[0].legend(h[:2],l[:2],bbox_to_anchor=(-0.25,0.95,4,0.2),\
#                 mode='expand',fontsize=18,**kw)
# leg2 = ax1[0].legend(h[2:],l[2:],bbox_to_anchor=(-0.25,1.05,4,0.2),\
#                 mode='expand',fontsize=18,**kw2)
leg1 = ax1[0].legend(h[:],l[:],bbox_to_anchor=(-0.46,0.95,4.6,0.2),\
                mode='expand',fontsize=18,**kw)
# leg2 = ax1[0].legend(h[2:],l[2:],bbox_to_anchor=(-0.42,1.05,4.4,0.2),\
#                 mode='expand',fontsize=18,**kw2)
ax1[0].add_artist(leg1)
# ax1[0].add_artist(leg2)
# ax1[1].set_yticklabels(['36','36','38','40','42','44','46'])
# ax1[1].set_yticklabels(['25','27.5','30','32.5','35','37.5','40','42.5','45','47.5'])
# ax1[1].set_yticklabels(['25','30','35','40','45'])
# ax1[1].set_yticklabels(['27.5','30','32.5','35','37.5','40','42.5','45','47.5'])
ax1[1].set_yticklabels(['25','30','35','40','45'])
f1.tight_layout()

ax1[0].tick_params(axis='both', which='major', labelsize=18)
ax1[1].tick_params(axis='both', which='major', labelsize=18)
ax1[2].tick_params(axis='both', which='major', labelsize=18)
# f1.savefig(cwd+'/drl_plots/drl_ovr_comp.pdf',dpi=1000, bbox_inches='tight')
# f1.savefig(cwd+'/drl_plots/chris_drl_ovr_comp.pdf',dpi=1000, bbox_inches='tight')

f1.savefig(cwd+'/drl_plots/testml.pdf',dpi=1000, bbox_inches='tight')







