# -*- coding: utf-8 -*-
"""
@author: henry
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

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

for gamma in [0.6,0.7,0.8]:
    
    with open(cwd+'/data/10_'+str(ep_start)+'_rewardtest_large_'+str(gamma),'rb') as f:
        data = pickle.load(f)

    data_fixer = moving_average(data,2000)
    
    if gamma == 0.6:
        ax1[0].plot(data_fixer[:10000],label=str(gamma),linestyle='solid')
    elif gamma == 0.7:
        ax1[0].plot(data_fixer[:10000],label=str(gamma),linestyle='dashed')
    else:
        ax1[0].plot(data_fixer[:10000],label=str(gamma),linestyle='dotted')

ax1[0].set_title('moving avg 1000, reward, ep_start = ' + str(ep_start))
ax1[0].grid(True)
ax1[0].set_xlabel('Epoch')
ax1[0].set_ylabel('Reward')
ax1[0].legend()

# plt.xlabel('10th iterations instance')
# plt.ylabel('average reward over the latest 10 iterations')
# plt.title('final iteration was ' +str(10*i))

# with open(cwd+'/data/0_'+str(ep_start)+'_'+'reward'\
#           +'test_large'+'_'+str(gamma),'rb') as f:
#     data = pickle.load(f)

# plot average reward per 10 iterations
# data_fixer = [sum(data[j:j+1000])/1000 for j,jj in enumerate(data) if j % 1000 == 0]
# plots for battery

for gamma in [0.6,0.7,0.8]:
    with open(cwd+'/data/10_'+str(ep_start)+'_batterytest_large_'+str(gamma),'rb') as f:
        data_b = pickle.load(f)
    
    data_b2 = [mean(i) for i in data_b]
    data_b2 = moving_average(data_b2,1000)
    
    if gamma == 0.6:
        ax1[1].plot(data_b2[:10000],label=str(gamma),linestyle='solid')
    elif gamma == 0.7:
        ax1[1].plot(data_b2[:10000],label=str(gamma),linestyle='dashed')
    else:
        ax1[1].plot(data_b2[:10000],label=str(gamma),linestyle='dotted')
    
    
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






































