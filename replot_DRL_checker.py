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
ep_start = 0.8
# gamma = 0.8 #gamma = 0.7 by default

with open(cwd+'/data/0_'+str(ep_start)+'_rewardtest_large_0.7_extra','rb') as f:
    data = pickle.load(f)

# with open(cwd+'/data/0_'+str(ep_start)+'_'+'reward'\
#           +'test_large'+'_'+str(gamma),'rb') as f:
#     data = pickle.load(f)

# plot average reward per 10 iterations
# data_fixer = [sum(data[j:j+1000])/1000 for j,jj in enumerate(data) if j % 1000 == 0]

data_fixer = moving_average(data,1000)

plt.figure()
plt.plot(data_fixer[:10000])
plt.title('moving avg 1000, reward, ep_start='+str(ep_start))
    # plt.xlabel('10th iterations instance')
    # plt.ylabel('average reward over the latest 10 iterations')
    # plt.title('final iteration was ' +str(10*i))


# plots for battery
with open(cwd+'/data/0_'+str(ep_start)+'_batterytest_large_0.7_extra','rb') as f:
    data_b = pickle.load(f)

# with open(cwd+'/data/0_'+str(ep_start)+'_'+'battery'\
#           +'test_large'+'_'+str(gamma),'rb') as f:
#     data_b = pickle.load(f)

data_b2 = [mean(i) for i in data_b]
data_b2_1 = [i[0] for i in data_b]
data_b2_2 = [i[1] for i in data_b]

data_b2 = moving_average(data_b2,1000)

plt.figure(2)
plt.plot(data_b2[:10000])
plt.title('moving avg 1000, avg battery levels,'+str(ep_start))

# plt.figure(3)
# plt.plot(data_b2_1)
# plt.title('battery level 1 '+str(ep_start))

# plt.figure(4)
# plt.plot(data_b2_2)
# plt.title('battery level 2 '+str(ep_start))

# visits
with open(cwd+'/data/'+str(ep_start)+'_visit_freq_large_0.7','rb') as f:
    data_freq = pickle.load(f)

data_b2 = moving_average(data_b2,1000)

plt.figure(3)
plt.plot(data_b2[:10000])
plt.title('moving avg 1000, avg battery levels,'+str(ep_start))



# min_data = [min(data[j:j+1000]) for j,jj in enumerate(data) if j % 1000 == 0]


# plt.figure(2)
# plt.plot(min_data[:-1])
# plt.title('min')


# median_data = [median(data[j:j+1000]) for j,jj in enumerate(data) if j%1000 == 0]

# plt.figure(3)
# plt.plot(median_data)
# plt.title('median')








































