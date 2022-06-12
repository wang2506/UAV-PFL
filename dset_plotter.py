# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:13:26 2022

@author: ch5b2
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from copy import deepcopy
import random
import pickle
from Shared_ML_code.fl_parser import ml_parser


# get all datasets
trans_mnist = transforms.Compose([transforms.ToTensor(), \
                                  transforms.Normalize((0.1307,),(0.3081,))])
dtrain_m = torchvision.datasets.MNIST('./data/mnist/',train=True,download=False,\
                                            transform=trans_mnist)
dtest_m = torchvision.datasets.MNIST('./data/mnist/',train=False,download=False,\
                                          transform=trans_mnist)

dtrain_f = torchvision.datasets.FashionMNIST('./data/fmnist/',train=True,download=False,\
                                transform=transforms.ToTensor())
dtest_f = torchvision.datasets.FashionMNIST('./data/fmnist/',train=False,download=False,\
                                transform=transforms.ToTensor())

# %% plot the datasets
import matplotlib.pyplot as plt

m_targets = dtrain_m.targets
f_targets = dtrain_f.targets

## select one of each
m_imgs = []
f_imgs = []
for i in range(10):
    m_imgs.append(dtrain_m[np.where(m_targets == i)[0][0]][0])
    f_imgs.append(dtrain_f[np.where(f_targets == i)[0][0]][0])


fig,ax = plt.subplots(2,5,figsize=(5,2),sharey=True,dpi=500)

for i,j in enumerate(range(0,5,1)):
    ax[0,i].imshow(m_imgs[j][0],cmap='gray')
    ax[0,i].set_xticks([])
    ax[0,i].set_yticks([])    
    
    ax[1,i].imshow(f_imgs[j][0],cmap='gray')
    ax[1,i].set_xticks([])
    ax[1,i].set_yticks([])   

    ax[1,i].set_xlabel('Label:'+str(j))    

ax[0,0].set_ylabel('MNIST')
ax[1,0].set_ylabel('FMNIST')

cwd = os.getcwd()

fig.savefig(cwd+'/plots/dsets.pdf',bbox_inches='tight',dpi=500)


