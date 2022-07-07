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
from Shared_ML_code.cdsets import RML

# get all datasets
trans_mnist = transforms.Compose([transforms.ToTensor(), \
                                  transforms.Normalize((0.1307,),(0.3081,))])
dtrain_m = torchvision.datasets.MNIST('./data/mnist/',train=True,download=False,\
                                            transform=trans_mnist)
dtest_m = torchvision.datasets.MNIST('./data/mnist/',train=False,download=False,\
                                          transform=trans_mnist)
#True
dtrain_f = torchvision.datasets.FashionMNIST('./data/fmnist/',train=True,download=False,\
                                transform=transforms.ToTensor())
dtest_f = torchvision.datasets.FashionMNIST('./data/fmnist/',train=False,download=False,\
                                transform=transforms.ToTensor())

# # trans_cifar10 = transforms.Compose(
# #     [transforms.ToTensor(),
# #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
# dtrain_c10 = torchvision.datasets.CIFAR10('./data/cifar/',train=True,download=True)#,\
#                                 # transform=trans_cifar10)
# dtest_c10 = torchvision.datasets.CIFAR10('./data/cifar/',train=False,download=True)#,\
#                                 # transform=trans_cifar10)

# dtrain_celeba = torchvision.datasets.CelebA('./data/celeba/',split='test',target_type='identity',\
#                                         download=False)

dtrain_rml = dataset_train = RML(ldir='./data/radio_ml/',train=True)

# %% plot the datasets
import matplotlib.pyplot as plt
# plt.rcParams["figure.autolayout"] = True

m_targets = dtrain_m.targets
f_targets = dtrain_f.targets
# c_targets = np.array(dtrain_c10.targets)
rml_targets = dtrain_rml.y_data

## select one of each
m_imgs = []
f_imgs = []
# c_imgs = []
rml_imgs = []
for i in range(4):
    m_imgs.append(dtrain_m[np.where(m_targets == i)[0][0]][0])
    f_imgs.append(dtrain_f[np.where(f_targets == i)[0][0]][0])
    # c_imgs.append(dtrain_c10[np.where(c_targets == i)[0][0]][0])
    temp_rml = dtrain_rml.x_data[np.where(rml_targets == i)[0][0]]
    rml_imgs.append(temp_rml[0,0,:])#[20:80])

fig,ax = plt.subplots(4,4,figsize=(5,2),dpi=500)

for i,j in enumerate(range(0,4,1)):
    ax[0,i].imshow(m_imgs[j][0],cmap='gray')
    ax[0,i].set_xticks([])
    ax[0,i].set_yticks([])    
    
    ax[1,i].imshow(f_imgs[j][0],cmap='gray')
    ax[1,i].set_xticks([])
    ax[1,i].set_yticks([])

    # # temp = np.dstack((c_imgs[j][0],c_imgs[j][1],c_imgs[j][2]))
    # ax[2,i].imshow(c_imgs[j])#temp) #[c_imgs[j][0],c_imgs[j][1],c_imgs[j][2]])
    # ax[2,i].set_xticks([])
    # ax[2,i].set_yticks([])

    ax[3,i].plot(rml_imgs[j])
    ax[3,i].set_xticks([])
    ax[3,i].set_yticks([])

    ax[3,i].set_xlabel('Label:'+str(j))    

ax[0,0].set_ylabel('MNIST')
ax[1,0].set_ylabel('FMNIST')
ax[2,0].set_ylabel('CIFAR')
ax[3,0].set_ylabel('RADIOML')

cwd = os.getcwd()

# fig.savefig(cwd+'/plots/dsets.pdf',bbox_inches='tight',dpi=500)


