# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:31:43 2020

@author: henry
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

from Shared_ML_code.neural_nets import MLP, CNN, FedAvg, FPAvg, LocalUpdate, \
    LocalUpdate_PFL, FedAvg2, LocalUpdate_FO_PFL, LocalUpdate_HF_PFL
from Shared_ML_code.testing import test_img, test_img2


# %% import neural network data
# seed declarations
init_seed = 1
random.seed(init_seed)
np.random.seed(init_seed)
torch.manual_seed(init_seed)

# data import and device spec
trans_mnist = transforms.Compose([transforms.ToTensor(), \
                                  transforms.Normalize((0.1307,),(0.3081,))])
dataset_train = torchvision.datasets.MNIST('./data/mnist/',train=True,download=False,\
                                            transform=trans_mnist)
dataset_test = torchvision.datasets.MNIST('./data/mnist/',train=False,download=False,\
                                          transform=trans_mnist)
device = torch.device('cuda')
#device = torch.device('cpu')

# %% filtering the ML data
# label split
train = {i: [] for i in range(10)}
for index, (pixels,label) in enumerate(dataset_train):
    train[label].append(index)
    
test = {i: [] for i in range(10)} 
for index, (pixels,label) in enumerate(dataset_test):
    test[label].append(index)    

data_source = 'mnist' # delete once argparse is configured

# assign datasets to nodes
clusters = 3
swarms = 3
swarm_period = 2#5
global_period = 2
cycles = 20
total_time = swarm_period*global_period*cycles

nodes_per_cluster = [np.random.randint(2,6) for i in range(swarms)]

#labels_per_node = [np.random.randint(1,6) for i in range(nodes)] #number of labels changes over time
#labels_set = {i: [] for i in range(nodes)} #randomly determined based on labels_per_node

# labels_per_node (i.e., distribution) changes over time...
# static_lpc = [np.random.randint(2,5) for i in range(swarms)] #static qty of labels per node
static_lpc = [8 for i in range(swarms)]

static_ls = {i: [] for i in range(swarms)} # actual labels at each node

# variable lpn and ls have rows: time, cols: nodes
var_lpc = np.zeros((total_time,swarms))
for i in range(total_time):
    var_lpc[i,:] = [np.random.randint(2,5) for i in range(swarms)]
    
var_ls = {j: {i: [] for i in range(swarms)} for j in range(total_time)}

## TODO: epsilon based changes in distribution - calc KL divergence for that

# %% populating ML label holders
## TODO: epsilon based changes - see KL divergence     
def pop_labels(temp_lpn,temp_ls,max_labels=10):
    for i,j in enumerate(temp_lpn):
        j = int(j)
        temp_ls[i] = sorted(random.sample(range(max_labels),j))
    return temp_ls

# pop holders
static_ls = pop_labels(static_lpc,static_ls)

for i in range(total_time):
    var_ls[i] = pop_labels(var_lpc[i,:],var_ls[i])


# random data qty per label
avg_qty = int(len(dataset_train)/(sum(nodes_per_cluster))) #*total_time))

# need to determine data per device and total data per swarm
static_qty = np.random.normal(avg_qty,avg_qty/10,size=(sum(nodes_per_cluster))).astype(int)
static_data_per_swarm = []
counter = 0
for i,j in enumerate(nodes_per_cluster):
    static_data_per_swarm.append(0)
    
    for k in range(j):
        static_data_per_swarm[i] += static_qty[counter]
        counter += 1
    

var_qty = [np.random.normal(avg_qty,avg_qty/10,size=(sum(nodes_per_cluster))).astype(int) \
           for j in range(total_time)]

# calculate training datasets per node
static_nts = {i: [] for i in range(sum(nodes_per_cluster))} #static_node_train_sets
var_nts = {j:{i:[] for i in range(sum(nodes_per_cluster))} for j in range(total_time)}

def pop_nts(temp_ls,temp_qty,temp_nts,npc,train=train): #nts - node training set
    counter = 0
    
    for ind_t_cluster, t_cluster in enumerate(npc): #npc = [3,5,2], nodes per cluster
        temp_ls_inner = temp_ls[ind_t_cluster]    
        for i in range(t_cluster): 
            for curr_label in temp_ls_inner:
                temp_nts[counter] += random.sample(train[curr_label],\
                                int(temp_qty[counter]/len(temp_ls_inner)))
            
            counter += 1

    return temp_nts

static_nts = pop_nts(static_ls,static_qty,static_nts,nodes_per_cluster)

## no variable time at the moment, get static_nts working first
# for j in range(total_time):
#     var_nts[j] = pop_nts(var_ls[j],var_qty[j],var_nts[j],nodes_per_cluster)

# # saving the data
# cwd = os.getcwd()
# with open(cwd+'/data/'+str(init_seed)+data_source+str(nodes)+'_lpn','wb') as f:
#     pickle.dump()

# %% personalize the testing dataset
## basically just sort the testing dataset into indexes for each cluster
cluster_test_sets = {i:[] for i in range(clusters)} #indexed by cluster

for i in range(clusters):
    cluster_ls = static_ls[i]
    
    for j in cluster_ls:
        cluster_test_sets[i] += test[j]

# %% create neural networks
## setup FL
d_in = np.prod(dataset_train[0][0].shape)
d_h = 64
d_out = 10
global_net = MLP(d_in,d_h,d_out).to(device)
print(global_net)
global_net.train()
# default_w = deepcopy(global_net.state_dict())
cwd = os.getcwd()
with open(cwd+'/data/default_w','rb') as f:
    default_w = pickle.load(f)

# one central model object for each swarm
fl_swarm_models = [MLP(d_in,d_h,d_out).to(device) for i in range(swarms)]
for i in fl_swarm_models:
    i.load_state_dict(default_w)
    i.train()

## setup PFL - same as FL, just an additional object
pfl_swarm_models = [MLP(d_in,d_h,d_out).to(device) for i in range(swarms)]
for i in pfl_swarm_models:
    i.load_state_dict(default_w)
    i.train()

FO_hn_pfl_swarm_models = [MLP(d_in,d_h,d_out).to(device) for i in range(swarms)]
for i in FO_hn_pfl_swarm_models:
    i.load_state_dict(default_w)
    i.train()

HF_hn_pfl_swarm_models = [MLP(d_in,d_h,d_out).to(device) for i in range(swarms)]
for i in HF_hn_pfl_swarm_models:
    i.load_state_dict(default_w)
    i.train()

## ovr ML params setup
lr = 1e-3
# lr2 = 1e-3

# %% running for all time
fl_acc = []
hn_pfl_acc = [] 
FO_hn_pfl_acc = []
HF_hn_pfl_acc = []

for t in range(total_time):
    swarm_w = {i:[] for i in range(swarms)}
    # data_processed = {i:0 for i in range(swarms)}
    
    #### Hierarchical-FL procedure 
    ### 1. create object for each node/device
    ### 2. after \tau1 = swarm_period iterations, aggregate cluster-wise (weighed)
    ### 3. after \tau2 = global_period swarm-wide aggregations, aggregate globally (weighted again)
    
    print('iteration:{}'.format(t))
    
    print('hierarchical FL begins here')
    
    uav_counter = 0
    for ind_i,val_i in enumerate(nodes_per_cluster):
        for j in range(val_i): # each uav in i
            local_obj = LocalUpdate(device,bs=10,lr=lr,epochs=swarm_period,\
                    dataset=dataset_train,indexes=static_nts[uav_counter])
            _,w,loss = local_obj.train(net=deepcopy(fl_swarm_models[ind_i]).to(device))
            
            swarm_w[ind_i].append(w)
            
            uav_counter += 1
    
    if (t+1) % (swarm_period*global_period) == 0:
        ## then a swarm-wide agg followed immediately by a global
        # swarm-wide agg
        t_static_qty = deepcopy(static_qty).tolist()
        
        t_swarm_total_qty = []
        w_swarms = []
        
        for ind_i,val_i in enumerate(nodes_per_cluster):
            t2_static_qty = t_static_qty[:val_i]
            del t_static_qty[:val_i]
            
            t3_static_qty = [i*swarm_period for i in t2_static_qty]
            
            w_avg_swarm = FedAvg2(swarm_w[ind_i],t3_static_qty)
            
            t_swarm_total_qty.append(sum(t3_static_qty))
            w_swarms.append(w_avg_swarm)
        
        # global agg
        w_global = FedAvg2(w_swarms,t_swarm_total_qty)
        
        
        for i in fl_swarm_models:
            i.load_state_dict(w_global)
            i.train()
            
    else:
        ## run FL swarm-wide aggregation only
        t_static_qty = deepcopy(static_qty).tolist()
        
        for ind_i,val_i in enumerate(nodes_per_cluster):
            t2_static_qty = t_static_qty[:val_i]
            del t_static_qty[:val_i]
            
            t3_static_qty = [i*swarm_period for i in t2_static_qty]
            
            w_avg_swarm = FedAvg2(swarm_w[ind_i],t3_static_qty)

            fl_swarm_models[ind_i].load_state_dict(w_avg_swarm)
            fl_swarm_models[ind_i].train()

    
    ## for clarity, splitting this outside of the other if-else statement
    ## evaluate model performance
    if (t+1) % (swarm_period*global_period) == 0:
        fl_acc_temp = 0
        for i,ii in enumerate(fl_swarm_models):
            ii.eval()
            fl_acc_temp += test_img2(ii,dataset_test,bs=10,\
                    indexes=cluster_test_sets[i],device=device)[0] * static_data_per_swarm[i] \
                / sum(static_data_per_swarm)
    
        # fl_acc.append(fl_acc_temp/len(fl_swarm_models))
        fl_acc.append(fl_acc_temp)
        
        # print(fl_acc_temp)
        # print(fl_acc[-1])

    
# %% saving results 
cwd = os.getcwd()

with open(cwd+'/data/fl_acc_test_base','wb') as f:
    pickle.dump(fl_acc,f)


# %% graveyard

#### HN-PFL procedure 
### 1. create object for each node/device
### 2. after \tau1 = swarm_period iterations, aggregate cluster-wise (weighed)
### 3. after \tau2 = global_period swarm-wide aggregations, aggregate globally (unweighted)

# print('HN-PFL begins here')

# uav_counter = 0
# for ind_i,val_i in enumerate(nodes_per_cluster):
#     for j in range(val_i): # each uav in i
#         local_obj = LocalUpdate_PFL(device,bs=10,lr=lr,epochs=swarm_period,\
#                 dataset=dataset_train,indexes=static_nts[uav_counter])
#         _,w,loss = local_obj.train(net=deepcopy(pfl_swarm_models[ind_i]).to(device))
        
#         swarm_w[ind_i].append(w)
        
#         uav_counter += 1





