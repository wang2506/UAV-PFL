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

# dataset_train = torchvision.datasets.FashionMNIST('./data/fmnist/',train=True,download=False,\
#                                 transform=transforms.ToTensor())
# dataset_test = torchvision.datasets.FashionMNIST('./data/fmnist/',train=False,download=False,\
#                                 transform=transforms.ToTensor())

device = torch.device('cuda:2')
# device = torch.device('cpu')

# %% filtering the ML data
# label split
train = {i: [] for i in range(10)}
for index, (pixels,label) in enumerate(dataset_train):
    train[label].append(index)
    
test = {i: [] for i in range(10)} 
for index, (pixels,label) in enumerate(dataset_test):
    test[label].append(index)    

data_source = 'mnist' # delete once argparse is configured
# data_source = 'fmnist'

# assign datasets to nodes
clusters = 10
swarms = 10

total_time = 120
swarm_period = 2
# global_period = 2
# cycles = 20


nodes_per_cluster = [np.random.randint(2,4) for i in range(swarms)]

#labels_per_node = [np.random.randint(1,6) for i in range(nodes)] #number of labels changes over time
#labels_set = {i: [] for i in range(nodes)} #randomly determined based on labels_per_node

# labels_per_node (i.e., distribution) changes over time...
for save_type in ['extreme']:#,'extreme']: #['extreme','mild','iid']:
    if save_type == 'extreme':
        static_lpc = [1 for i in range(swarms)] #static qty of labels per node
    elif save_type == 'mild':
        static_lpc = [np.random.randint(3,4) for i in range(swarms)] #static qty of labels per node
    else:
        static_lpc = [10 for i in range(swarms)]
    
    static_ls = {i: [] for i in range(swarms)} # actual labels at each node
    
    # variable lpn and ls have rows: time, cols: nodes
    var_lpc = np.zeros((total_time,swarms))
    for i in range(total_time):
        var_lpc[i,:] = [np.random.randint(2,5) for i in range(swarms)]
        
    var_ls = {j: {i: [] for i in range(swarms)} for j in range(total_time)}
    
    ## TODO: epsilon based changes in distribution - calc KL divergence for that
    
    # %% populating ML label holders
    ## TODO: epsilon based changes - see KL divergence     
    def pop_labels(temp_lpn,temp_ls,max_labels=10,flag=True):
        starting_list = list(range(10))
        for i,j in enumerate(temp_lpn):
            j = int(j)
            tts = sorted(random.sample(range(max_labels),j))
            
            if flag == True: #extreme flag
                if tts[0] in starting_list:
                    temp_ls[i] = tts
                    del starting_list[starting_list.index(tts[0])]
                else:
                    sl_temp = starting_list[np.random.randint(0,len(starting_list))]
                    temp_ls[i].append(sl_temp)
                    
                    del starting_list[starting_list.index(sl_temp)]
            else:
                # first ensure that everying in the starting list is covered
                if len(starting_list) != 0:
                    if j >= len(starting_list):
                        tts = deepcopy(starting_list)
                        starting_list = []
                        tts2 = sorted(random.sample(range(max_labels),j-len(starting_list)))
                        
                        for val in tts2:
                            tts.append(val)
                    else: 
                        tts = sorted(random.sample(starting_list,j))
                    
                        for val in tts:
                            del starting_list[starting_list.index(val)]
                    temp_ls[i] = tts
                else:
                    temp_ls[i] = tts
                
        return temp_ls
    
    # pop holders
    if save_type == 'extreme':
        static_ls = pop_labels(static_lpc,static_ls)
    else:
        static_ls = pop_labels(static_lpc,static_ls,flag=False)
    # static_ls = {0:[0,1,2,3],1:[5,6,7],2:[4,8,9]}
    
    for i in range(total_time):
        var_ls[i] = pop_labels(var_lpc[i,:],var_ls[i])
    
    
    # random data qty per label
    # avg_qty = int(len(dataset_train)/(sum(nodes_per_cluster)*total_time))
    avg_qty = 1000 #int(len(dataset_train)/sum(nodes_per_cluster))
    # avg_qty = 650
    
    
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
    all_test_indexes = []
    # for i in range(10): #10 labels
        # all_test_indexes += test[i]
    all_test_indexes = list(range(10000)) #10k test images in MNIST and FMNIST
    
    
    for i in range(clusters):
        cluster_ls = static_ls[i]
        
        for j in cluster_ls:
            cluster_test_sets[i] += test[j]
    
    # %% create neural networks
    cwd = os.getcwd()
    ## setup FL
    nn_style = 'CNN'
    # nn_style = 'MLP'
    
    if nn_style == 'MLP':
        d_in = 784 #np.prod(dataset_train[0][0].shape)
        d_h = 64
        d_out = 10
        global_net = MLP(d_in,d_h,d_out).to(device)
        
        with open(cwd+'/data/default_w','rb') as f:
            default_w = pickle.load(f)  
        
    else:    
        nchannels = 1
        nclasses = 10
        global_net = CNN(nchannels,nclasses).to(device)
        
        with open(cwd+'/data/CNN_default_w','rb') as f:
            default_w = pickle.load(f)        
    
    print(global_net)
    
    global_net.train()
    
    
    # one central model object for each swarm
    
    # ## setup PFL - same as FL, just an additional object
    # pfl_swarm_models = [MLP(d_in,d_h,d_out).to(device) for i in range(swarms)]
    # for i in pfl_swarm_models:
    #     i.load_state_dict(default_w)
    #     i.train()
    
    # FO_hn_pfl_swarm_models = [MLP(d_in,d_h,d_out).to(device) for i in range(swarms)]
    # for i in FO_hn_pfl_swarm_models:
    #     i.load_state_dict(default_w)
    #     i.train()
    
    # HF_hn_pfl_swarm_models = [MLP(d_in,d_h,d_out).to(device) for i in range(swarms)]
    # for i in HF_hn_pfl_swarm_models:
    #     i.load_state_dict(default_w)
    #     i.train()
    
    ## ovr ML params setup
    lr = 1e-2 #1e-2
    # lr2 = 1e-3
    
    # %% running for all time
    batch_size = 30 #12 #evenly divisible by 3
    
    swarm_period = 1
    global_period = 1
    for ratio in [1,2,4]: #[0.5,1,1.5,2,2.5]:
        if ratio == 0:
            global_period = 1
            # swarm_period = 1
        else:
            global_period = swarm_period*ratio
            # swarm_period = global_period*ratio
            
        cycles = total_time/(swarm_period*global_period)
        # total_time = swarm_period*global_period*cycles
    
        fl_acc = []
        hn_pfl_acc = [] 
        FO_hn_pfl_acc = []
        HF_hn_pfl_acc = []
        total_loss = []
        
        fl_acc_full = []
        total_loss_full = []
        
        if nn_style =='MLP':
            fl_swarm_models = [MLP(d_in,d_h,d_out).to(device) for i in range(swarms)]
        else:
            fl_swarm_models = [CNN(nchannels,nclasses).to(device) for i in range(swarms)]   
            print(default_w['fc2.bias'])            
        
        for i in fl_swarm_models:
            i.load_state_dict(default_w)
            i.train()
            
        
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
                    local_obj = LocalUpdate(device,bs=batch_size,lr=lr,epochs=1,\
                            dataset=dataset_train,indexes=static_nts[uav_counter])
                    # _,w,loss = local_obj.train(net=deepcopy(fl_swarm_models[ind_i]).to(device))
                    #epochs = swarm_period
                    
                    _,w,loss = local_obj.train(net=fl_swarm_models[ind_i].to(device))
                    
                    swarm_w[ind_i].append(w)
                    
                    uav_counter += 1
            
            # for i in fl_swarm_models:
            #     print(i.state_dict()['fc2.bias'])            
            
            
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
                    
            elif (t+1)% swarm_period == 0:
                ## run FL swarm-wide aggregation only
                t_static_qty = deepcopy(static_qty).tolist()
                
                for ind_i,val_i in enumerate(nodes_per_cluster):
                    t2_static_qty = t_static_qty[:val_i]
                    del t_static_qty[:val_i]
                    
                    t3_static_qty = [i*swarm_period for i in t2_static_qty]
                    
                    w_avg_swarm = FedAvg2(swarm_w[ind_i],t3_static_qty)
        
                    fl_swarm_models[ind_i].load_state_dict(w_avg_swarm)
                    fl_swarm_models[ind_i].train()
        
            # break 
            ## for clarity, splitting this outside of the other if-else statement
            ## evaluate model performance
            if ((t+1) % (swarm_period*global_period) == 0):
            # if True: # every iteration - speed check
                fl_acc_temp = 0
                total_loss_temp = 0
                
                fl_acc_temp_all = 0
                total_loss_temp_all = 0 
                
                for i,ii in enumerate(fl_swarm_models):
                    ii.eval()
                    temp_acc, loss = test_img2(ii,dataset_test,bs=batch_size,\
                            indexes=cluster_test_sets[i],device=device)
                    
                    # temp_acc_full, loss_full = test_img2(ii,dataset_test,\
                    #         bs=batch_size,indexes=all_test_indexes,device=device)
                    
                    fl_acc_temp += temp_acc/len(fl_swarm_models)
                    total_loss_temp += loss/len(fl_swarm_models) #swarms
                    # fl_acc_temp += temp_acc * static_data_per_swarm[i] \
                        # / sum(static_data_per_swarm)
                    # total_loss_temp += loss * static_data_per_swarm[i] \
                        # / sum(static_data_per_swarm)
                    
                    # fl_acc_temp_all += temp_acc_full * static_data_per_swarm[i] \
                    #     / sum(static_data_per_swarm)
                    # total_loss_temp_all += loss_full * static_data_per_swarm[i] \
                    #     / sum(static_data_per_swarm)
                
                fl_acc_temp_all, total_loss_temp_all = test_img2(ii,dataset_test,\
                        bs=batch_size,indexes=all_test_indexes,device=device)
                
                # fl_acc.append(fl_acc_temp/len(fl_swarm_models))
                fl_acc.append(fl_acc_temp)
                total_loss.append(total_loss_temp)
                
                fl_acc_full.append(fl_acc_temp_all)
                total_loss_full.append(total_loss_temp_all)
                
                print(fl_acc[-1])
                print(fl_acc_full[-1])
                # print(total_loss)
        
        # ### calculate optim variables    
        # swarm_w_prev = default_w # used to calc optim variables
        # comp_w = swarm_w[0] # arbitrary
        
        # ## B_j, need magnitude
        # mag_B_j = 0
        # for i in default_w.keys():
        #     mag_B_j += torch.norm(1/lr*default_w[i] - 1/lr*comp_w[i])
            
        #     grad_diffs[0].append(1/lr*default_w[i] - 1/lr*comp_w[i])
        #     grad_diffs[1].append(1/lr*default_w[i] - 1/lr*swarm_w[1][i])
            
        # print(mag_B_j)
        # mu_j = 0
        # mu_j_grad = 0
        # mu_j_params = 0
        # for i in default_w.keys():
        #     mu_j_grad += torch.norm(1/lr * comp_w[i] - 1/lr * swarm_w[1][i])
        #     mu_j_params += torch.norm(comp_w[ )
        
    
        # B_j = [i - swarm_w] 
        
        # saving results
        cwd = os.getcwd()
        
        # streamline later this if-else is unneeded, but its 2 am rn
        if save_type == 'extreme':
            with open(cwd+'/data/fl_acc_'+save_type+'_'+str(ratio)+'_'+str(data_source)\
                      +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style,'wb') as f:
                pickle.dump(fl_acc,f)
        
            with open(cwd+'/data/fl_loss_'+save_type+'_'+str(ratio)+'_'+str(data_source)\
                      +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style,'wb') as f:
                pickle.dump(total_loss,f)
            
            with open(cwd+'/data/full_fl_acc_'+save_type+'_'+str(ratio)+'_'+str(data_source)\
                      +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style,'wb') as f:
                pickle.dump(fl_acc_full,f)
        
            with open(cwd+'/data/full_fl_loss_'+save_type+'_'+str(ratio)+'_'+str(data_source)\
                      +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style,'wb') as f:
                pickle.dump(total_loss_full,f)
            
        elif save_type == 'mild':
            with open(cwd+'/data/fl_acc_'+save_type+'_'+str(ratio)+'_'+str(data_source)\
                      +'_'+str(swarm_period)+'_'+str(global_period),'wb') as f:
                pickle.dump(fl_acc,f)
        
            with open(cwd+'/data/fl_loss_'+save_type+'_'+str(ratio)+'_'+str(data_source)\
                      +'_'+str(swarm_period)+'_'+str(global_period),'wb') as f:
                pickle.dump(total_loss,f)
        # else:
        #     with open(cwd+'/data/fl_acc_'+save_type+'_'+str(ratio)+'_'+str(data_source)\
        #               +'_'+str(swarm_period)+'_'+str(global_period),'wb') as f:
        #         pickle.dump(fl_acc,f)
        
        #     with open(cwd+'/data/fl_loss_'+save_type+'_'+str(ratio)+'_'+str(data_source)\
        #               +'_'+str(swarm_period)+'_'+str(global_period),'wb') as f:
        #         pickle.dump(total_loss,f)
        

        
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





