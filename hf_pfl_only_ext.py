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
    LocalUpdate_PFL, FedAvg2, LocalUpdate_trad_FO, LocalUpdate_HF_PFL, LocalUpdate_trad_HF
from Shared_ML_code.testing import test_img, test_img2
from Shared_ML_code.fl_parser import ml_parser

# gc.collect()
# torch.cuda.empty_cache()

# %% settings/parser
settings = ml_parser()

# %% import neural network data
# seed declarations
init_seed = 1
random.seed(init_seed)
np.random.seed(init_seed)
torch.manual_seed(init_seed)

if settings.comp == 'gpu':
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

# data import and device spec
if settings.data_style == 'mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), \
                                      transforms.Normalize((0.1307,),(0.3081,))])
    dataset_train = torchvision.datasets.MNIST('./data/mnist/',train=True,download=False,\
                                                transform=trans_mnist)
    dataset_test = torchvision.datasets.MNIST('./data/mnist/',train=False,download=False,\
                                              transform=trans_mnist)
elif settings.data_style == 'fmnist': 
    dataset_train = torchvision.datasets.FashionMNIST('./data/fmnist/',train=True,download=False,\
                                    transform=transforms.ToTensor())
    dataset_test = torchvision.datasets.FashionMNIST('./data/fmnist/',train=False,download=False,\
                                    transform=transforms.ToTensor())

# %% filtering the ML data
# label split
train = {i: [] for i in range(10)}
for index, (pixels,label) in enumerate(dataset_train):
    train[label].append(index)
    
test = {i: [] for i in range(10)} 
for index, (pixels,label) in enumerate(dataset_test):
    test[label].append(index)    

# %% filtering continued (the previous chunk is slow)
# assign datasets to nodes
total_time = 40#120
nodes_per_swarm = [np.random.randint(2,4) for i in range(settings.swarms)]

# labels_per_node assignment function
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

## main loop begins here ##
for save_type in [settings.iid_style]:
    # labels_per_cluster determination
    if save_type == 'extreme': #lpc = labels_per_cluster
        lpc = [1 for i in range(settings.swarms)] #static qty of labels per node
    elif save_type == 'mild':
        lpc = [np.random.randint(3,4) for i in range(settings.swarms)] #static qty of labels per node
    else:
        lpc = [10 for i in range(settings.swarms)]
    
    # data structure [online vs static data distributions]
    ## populating ML label holders
    if settings.online == True:
        var_lpc = np.zeros((total_time,settings.swarms))
        for i in range(total_time):
            var_lpc[i,:] = deepcopy(lpc)
        lpc = deepcopy(var_lpc) #overwrite for online data distributions
        ls = {j: {i: [] for i in range(settings.swarms)} for j in range(total_time)}
        
        if save_type == 'extreme':
            for i in range(total_time):
                ls[i] = pop_labels(lpc[i,:],ls[i])
        else:
            for i in range(total_time):
                ls[i] = pop_labels(lpc[i,:],ls[i],flag=False)
        
    else:
        ls = {i: [] for i in range(settings.swarms)} # actual labels at each swarm
        
        if save_type == 'extreme':
            ls = pop_labels(lpc,ls)
        else:
            ls = pop_labels(lpc,ls,flag=False)
    
    # %% populating data for nodes/swarms
    
    # data per device and total data per swarm
    avg_qty = 1000 #int(len(dataset_train)/sum(nodes_per_cluster)) # 650
    
    def pop_data_qty(data_holder,data_qty,nodes_per_swarm=nodes_per_swarm):
        counter = 0
        for ind_swarm,nodes_swarm in enumerate(nodes_per_swarm):
            data_holder.append(0)
            
            for node_ind in range(nodes_swarm):
                data_holder[ind_swarm] += data_qty[counter]
                counter += 1
                
        return data_holder
    
    if settings.online == True:
        data_per_swarm = {i: [] for i in range(total_time)}
        data_qty = np.random.normal(avg_qty,avg_qty/10,\
            size=(total_time,sum(nodes_per_swarm))).astype(int)
        
        # populate data qty tracker for easy aggregations
        for i in range(total_time):
            
            data_per_swarm[i] = pop_data_qty(data_per_swarm[i],data_qty[i,:])
    
    else:
        data_per_swarm = []
        data_qty = np.random.normal(avg_qty,avg_qty/10,\
            size=(sum(nodes_per_swarm))).astype(int)
        
        data_per_swarm = pop_data_qty(data_per_swarm,data_qty)
        
    # %% populating data indexes for swarms/nodes [training sets]
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
    
    # calculate training datasets per node
    if settings.online == True:
        node_train_sets = {j:{i:[] for i in range(sum(nodes_per_swarm))} \
                for j in range(total_time)}
        for j in range(total_time):
            node_train_sets[j] = pop_nts(ls[j],data_qty[j],\
                            node_train_sets[j],nodes_per_swarm)
            
    else:
        node_train_sets = {i: [] for i in range(sum(nodes_per_swarm))}
        node_train_sets = pop_nts(ls,data_qty,\
                        node_train_sets,nodes_per_swarm)
    
    # # saving the data
    # cwd = os.getcwd()
    # with open(cwd+'/data/'+str(init_seed)+data_source+str(nodes)+'_lpn','wb') as f:
    #     pickle.dump()
    
    # %% same as above for [testing dataset]
    ## basically just sort the testing dataset into indexes for each swarm
    swarm_test_sets = {i:[] for i in range(settings.swarms)} #indexed by swarm
    all_test_indexes = list(range(10000)) #10k test images in MNIST and FMNIST
    
    for i in range(settings.swarms):
        swarm_ls = deepcopy(ls[i])
        
        for j in swarm_ls:
            swarm_test_sets[i] += test[j]
    
    # %% create neural networks
    cwd = os.getcwd()
    ## setup FL
    if settings.nn_style == 'MLP':
        d_in = 784 #np.prod(dataset_train[0][0].shape)
        d_h = 64
        d_out = 10
        global_net = MLP(d_in,d_h,d_out).to(device)
        
        with open(cwd+'/data/default_w','rb') as f:
            default_w = pickle.load(f)  
        
        lr,lr2 = 1e-2,1e-2 #MLP
    else:
        nchannels = 1
        nclasses = 10
        global_net = CNN(nchannels,nclasses).to(device)
        
        # with open(cwd+'/data/CNN_default_w','rb') as f:
        #     default_w = pickle.load(f)        
    
        with open(cwd+'/data/CNN_new_w','rb') as f:
            default_w = pickle.load(f)             
        
        lr,lr2 = 1e-3,1e-2 #CNN
        
    print(global_net)
    global_net.train()
    
    # %% running for all time
    batch_size = 30 #12 #evenly divisible by 3
    
    # determine ratio inits 
    if settings.ratio_dynamic == False:
        swarm_period = 1
        global_period = 1
    else:
        swarm_period = settings.rd_val
        global_period = settings.rd_val
    
    ## main loop for ratio variance ##
    for ratio in [1,2,4,6,8,10]:
        # ratio dynamics
        if settings.ratio == 'global': #global dynamic, swarm varied
             global_period = swarm_period * ratio
        elif settings.ratio == 'swarm':
             swarm_period = global_period * ratio
            
        cycles = total_time/(swarm_period*global_period)
        # total_time = swarm_period*global_period*cycles
    
        HF_hn_pfl_acc, total_loss = [], []
        HF_hn_pfl_acc_full, total_loss_full = [], []
        
        if settings.nn_style =='MLP':
            HF_hn_pfl_swarm_models = [MLP(d_in,d_h,d_out).to(device) for i in range(settings.swarms)]
        else:
            HF_hn_pfl_swarm_models = [CNN(nchannels,nclasses).to(device) for i in range(settings.swarms)]   
            # print(default_w['fc2.bias'])            
        
        for i in HF_hn_pfl_swarm_models:
            i.load_state_dict(default_w)
            i.train()   
        
        ## main loop for hierarchical FL ##
        ### Hierarchical-FL procedure 
        ### 1. create object for each node/device
        ### 2. after \tau1 = swarm_period iterations, aggregate cluster-wise (weighed)
        ### 3. after \tau2 = global_period swarm-wide aggregations, aggregate globally (weighted again)        
        for t in range(total_time):
            swarm_w = {i:[] for i in range(settings.swarms)}
            # data_processed = {i:0 for i in range(swarms)}

            print('iteration:{}'.format(t))
            print('HN-HF-PFL begins here')
            
            uav_counter = 0
            for ind_i,val_i in enumerate(nodes_per_swarm):
                for j in range(val_i): # each uav in i
                    if settings.online == False:
                        local_obj = LocalUpdate_HF_PFL(device,bs=batch_size,lr=lr,lr2=lr2,epochs=1,\
                                dataset=dataset_train,indexes=node_train_sets[uav_counter])
                    else:
                        local_obj = LocalUpdate_HF_PFL(device,bs=batch_size,lr=lr,lr2=lr2,epochs=1,\
                                dataset=dataset_train,indexes=node_train_sets[t][uav_counter])
                            
                    _,w,loss = local_obj.train(net=deepcopy(HF_hn_pfl_swarm_models[ind_i]).to(device))
                    
                    swarm_w[ind_i].append(w)
                    uav_counter += 1
            
            # for i in HF_hn_pfl_swarm_models:
            #     print(i.state_dict()['fc2.bias'])

            # aggregation cycles
            if (t+1) % (swarm_period*global_period) == 0: # global agg
                # swarm-wide agg
                if settings.online == False:
                    temp_qty = deepcopy(data_qty).tolist()
                else:
                    temp_qty = 0*data_qty[t]
                    for t_prime in range(swarm_period*global_period):
                        temp_qty += data_qty[t-t_prime]
                    temp_qty = temp_qty.tolist()
                    
                t_swarm_total_qty = []
                w_swarms = []
                
                for ind_i,val_i in enumerate(nodes_per_swarm):
                    t2_static_qty = temp_qty[:val_i]
                    del temp_qty[:val_i]
                    
                    t3_static_qty = [i*swarm_period for i in t2_static_qty]

                    w_avg_swarm = FedAvg2(swarm_w[ind_i],t3_static_qty)
                    
                    t_swarm_total_qty.append(sum(t3_static_qty))
                    w_swarms.append(w_avg_swarm)
                
                # global agg
                w_global = FedAvg2(w_swarms,t_swarm_total_qty)
                
                for i in HF_hn_pfl_swarm_models:
                    i.load_state_dict(w_global)
                    i.train()
                    
            elif (t+1)% swarm_period == 0:
                ## run FL swarm-wide aggregation only
                if settings.online == False:
                    temp_qty = deepcopy(data_qty).tolist()
                else:
                    temp_qty = 0*data_qty[t]
                    for t_prime in range(swarm_period*global_period):
                        temp_qty += data_qty[t-t_prime]
                    temp_qty = temp_qty.tolist()
                    
                for ind_i,val_i in enumerate(nodes_per_swarm):
                    t2_static_qty = temp_qty[:val_i]
                    del temp_qty[:val_i]
                    
                    t3_static_qty = [i*swarm_period for i in t2_static_qty]
                    
                    w_avg_swarm = FedAvg2(swarm_w[ind_i],t3_static_qty)
        
                    HF_hn_pfl_swarm_models[ind_i].load_state_dict(w_avg_swarm)
                    HF_hn_pfl_swarm_models[ind_i].train()
        
            ## evaluate model performance
            if ((t+1) % (swarm_period*global_period) == 0):
                HF_hn_pfl_acc_temp, total_loss_temp = 0, 0
                
                HF_hn_pfl_acc_temp_all, total_loss_temp_all = 0, 0
                
                for i,ii in enumerate(HF_hn_pfl_swarm_models):
                    ii.eval()
                    temp_acc, loss = test_img2(ii,dataset_test,bs=batch_size,\
                            indexes=swarm_test_sets[i],device=device)
                    
                    # temp_acc_full, loss_full = test_img2(ii,dataset_test,\
                    #         bs=batch_size,indexes=all_test_indexes,device=device)
                    
                    HF_hn_pfl_acc_temp += temp_acc/len(HF_hn_pfl_swarm_models)
                    total_loss_temp += loss/len(HF_hn_pfl_swarm_models) #swarms
                    
                    # HF_hn_pfl_acc_temp_all += temp_acc_full * static_data_per_swarm[i] \
                    #     / sum(static_data_per_swarm)
                    # total_loss_temp_all += loss_full * static_data_per_swarm[i] \
                    #     / sum(static_data_per_swarm)
                
                # HF_hn_pfl_acc_temp_all, total_loss_temp_all = test_img2(ii,dataset_test,\
                #         bs=batch_size,indexes=all_test_indexes,device=device)
                
                HF_hn_pfl_acc.append(HF_hn_pfl_acc_temp)
                total_loss.append(total_loss_temp)
                
                # HF_hn_pfl_acc_full.append(HF_hn_pfl_acc_temp_all)
                # total_loss_full.append(total_loss_temp_all)
                
                print(HF_hn_pfl_acc[-1])
                # print(HF_hn_pfl_acc_full[-1])
                # print(total_loss)
        
        ## final instance for localized gradient descents
        print('final iteration - localized only')        
        swarm_w = {i:[] for i in range(settings.swarms)}
        
        uav_counter = 0
        for ind_i,val_i in enumerate(nodes_per_swarm):
            for j in range(val_i): # each uav in i
                if settings.online == False:
                    local_obj = LocalUpdate(device,bs=batch_size,lr=lr,lr2=lr2,epochs=1,\
                            dataset=dataset_train,indexes=node_train_sets[uav_counter])
                else:
                    local_obj = LocalUpdate(device,bs=batch_size,lr=lr,lr2=lr2,epochs=1,\
                            dataset=dataset_train,indexes=node_train_sets[t][uav_counter])
                
                _,w,loss = local_obj.train(net=deepcopy(HF_hn_pfl_swarm_models[ind_i]).to(device))
                
                swarm_w[ind_i].append(w)
                
                uav_counter += 1
        
        ## run FL swarm-wide aggregation only
        if settings.online == False:
            temp_qty = deepcopy(data_qty).tolist()
        else:
            temp_qty = 0*data_qty[t]
            for t_prime in range(swarm_period*global_period):
                temp_qty += data_qty[t-t_prime]
            temp_qty = temp_qty.tolist()
            
        for ind_i,val_i in enumerate(nodes_per_swarm):
            t2_static_qty = temp_qty[:val_i]
            del temp_qty[:val_i]
            
            t3_static_qty = [i*swarm_period for i in t2_static_qty]
            
            w_avg_swarm = FedAvg2(swarm_w[ind_i],t3_static_qty)

            HF_hn_pfl_swarm_models[ind_i].load_state_dict(w_avg_swarm)
            HF_hn_pfl_swarm_models[ind_i].train()
        
        HF_hn_pfl_acc_temp = 0
        total_loss_temp = 0
        
        for i,ii in enumerate(HF_hn_pfl_swarm_models):
            ii.eval()
            temp_acc, loss = test_img2(ii,dataset_test,bs=batch_size,\
                    indexes=swarm_test_sets[i],device=device)
            
            HF_hn_pfl_acc_temp += temp_acc/len(HF_hn_pfl_swarm_models)
            total_loss_temp += loss/len(HF_hn_pfl_swarm_models) #swarms

        HF_hn_pfl_acc.append(HF_hn_pfl_acc_temp)
        total_loss.append(total_loss_temp)
        
        # saving results
        cwd = os.getcwd()
        
        # streamline later this if-else is unneeded, but its 2 am rn
        if settings.iid_style == 'extreme':
            with open(cwd+'/data/hn_pfl_acc_'+settings.iid_style+'_'+str(ratio)+'_'+\
                settings.data_style+'_'+str(swarm_period)+'_'+str(global_period)+\
                '_'+settings.nn_style+'_debug','wb') as f:
                pickle.dump(HF_hn_pfl_acc,f)
        
            with open(cwd+'/data/hn_pfl_loss_'+settings.iid_style+'_'+str(ratio)+'_'+\
                settings.data_style+'_'+str(swarm_period)+'_'+str(global_period)+\
                '_'+settings.nn_style+'_debug','wb') as f:
                pickle.dump(total_loss,f)
            
            # with open(cwd+'/data/full_hn_pfl_acc_'+save_type+'_'+str(ratio)+'_'+str(data_source)\
            #           +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style,'wb') as f:
            #     pickle.dump(HF_hn_pfl_acc_full,f)
            
            # with open(cwd+'/data/full_hn_pfl_loss_'+save_type+'_'+str(ratio)+'_'+str(data_source)\
            #           +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style,'wb') as f:
            #     pickle.dump(total_loss_full,f)
            
        elif settings.iid_style == 'mild':
            with open(cwd+'/data/hn_pfl_acc_'+settings.iid_style+'_'+str(ratio)+'_'+\
                settings.data_style+'_'+str(swarm_period)+'_'+str(global_period)+\
                '_'+settings.nn_style+'_debug','wb') as f:
                pickle.dump(HF_hn_pfl_acc,f)
        
            with open(cwd+'/data/hn_pfl_loss_'+settings.iid_style+'_'+str(ratio)+'_'+\
                settings.data_style+'_'+str(swarm_period)+'_'+str(global_period)+\
                '_'+settings.nn_style+'_debug','wb') as f:
                pickle.dump(total_loss,f)
        
        # else:
        #     with open(cwd+'/data/full_hn_pfl_acc_'+save_type+'_'+str(ratio)+'_'+str(data_source)\
        #               +'_'+str(swarm_period)+'_'+str(global_period),'wb') as f:
        #         pickle.dump(HF_hn_pfl_acc_full,f)
        
        #     with open(cwd+'/data/full_hn_pfl_loss_'+save_type+'_'+str(ratio)+'_'+str(data_source)\
        #               +'_'+str(swarm_period)+'_'+str(global_period),'wb') as f:
        #         pickle.dump(total_loss,f)
        
# %% graveyard

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



