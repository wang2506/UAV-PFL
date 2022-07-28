# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 13:38:50 2021

@author: henry
"""
# %% pack imports
import numpy as np
from collections import deque
import argparse
import random

import pickle as pk
import os
from copy import deepcopy
from itertools import combinations, permutations, product
from math import factorial
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Flatten, \
    Conv1D, MaxPooling1D, Dropout, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import time

# seed = 2 #1
# seed = 1 #original runs use seed = 1
# seed = 2
# seed = 6
# seed = 3
seed = 4

start = time.time()
# %% parser function
parser = argparse.ArgumentParser()

## UAV and device cluster setups
parser.add_argument('--U_swarms',type=int,default=3,\
                    help='number of UAV swarms') #4
parser.add_argument('--Clusters',type=int,default=8,\
                    help='number of device clusters') #10
parser.add_argument('--total_UAVs',type=int,default=6,\
                    help='aggregate number of UAVs')
parser.add_argument('--min_uavs_swarm',type=int,default=4,\
                    help='minimum number of uavs per swarm')
parser.add_argument('--max_uavs_swarm',type=int,default=6,\
                    help='maximum number of uavs per swarm')

parser.add_argument('--UAV_init_ratio',type=float,default=0.66,\
                    help='initial worker to coorindator ratio') #0.8
parser.add_argument('--recharge_points',type=int, default=2,\
                    help='number of recharge points') #50% of total clusters; 3

## RL probs (epsilon and gamma)
parser.add_argument('--ep_greed',type=float,default=0.7,\
                    help='epsilon greedy val') #0.6,0.7,0.8
parser.add_argument('--ep_min',type=float,default=0.005,\
                    help='epsilon minimum')
parser.add_argument('--g_discount',type=float,default=0.7,\
                    help='gamma discount factor') #0.6,0.7,0.8
parser.add_argument('--replay_bs',type=int,default=20,\
                    help='experience replay batch size') #20, 50
parser.add_argument('--linear',type=bool,default=False,\
                    help='MLP or CNN') ## argpase cannot evaluate booleans OOB - fix later
parser.add_argument('--RNN',type=bool,default=True)
parser.add_argument('--cnn_range',type=int, default=2,\
                    help='conv1d range')

# ovr parameters
parser.add_argument('--G_timesteps',type=int,default=30000,\
                    help='number of swarm movements') #10,000
parser.add_argument('--training',type=int,default=1,\
                    help='training or testing the DRL')
parser.add_argument('--centralized',type=bool,default=True,\
                    help='centralized or decentralized')
parser.add_argument('--DQN_update_period',type=int,default=20,\
                    help='DQN update period') #10, 20, 50
#8,3,2,4,2,2

# parameters to find a greedy baseline
parser.add_argument('--greed_base',type=bool,default=False,\
                    help='greedy baseline calculation')
parser.add_argument('--greed_style',type=int,default=0,\
                    help='0: graph greed, 1: min dist, 2: rng min dist')
parser.add_argument('--rng_thresh',type=float,default=0.2,\
                    help='rng min dist threshold')

parser.add_argument('--dynamic',type=bool,default=False,\
                    help='Dynamic model drifts')

# hard coded perviously, need to backtrack to get this automated
# parser.add_argument('--dynamic_drift',type=bool,default=False,\
                    # help='dynamic model drift')

parser.add_argument('--brt',type=str,default='debug2',\
                    choices=['debug','debug2','medium','high','low',\
                             'vhigh','vhigh2','vhigh3','hlow','vlow','vvlow'],\
                    help='Battery Recharge Threshold')
parser.add_argument('--seed',type=int,default=4)
parser.add_argument('--pen',type=str,default='high',\
                    choices=['high','medium','low'])
parser.add_argument('--cap',type=str,default='low',\
                    choices=['low','medium','high'])

args = parser.parse_args()

seed = args.seed
print('seed:'+str(seed))
np.random.seed(seed)
random.seed(seed)
rng = np.random.default_rng(seed=seed)

# %% DQN object 

class DQN:
    def __init__(self,args,optimizer=Adam(learning_rate=0.001)):
        # inits
        self.U = np.arange(0,args.U_swarms,1) #the swarms as an array 
        self.C = np.random.randint(0,10,size=args.Clusters) # measures their difficulty (i.e., need more revisits)
        self.total_UAVs = args.total_UAVs
        
        self.recharge_points = np.arange(0,args.recharge_points,1) ## assume that recharging takes T?
        
        self.total_time = args.G_timesteps
        ## (swarm 0 position, swarm 0 min energy),... , (latest cluster visit times), ...
        
        # the factorial choosing lets us determine active swarms
        # and we have positions
        # need 3 scalars
        # self.action_size = factorial(args.Clusters+args.recharge_points)\
        #     /( factorial(args.U_swarms) *\
        #     factorial(args.Clusters+args.recharge_points-args.U_swarms)) \
        #     * 3 * 2* 2 # 3 Tc choices, 2 taus1, 2 taus2                
            #/factorial(args.Clusters+args.recharge_points-args.U_swarms) # all possible permutations
        
        temp_as = [i for i in permutations(range(args.Clusters+args.recharge_points),args.U_swarms)]
        
        self.action_size = len(temp_as)*3*2*2    
        
        
        #self.ep_greed = args.ep_greed
        self.g_discount = args.g_discount
        self.past_exps = deque(maxlen=300) #300 may be too small

        self.optimizer = optimizer
        
        # RL Q-function nets
        if args.linear == True:
            # uav positions, cluster last visit times, uav min battery levels
            # state breakdown:
            # 1) swarm active boolean vec
            # 2) concept drift at clusters
            # 3) min uav energy
            # 4) gradient of all clusters - a scalar; sum of learning perfs -> removed
            # 5) position of all swarms - use cluster + recharge station boolean
            # 6) Ts taus1 and taus2 -> 3 scalars
            
            self.input_size = args.U_swarms + args.Clusters + \
                args.U_swarms + args.Clusters + \
                args.recharge_points + 3  # +1 before args.Clusters
            
            self.q_net = self.build_linear_NN()
            #self.target_network = deepcopy(self.q_net) #deepcopy fails on TF pickled objects
            self.target_network = self.build_linear_NN()

        elif args.RNN == True:
            ## input shape
            self.input_size = args.U_swarms + args.Clusters + \
                args.U_swarms + args.Clusters + \
                args.recharge_points + 3  # +1 before args.Clusters
            # self.input_size = 2*self.input_size
            
            self.q_net = self.build_RNN()
            self.target_network = self.build_RNN()
        else:
            self.input_size = [args.cnn_range, args.U_swarms + args.Clusters]
            
            self.q_net = self.build_CNN()
            self.target_network = self.build_CNN()
        
        self.align_target_model()

        self.init_UAV_swarm_allocation(args)

    def init_UAV_swarm_allocation(self,args):
        # ## calculate the initial UAV swarm allocation
        # ## if there cannot be an equal allocation (i.e., all swarms init the same size)
        # ## then, run an incremental loop until all swarms populated
        # ups_holder = int(self.total_UAVs/args.U_swarms) * np.ones(args.U_swarms)
        # if self.total_UAVs % args.U_swarms != 0:
        #     #avg assign first, then randomly add the remainder in
        #     while np.sum(ups_holder) != self.total_UAVs:
        #         #randomly add 1 to a random swarm
        #         ups_holder[int(rng.random()*self.total_UAVs)] += 1

        # self.UAVs_per_swarm = ups_holder
        
        # ## allocate workers and coordinators (ratio rounding should favor workers)
        # self.coordinators_per_swarm = [int((1-args.UAV_init_ratio)*i) for i in self.UAVs_per_swarm]
        # self.workers_per_swarm = [self.UAVs_per_swarm[self.coordinators_per_swarm.index(i)]-i \
        #                         for i in self.coordinators_per_swarm]
        
        ## load in swarm and cluster characteristics from base file
        self.workers_per_swarm = []
        self.coordinators_per_swarm = []
        self.devices_per_cluster = []
        
        cwd = os.getcwd()
        for i in range(args.U_swarms):
            with open(cwd+'/geo_optim_chars/workers_swarm_no'+str(i),'rb') as f:
                self.workers_per_swarm.append(pk.load(f)[0])
            
            with open(cwd+'/geo_optim_chars/coordinators_swarm_no'+str(i),'rb') as f:
                self.coordinators_per_swarm.append(pk.load(f)[0])
        
        for i in range(args.Clusters):
            with open(cwd+'/geo_optim_chars/devices_cluster_no'+str(i),'rb') as f:
                self.devices_per_cluster.append(pk.load(f)[0])
        
        print(self.workers_per_swarm)
        print(self.coordinators_per_swarm)
        print(self.devices_per_cluster)
                
    ##########################################################################
    ## utilities
    def store(self,state,action,reward,next_state,args=args,\
              action2=0, reward2=0, next_state2=0):
        
        ## state contains cluster index + time of last swarm visit at cluster
        ## [to add min UAV energy within the swarm [[a_0,a_1,a_2]
        ## where a_0 = the device cluster/recharge position of the a-th swarm
        ## a_1 = time of last visit at cluster
        ## a_2 = the minimum UAV energy within the a-th swarm - losing charge incurs huge negative
        
        ## we index the device cluster by sequential integers and not x-y grid
        ## based on the index differential, we can determine the cost of commuting
        
        ## At the moment, the state is just the device cluster index
        ## randomly assigned at initialization [later on, need to make it so that
        ## it is assigned based on geography]
        
        if args.linear == True or args.RNN == True:
            state = np.reshape(state,[1, len(state)])
            next_state = np.reshape(next_state,[1,len(next_state)])
        
            # saving
            self.past_exps.append((state,action,reward,next_state))
        
        else:
            state1 = np.reshape(state,[1, len(state)])
            state2 = np.reshape(next_state,[1,len(next_state)])
            state3 = np.reshape(next_state2,[1,len(next_state2)])
            
            temp = np.array([[state,action,reward,state2],[state2,action2,reward2,state3]])
            self.past_exps.append(temp)
        
    def build_linear_NN(self):
        model = Sequential()

        model.add(Dense(60,activation='relu',input_shape = [self.input_size]))
        model.add(Dense(80,activation='relu'))
        model.add(Dense(60,activation='relu'))
        # model.add(Dense(100,activation='relu'))
        model.add(Dense(self.action_size,activation='linear')) ##this should be the size of the action space

        model.compile(loss='mse',optimizer=self.optimizer)

        return model

## TODO: add https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9448143
# this is deep recurrent NN [aug 2021 - JSAC    ]
## add https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9605672
# for the soft value function [jan 2022]

    def build_RNN(self):
        model = Sequential()
        
        model.add(LSTM(64,input_shape=(self.input_size,1),activation='tanh'))#activation='sigmoid'))
        model.add(Dense(60, activation='relu'))#,use_bias=True)
        model.add(Dense(80, activation='relu'))#,use_bias=True)
        model.add(Dense(60, activation='relu'))#,use_bias=True)
        model.add(Dense(self.action_size,activation='linear')) #linear activation == no activation
        
        model.compile(optimizer=self.optimizer,loss='mse')
        #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
        #,loss='mse')# , metrics=['accuracy'])

        return model
    
    def build_CNN(self):
        model = Sequential()
        
        model.add(Conv1D(filters=4,kernel_size=1,activation='relu',\
                         input_shape = self.input_size ))
        model.add(Conv1D(filters=16,kernel_size=1,activation='relu'))
        # model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())
        model.add(Dense(20,activation='relu'))
        model.add(Dense(self.action_size,activation='softmax'))
        
        model.compile(loss='categorical_crossentropy',optimizer=self.optimizer, metrics=['accuracy'])
        
        return model
    
    def align_target_model(self):
        self.target_network.set_weights(self.q_net.get_weights())

    ##########################################################################
    ## action
    def calc_action(self,state,args,ep_greed,\
        action_space,prev_action_ind=0,\
        min_md_pt_clusters=0,min_md_pt_recharge=0):
        
        ## with prob epsilon choose random action
        if args.greed_base == False:
            if np.random.rand() <= ep_greed:
                ## randomly select next device cluster + update the cluster holder            
                action_indexes = np.random.randint(0,self.action_size)
            else:
                ## choose action with highest q-value
                if args.linear == True:
                    state = np.reshape(state,[1,self.input_size])
                elif args.RNN == True:
                    state = np.reshape(state,[1,self.input_size,1])
                else:
                    state = np.reshape(state,[1,self.input_size[0],self.input_size[1]])
                
                q_values = self.q_net.predict(state)
                action_indexes = np.argmax(q_values[0])
        
            #print(q_values)
            
        # print('state')
        # print(state)
        # print('end state')
        # args.greed_base = True
        # args.greed_style = 1
        # greed_base = deepcopy(args.greed_base)
        if args.greed_base == True:
            if args.greed_style == 0:
                # first look at the state, find the current positions
                # then travel to the next position by index
                # pos_temp = np.where(np.array(state[-13:-3]).astype(int) == 0)
                pos_temp = action_space[prev_action_ind]
                pos_temp2 = list(pos_temp[1])
                pos_temp_new = np.zeros_like(pos_temp2)
                
                
                # look at energy, if energy falls below min_thresh (8440)
                # go recharge
                nrg_temp = state[-16:-13]
                for ind_nrg,nrg_temp_inst in enumerate(nrg_temp):
                    if nrg_temp_inst < 8440:
                        if 7 not in pos_temp_new:
                            pos_temp_new[ind_nrg] = 7
                            pos_temp2[ind_nrg] = 7
                        elif 8 not in pos_temp_new:
                            pos_temp_new[ind_nrg] = 8
                            pos_temp2[ind_nrg] = 8
                        else:
                            pos_temp_new[ind_nrg] = 9
                            pos_temp2[ind_nrg] = 9
                    else:
                        # increment by 1
                        pos_temp2[ind_nrg] += 1
                        # wrap around
                        if pos_temp2[ind_nrg] == 10:
                            pos_temp2[ind_nrg] = 0
                
                # choose temporal traits randomly
                # so, find all sets that match pos_temp2
                # ast = action space temp
                ast = np.array([list(atemp[1]) for atemp in action_space])
                as_ts = np.where((pos_temp2[0] == ast[:,0]) \
                    & (pos_temp2[1] == ast[:,1]) & (pos_temp2[2] == ast[:,2]))            
                
                # as_temp_search = np.where(bool_temp == True)
                #np.where(pos_temp2 == action_space_temp)
                
                # do a greed method for action determination
                if len(as_ts[0]) == 0:
                    action_indexes = as_ts[0]
                else:
                    action_indexes = as_ts[0][np.random.randint(0,len(as_ts[0]))]
                    
                # rev_action_ind + 1 
            elif args.greed_style == 1: #min distance
                pos_temp = action_space[prev_action_ind]
                pos_temp2 = list(pos_temp[1])
                pos_temp_new = np.zeros_like(pos_temp2)-1
                
                # look at energy, if energy falls below min_thresh (8440)
                # go recharge
                nrg_temp = state[-16:-13]
                for ind_nrg,nrg_temp_inst in enumerate(nrg_temp):
                    if nrg_temp_inst < 12660: #8440: #this needs margin
                        if min_md_pt_recharge[pos_temp2[ind_nrg]] \
                            not in pos_temp_new:
                            pos_temp_new[ind_nrg] = \
                                min_md_pt_recharge[pos_temp2[ind_nrg]]
                            pos_temp2[ind_nrg] = \
                                min_md_pt_recharge[pos_temp2[ind_nrg]]
                        else: #you take whatever is available
                            if 7 not in pos_temp_new:
                                pos_temp_new[ind_nrg] = 7
                                pos_temp2[ind_nrg] = 7
                            elif 8 not in pos_temp_new:
                                pos_temp_new[ind_nrg] = 8
                                pos_temp2[ind_nrg] = 8
                            else:
                                pos_temp_new[ind_nrg] = 9
                                pos_temp2[ind_nrg] = 9
                    else:
                        # travel to nearest distance cluster
                        if min_md_pt_clusters[pos_temp2[ind_nrg]] \
                            not in pos_temp_new:
                            pos_temp_new[ind_nrg] = \
                                min_md_pt_clusters[pos_temp2[ind_nrg]]
                            pos_temp2[ind_nrg] = \
                                min_md_pt_clusters[pos_temp2[ind_nrg]]
                        else: # you take a random one
                            tt_cluster = np.random.randint(0,8)
                            while tt_cluster in pos_temp_new:
                                tt_cluster = np.random.randint(0,8)
                            pos_temp_new[ind_nrg] = tt_cluster
                            pos_temp2[ind_nrg] = tt_cluster
                
                # choose temporal traits randomly
                # so, find all sets that match pos_temp2
                # ast = action space temp
                ast = np.array([list(atemp[1]) for atemp in action_space])
                as_ts = np.where((pos_temp2[0] == ast[:,0]) \
                    & (pos_temp2[1] == ast[:,1]) & (pos_temp2[2] == ast[:,2]))   
                
                if len(as_ts[0]) == 0:
                    action_indexes = as_ts[0]
                else:
                    action_indexes = as_ts[0][np.random.randint(0,len(as_ts[0]))]
            
            elif args.greed_style == 2: # rng minimum distance
                pos_temp = action_space[prev_action_ind]
                pos_temp2 = list(pos_temp[1])
                pos_temp_new = np.zeros_like(pos_temp2)-1
                
                # look at energy, if energy falls below min_thresh (8440)
                # go recharge
                nrg_temp = state[-16:-13]
                for ind_nrg,nrg_temp_inst in enumerate(nrg_temp):
                    if nrg_temp_inst < 12660: #8440: #this needs margin
                        if min_md_pt_recharge[pos_temp2[ind_nrg]] \
                            not in pos_temp_new:
                            pos_temp_new[ind_nrg] = \
                                min_md_pt_recharge[pos_temp2[ind_nrg]]
                            pos_temp2[ind_nrg] = \
                                min_md_pt_recharge[pos_temp2[ind_nrg]]
                        else: #you take whatever is available
                            if 7 not in pos_temp_new:
                                pos_temp_new[ind_nrg] = 7
                                pos_temp2[ind_nrg] = 7
                            elif 8 not in pos_temp_new:
                                pos_temp_new[ind_nrg] = 8
                                pos_temp2[ind_nrg] = 8
                            else:
                                pos_temp_new[ind_nrg] = 9
                                pos_temp2[ind_nrg] = 9
                    else:
                        # travel to nearest distance cluster
                        # scaled by a random factor
                        if np.random.rand() < args.rng_thresh:
                            if min_md_pt_clusters[pos_temp2[ind_nrg]] \
                                not in pos_temp_new:
                                pos_temp_new[ind_nrg] = \
                                    min_md_pt_clusters[pos_temp2[ind_nrg]]
                                pos_temp2[ind_nrg] = \
                                    min_md_pt_clusters[pos_temp2[ind_nrg]]
                            else: # you take a random one
                                tt_cluster = np.random.randint(0,8)
                                while tt_cluster in pos_temp_new:
                                    tt_cluster = np.random.randint(0,8)
                                pos_temp_new[ind_nrg] = tt_cluster
                                pos_temp2[ind_nrg] = tt_cluster
                        else: #decide at random - branching/heuristic min distance
                            tt_cluster = np.random.randint(0,8)
                            while tt_cluster in pos_temp_new:
                                tt_cluster = np.random.randint(0,8)
                            pos_temp_new[ind_nrg] = tt_cluster
                            pos_temp2[ind_nrg] = tt_cluster
                            
                # choose temporal traits randomly
                # so, find all sets that match pos_temp2
                # ast = action space temp
                ast = np.array([list(atemp[1]) for atemp in action_space])
                as_ts = np.where((pos_temp2[0] == ast[:,0]) \
                    & (pos_temp2[1] == ast[:,1]) & (pos_temp2[2] == ast[:,2]))   
                
                if len(as_ts[0]) == 0:
                    action_indexes = as_ts[0]
                else:
                    action_indexes = as_ts[0][np.random.randint(0,len(as_ts[0]))]
                
            else:
                raise Exception("That option isn't available")
        
        return action_indexes #which index the swarm should go to next
    
    ##########################################################################
    ## retraining procedure (experience replay)
    def retrain_exp_replay(self,batch_size,args=args):
        minibatch = random.sample(self.past_exps,batch_size)
        
        if args.linear == True or args.RNN == True:
        
            for state,action,reward,next_state in minibatch:
                
                if args.RNN == True: #linear net doesn't need to be reshaped
                    state = np.reshape(state,[1,self.input_size,1])                
                    next_state = np.reshape(next_state,[1,self.input_size,1])
                    
                target = self.q_net.predict(state)
                # print(target)
                # raise NameError('HiThere')
                
                terminated = 0
                
                if terminated:
                    target[0][action] = reward
                else:
                    t = self.target_network.predict(next_state)
                    target[0][action] = reward + self.g_discount * np.amax(t)
                    
                self.q_net.fit(state,target,epochs=1,verbose=0)

# %% confirmation testing
test_DQN = DQN(args)

test_DQN.q_net.summary()
test_DQN.target_network.summary()

# %% function to calculate rewards

def reward_state_calc(test_DQN,current_state,current_action,current_action_space,\
        cluster_expectations,cluster_limits,min_battery_levels,historical_results,\
        travel_energy,current_swarm_pos,args=args):
    
    ## calculate current_state + current_action expected gain  

    ## determine next state, also changes last visits
    ## so far, this only determines the next UAV positions
    next_state_set = list(current_action_space[current_action]) #deepcopy(current_state)
    # this (Ts, (pos1,2,3,4), taus1, taus2)
    
    ## pull energy and gradient data from historical_results
    ts_start = next_state_set[0]
    new_positions = next_state_set[1] #this is a tuple
    taus1_next = next_state_set[2]
    taus2_next = next_state_set[3]
    
    # each swarm must be calculated individually
    if ts_start == 180:
        hfilter1 = historical_results[:160]
    elif ts_start == 220:
        hfilter1 = historical_results[160:320]
    else:
        hfilter1 = historical_results[320:]
    # learning, objective, energy in that order
    
    historical_data = []
    for i,j in enumerate(new_positions):
        if j < 8: #10: #then its at a device cluster; hard coded cluster
            # filter by swarm, device cluser, taus1 and taus2
            # each swarm has 10*2*2 entries 
            swarm_data = hfilter1[i*40:(i+1)*40]
            cluster_data = swarm_data[j*4:(j+1)*4]
            taus1_data = cluster_data[(taus1_next-1) * 2 : taus1_next * 2]
            historical_data.append(taus1_data[ taus2_next-1] )    
    
    ## update last visits to calculate model drift
    # includes recharge stations
    # state = [uav positions, cluster visit times, recharge visit times, min battery levels]
    
    # next_state_visits = [i+1 for i in current_state[len(test_DQN.U):-len(test_DQN.U)]]
    nsv_index1 = len(test_DQN.C) + len(test_DQN.recharge_points)
    next_state_visits = [i+1 for i in current_state[ -nsv_index1 -3 : -3 ] ]
    # battery_status = [i for i in current_state[-len(test_DQN.U):]]
    batt_indx1 = -3-len(test_DQN.C)-len(test_DQN.recharge_points)-len(test_DQN.U)
    batt_indx2 = -3-len(test_DQN.C)-len(test_DQN.recharge_points)
    battery_status = [i for i in current_state[ batt_indx1 : batt_indx2 ]]
    
    ## determine cluster_bat_drain - this is based on historical data
    # filter historical data
    cluster_bat_drain = np.zeros(shape=len(test_DQN.U)).tolist()
    index_counter = 0
    for i,j in enumerate(cluster_bat_drain):
        if new_positions[i] < 8:
            cluster_bat_drain[i] = historical_data[index_counter][2][-1] 
            index_counter += 1
            #fixed indexing because 3rd is energy, and final is consumption
    
    # new way to calculate reward as per equation (58) of the paper
    C = 100000 #O(100) or O(1000) try both
    c1 = 0.2 #O(1)
    c2 = 0.25 #O(1)
    c3 = 0.005 #O(0.01)
    
    # determine the reward from the complementary geoemetric programming
    reward_vec = np.zeros(shape=len(test_DQN.U)).tolist()
    index_counter = 0
    for i,j in enumerate(reward_vec):
        if new_positions[i] < 8:
            ## this previous working result
            reward_vec[i] = c1*historical_data[index_counter][1][-1] #1000*10
            # this should be [index_counter][1][-1]
            index_counter += 1
        else:
            # reward_vec[i] = 100000 #because of division, this leads to a C/C\approx 1 reward
            # reward_vec[i] = 2e2
            # reward_vec[i] = 1e2
            reward_vec[i] = 1e3
            
    # calculate the energy movement costs - also update the visitations vector
    em_hold = 0 
    for i,j in enumerate(new_positions): #next_state_set 
        # travel cost
        battery_status[i] -= travel_energy[current_swarm_pos[i],j]
        em_hold += c3*travel_energy[current_swarm_pos[i],j]        
        ## filter for device cluster or recharge station
        if j < 8: #len(next_state_visits) - len(test_DQN.recharge_points): # it is a device cluster
            battery_status[i] -= cluster_bat_drain[i] #[j] # drain by cluster needs
        
        else: #it is a recharge station
            if args.cap == 'low':
                cap_level = 48600
            elif args.cap == 'medium':
                cap_level = 48600*2
            elif args.cap == 'high':
                cap_level = 48600*3
            battery_status[i] = cap_level
            # battery_status[i] = 48600#2000 #100 #reset to 100%

        #previously was cluster_expectations[j] * next_state_visits[j]
        next_state_visits[j] = 0 # zero out since now it will be visited    
    
    ## zero out recharging stations in next_state_visits 
    ## DEBUG flag here
    # next_state_visits[-1] = 0
    # next_state_visits[-2] = 0
    
    # determine G(s)
    gs_hold = 0
    for i,j in enumerate(next_state_visits):
        if i < len(next_state_visits) - len(test_DQN.recharge_points): # it is a device cluster    
            if j * cluster_expectations[i] > cluster_limits[i]:
                gs_hold += c2*cluster_limits[i]
            else:
                gs_hold += j*c2*cluster_expectations[i]
    
    # add the value together
    current_reward = C/(sum(reward_vec)+em_hold+gs_hold)
    ml_reward_only = C/(sum(reward_vec)+gs_hold)
    # print('check the reward calc')
    # print(sum(reward_vec))
    # print(em_hold)
    # print(gs_hold)
    
    # check for battery failures (cannot afford to lose any UAVs)
    # bat_penalty = 0
    penalty = 0
    for i,j in enumerate(battery_status):
        if j < min_battery_levels[i]: #0:
            if args.pen == 'high':
                penalty += 1000 #20000
            elif args.pen == 'medium':
                penalty += 100
            elif args.pen == 'low':
                penalty += 10
            current_reward = 0 #force zero out current reward if ANY battery runs out
            
    current_reward -= penalty
    
    ## if swarm was at a recharging station previously, and stays at one, incur a penalty
    # compare current_swarm_pos and new_positions
    # if args.brt == 'debug2':
    # for i,j in enumerate(current_swarm_pos):
    #     if j == 8 or j == 9 :
    #         if new_positions[i] == 8 or new_positions[i] == 9:
    #             current_reward -= 100 #1e5 #high penalty
    
    ## calculate the next state
    ## needs to be rebuilt
    active_swarms = np.zeros(len(test_DQN.U)).tolist()
    for i,j in enumerate(new_positions):
        if j < 8: #then at cluster
            active_swarms[i] = 1
    
    # prior to this line, next_state_set = Ts, (pos1,2,3), taus1, taus2
    next_state_set = []
    next_state_set += active_swarms
    # model drift
    next_state_set += (np.multiply(cluster_expectations,\
                        next_state_visits[:len(test_DQN.C)] )  ).tolist()
    next_state_set += battery_status # update battery status
    next_state_set += next_state_visits #update visits; active low

    next_state_set += [ts_start,taus1_next,taus2_next]
    
    # print(next_state_set)

    return current_reward, next_state_set, new_positions, ml_reward_only


# %% function to calculate action space

## calculate the full action space
## this is static
## based on itertools product
def action_space_calc(all_clusters_recharges,num_swarms=args.U_swarms,\
    Ts=[180,220,260],taus1=[1,2],taus2=[1,2]):
    
    ## calculate all permutations
    # action_space = [i for i in permutations(all_clusters_recharges,num_swarms)]
    swarm_pos_permutations = [i for i in permutations(all_clusters_recharges,num_swarms)]
    
    action_space = list(product(Ts,swarm_pos_permutations,taus1,taus2))
    
    return action_space

# %% load in historical optimizer results
cwd = os.getcwd()
with open(cwd+'/geo_optim_chars/results_all','rb') as f:
    historical_results = pk.load(f)

# historical results incremented from the bottom up
# T_s_vec,swarms_vec,clusters_vec,tau_s1_vec,tau_s2_vec

# 160 per Ts
optim_180 = historical_results[:160]
optim_220 = historical_results[160:320]
optim_260 = historical_results[320:]

# 4 swarms - each with an even split
optim_180_s0 = optim_180[:40] # swarm0; each cluster takes 4, taus2 takes 2, then taus1 remains


# %% building the sequence for DRL
episodes = 1
reward_storage = []
ml_reward_only_storage = []
battery_storage = []
state_save = []

if args.centralized == True:
    reward_DQN = np.zeros((1,1,args.G_timesteps))
else:
    print('only centralized is currently supported')

# static action space - as finite swarm movement choices
action_space = action_space_calc(list(range(args.Clusters + args.recharge_points)))

#cluster_expectations = 100*np.random.rand(args.Clusters) # the distribution change over time
# cluster_expectations = 100*np.array([0.005,1.6,0.8,3,0.3,0.02])
# cluster_limits = 100*np.array([1,2.2,1.3,5,1.1,2.1])

## do the time_drift_min and time_drift_max for dynamic model drift
if args.dynamic == True:
    time_drift_min = 2*np.array(range(1,args.Clusters+1)) 
    # *np.random.rand(args.Clusters) #linear function for all clusters
    time_drift_max = deepcopy(time_drift_min) + np.array([13,11,15,3,2,5,5,1])
    cluster_limits = 20*time_drift_max
    cluster_expectations = deepcopy(time_drift_min)
else:
    ## static model drift
    ## reset seed in numpy random
    np.random.seed(args.seed)
    cluster_expectations = 25*np.random.rand(args.Clusters) #20
    cluster_limits = 20*cluster_expectations #3 - vary epsilon + gamma use this one
    # cluster_limits = 2*cluster_expectations


# calculate real movement costs from cluster to cluster to recharge station
min_dist = 500 #meters
max_dist = 1000 #1km 2km
# speed = 5000 #5km/hr
scaling = 0.1

min_speed_uav = 10 #m/s -> 0.5/1000 * 60 * 60 km/h
seconds_conversion = 2 #5
air_density = 1.225 
zero_lift_drag_max = 0.0378 #based on sopwith camel 
zero_lift_breakpoint = 0.0269
zero_lift_drag_min = 0.0161 #based on p-51 mustang
wing_area_max = 3 #m^2
wing_area_breakpoint = 1.75
wing_area_min = 0.5
oswald_eff = 0.8 #0.7 to 0.85
aspect_ratio = 4.5 #2 to 7 
weight_max = 10 #kg
weight_breakpoint = 5.05
weight_min = 0.1 #100g
kg_newton = 9.8 


c1 = 0.5 * air_density * (zero_lift_breakpoint +\
    (zero_lift_drag_max-zero_lift_breakpoint)*np.random.rand()) * \
    (wing_area_breakpoint + (wing_area_max - wing_area_breakpoint)*np.random.rand())

c2 = 2 * (weight_breakpoint + (weight_max - weight_breakpoint)*np.random.rand())**2 \
    / (np.pi * oswald_eff * (wing_area_breakpoint + \
    (wing_area_max - wing_area_breakpoint)*np.random.rand()) * air_density**3 )   

move_dists = min_dist + (max_dist-min_dist)*\
    np.random.rand(args.Clusters+args.recharge_points,args.Clusters+args.recharge_points)


## calc move_dist min indexes
# [index0 contains the min next move of any swarm at cluster 0]
min_md_pt = [] #min move dist points
min_md_pt_clusters = []
min_md_pt_recharge = []
for ind,move_vec in enumerate(move_dists):
    t_move_vec = deepcopy(move_vec)
    t_move_vec[ind] = 10000
    
    min_md_pt.append(np.argmin(t_move_vec))
    min_md_pt_clusters.append(np.argmin(t_move_vec[:7]))
    min_md_pt_recharge.append(7+np.argmin(t_move_vec[7:]))

#scaling * 
temp_energy = move_dists/min_speed_uav * (c1 * (min_speed_uav**3) + c2/min_speed_uav)

## return to prev
# cluster_bat_drain = np.array([3,5,5,6,2,1])
if args.cap == 'low':
    cap_level = 48600
elif args.cap == 'medium':
    cap_level = 48600*2
elif args.cap == 'high':
    cap_level = 48600*3

init_battery_levels = (cap_level* np.ones(args.U_swarms)).tolist()  #(2000* np.ones(args.U_swarms)).tolist() #70600
max_battery_levels = deepcopy(init_battery_levels)

if args.brt == 'medium':
    brt = 8440#*np.ones(args.U_swarms)).tolist() # initialize full battery
elif args.brt == 'high':
    brt = 8440*2 #16880
elif args.brt == 'low':
    brt = 4220
elif args.brt == 'hlow':
    brt = 4220*3
elif args.brt == 'vlow':
    brt = 2110
elif args.brt == 'vvlow':
    brt = 1055
elif args.brt == 'vhigh':
    brt = 8440*3
elif args.brt == 'vhigh2':
    brt = 8440*5
elif args.brt == 'vhigh3':
    brt = 8440*4
elif args.brt == 'debug' or args.brt == 'debug2':
    brt = -1e8

min_battery_levels = (brt*np.ones(args.U_swarms)).tolist()

Ts = [180,220,260]
taus1 = [1,2]
taus2 = [1,2]
## TODO: swarm battery plots
# saving some plots for debugging
fig_no = 0
cwd = os.getcwd()
state_set_all = []

freq_visits = {e:np.zeros(args.Clusters+args.recharge_points) for e in range(args.G_timesteps)}

## full loop beginning
for e in range(episodes):
    ## calculate and build a set of states for the UAV swarms
    
    ## randomly initialize state
    ## state_set indexed by swarm [swarm0 pos, swarm1 pos, etc., cluster 0 last visit, ...]
    init_state_set = [] # swarm boolean, concept drift, min uav energy, grad of all clusters,
    # position of all swarms - cluster + recharge boolean, Ts, taus1, taus2
    # concept drift absorbs previous visit boolean
    
    init_state_set += np.ones(args.U_swarms).tolist() # all active    
    
    init_last_visit = np.ones(args.Clusters) # building the last visit structure
    
    temp_C_set = list(range(args.Clusters))
    init_swarm_pos = [ ]
    for i,j in enumerate(test_DQN.U):
        init_rng_index = np.random.randint(0,len(temp_C_set))
        # init_state_set.append(temp_C_set[init_rng_index]) #populates swarm pos
        init_last_visit[temp_C_set[init_rng_index]] = 0
        
        init_swarm_pos.append(init_rng_index)
        
        del temp_C_set[init_rng_index]
    
    # print(init_last_visit)
    # determine drift based on init_last_visit
    if args.dynamic == True:
        cluster_expectations = deepcopy(time_drift_min) #initially it will be minimum
    
    init_state_set += (np.multiply(cluster_expectations,init_last_visit)).tolist()
    init_state_set += init_battery_levels
    # init_state_set.append( 250*args.Clusters )
    
    # current position - boolean flipped - active low
    init_state_set += init_last_visit.tolist() + np.ones(args.recharge_points).tolist()
    
    init_state_set += [260,2,2] #Ts, taus1, taus2
    
    # init_state_set += [int(i) for i in init_last_visit]

    # # include recharge stations in the init_state_set
    # init_state_set += np.ones(args.recharge_points).tolist() #[1,1] #as we don't visit the recharge stations initially
    # init_state_set += init_battery_levels #min_battery_levels #add in the battery levels
    
    ## iterate over the timesteps
    for timestep in range(args.G_timesteps):
        if args.dynamic == True:
            cluster_expectations = time_drift_min + \
                (time_drift_max-time_drift_min)*10*(timestep+1)/(args.G_timesteps)
            
        # calculate the reward
        if args.linear == True or args.RNN == True:
            ep_greed = np.max([args.ep_min, args.ep_greed*(1-10**(-3))**timestep])
            
            if timestep == 0:
                for freq_val in np.where(np.array(init_state_set[-13:-3]) == 0.0)[0]:
                    freq_visits[timestep][freq_val] = 1 #np.array(init_state_set[-13:-3])
                
                init_state_set = np.round(np.array(init_state_set).astype(float),4)
                
                action_set = test_DQN.calc_action(state=init_state_set, \
                        args=args,ep_greed =ep_greed,action_space=action_space,\
                        min_md_pt_clusters=min_md_pt_clusters,\
                        min_md_pt_recharge=min_md_pt_recharge)
                # map action_set to a movement
                
                # rewards, state_set = reward_state_calc(test_DQN,init_state_set,action_set,\
                #                 action_space,cluster_expectations,cluster_limits,\
                #                     cluster_bat_drain)
                rewards, state_set,next_swarm_pos, temp_ml_reward \
                    = reward_state_calc(test_DQN,init_state_set,\
                    action_set,action_space,cluster_expectations,cluster_limits,\
                    min_battery_levels,historical_results,temp_energy, init_swarm_pos)
                # cluster exepectatiosn + cluster limits are the model drift factors
                
                ## store experiences
                test_DQN.store(init_state_set,action_set,rewards,state_set)
                
            else:
                current_state_set = deepcopy(state_set)
                
                #general base start
                freq_visits[timestep] = deepcopy(freq_visits[timestep-1])
                for freq_val in next_swarm_pos:
                    freq_visits[timestep][freq_val] += 1
                # for freq_val in np.where(np.array(current_state_set[-13:-3]) == 0.0)[0]:
                #     freq_visits[timestep][freq_val] = \
                #         freq_visits[timestep-1][freq_val] + 1
                
                # # freq_visits[timestep] = freq_visits[timestep-1] + \
                # #     np.array(current_state_set[-13:-3])
                
                prev_action_set = deepcopy(action_set)
                
                current_state_set = np.round(np.array(current_state_set).astype(float),4)
                # current_state_set = np.reshape(current_state_set, \
                #         (current_state_set.shape[0], 1))
                
                action_set = test_DQN.calc_action(state=current_state_set, \
                    args=args,ep_greed =ep_greed,action_space=action_space, \
                    prev_action_ind=prev_action_set,\
                    min_md_pt_clusters=min_md_pt_clusters,\
                    min_md_pt_recharge=min_md_pt_recharge)
                
                rewards, state_set,next_swarm_pos, temp_ml_reward \
                    = reward_state_calc(test_DQN,current_state_set,\
                    action_set,action_space,cluster_expectations,cluster_limits,\
                    min_battery_levels,historical_results,temp_energy,next_swarm_pos) # cluster_bat_drain)
                
                ## store experiences
                test_DQN.store(current_state_set,action_set,rewards,state_set)
                
                # print(current_state_set)
                # print(state_set)
                # print(next_swarm_pos)                
                
            reward_DQN[0,0,timestep] = rewards
            
        else:
            print('wrong training setting')

        # print('freq_visits:')
        # print(freq_visits[timestep])
        # if timestep%10 == 0:
        #     input('a')
        
        ## printing check up
        if timestep % 10 == 0:
            #print(state_set)
            if args.linear == True or args.RNN == True:
                print('timestep='+str(timestep),
                      'reward_DQN ={:.2f}'.format(np.sum(reward_DQN,axis=0)[e][timestep]),
                      'reward ML only = {:.2f}'.format(temp_ml_reward),
                      'epsilon = {:.2f}'.format(args.ep_greed)
                      )

            # print(test_DQN.q_net.get_weights())


        # experience update -> this is the actual training
        if len(test_DQN.past_exps) > args.replay_bs:
            #print('tomato')
            test_DQN.retrain_exp_replay(args.replay_bs)
            
        # update label generation from taget model
        if timestep != 0 and timestep%args.DQN_update_period==0:
            test_DQN.align_target_model()
        
        #print(test_DQN.past_exps)
        
        # save all of the states
        
        if args.linear == True or args.RNN == True:
            state_save.append(state_set)
            
            # reward save for plots
            reward_storage.append(rewards)
            
            ml_reward_only_storage.append(temp_ml_reward)
            # battery save for plots
            # print(state_set[-args.U_swarms:])
            # battery_storage.append(state_set[-args.U_swarms:])
            state_set_bat2 = -args.Clusters - args.recharge_points -3
            state_set_bat1 = -args.U_swarms -args.Clusters - args.recharge_points -3
            
            # print(state_set[ state_set_bat1: state_set_bat2 ])
            
            battery_storage.append(state_set[ state_set_bat1 : state_set_bat2 ])
            
        # state_set_all.append(state_set)
        
        
        if timestep % 100 == 0:            
            print('saving results into file')
            # print(test_DQN.q_net.get_weights()[-1])
            
            # plt.figure(fig_no) # plot reward change over time - move this to separate file later
            
            # plt.plot(reward_storage)
            # plt.xlabel('time instance')
            # plt.ylabel('reward')
            # plt.title('reward over time')    
            
            # # save image
            # plt.savefig(cwd+'/plots/'+str(fig_no)+'_'+str(args.ep_greed)+'_'+'linear.png')
            
            # plt.clf()
            
            # TODO : not really a todo, just a quick scroller
            try:
                os.mkdir(cwd+'/drl_results')
                print('save instance')
            except:
                print('save instance')
            
            
            if args.greed_base == True:
                if args.greed_style == 2:
                    # save data
                    with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(args.ep_greed)+'_'+'reward'\
                              +'test_large'+'_'+str(args.g_discount)+'_greedy_'+\
                            str(args.greed_style)+'_rng_thresh_'+ str(args.rng_thresh)+\
                            '_'+args.brt,\
                            'wb') as f:
                        pk.dump(reward_storage,f)
                    # ml reward only data
                    with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(args.ep_greed)+'_'+\
                              'ml_reward_only'+'test_large'+'_'+str(args.g_discount)+'_greedy_'+\
                            str(args.greed_style)+'_rng_thresh_'+ str(args.rng_thresh)+\
                            '_'+args.brt,\
                            'wb') as f:
                        pk.dump(ml_reward_only_storage,f)                        
                    #+'_extra'
                    #str(fig_no)+
                    with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(args.ep_greed)+'_'+'battery'\
                              +'test_large'+'_'+str(args.g_discount)+'_greedy_'+\
                            str(args.greed_style)+'_rng_thresh_'+ str(args.rng_thresh)+\
                            '_'+args.brt,\
                            'wb') as f:
                        pk.dump(battery_storage,f)
                    #str(fig_no)+
                    with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(args.ep_greed)+'_'+'all_states'\
                              +'test_large'+'_'+str(args.g_discount)+'_greedy_'+\
                            str(args.greed_style)+'_rng_thresh_'+ str(args.rng_thresh)+\
                            '_'+args.brt,\
                            'wb') as f:
                        pk.dump(state_save,f)
                    
                    with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(args.ep_greed)+'_'+'visit_freq_large'+\
                              '_'+str(args.g_discount)+'_greedy_'+\
                            str(args.greed_style)+'_rng_thresh_'+ str(args.rng_thresh)+\
                            '_'+args.brt,\
                            'wb') as f:
                        pk.dump(freq_visits,f)
                    
                else:
                    # save data
                    with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(args.ep_greed)+'_'+'reward'\
                              +'test_large'+'_'+str(args.g_discount)+'_greedy_'+\
                            str(args.greed_style)+\
                            '_'+args.brt,'wb') as f:
                        pk.dump(reward_storage,f)
                    with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(args.ep_greed)+'_'+\
                              'ml_reward_only'+'test_large'+'_'+str(args.g_discount)+'_greedy_'+\
                            str(args.greed_style)+\
                            '_'+args.brt,'wb') as f:
                        pk.dump(ml_reward_only_storage,f)                        
                    #+'_extra'
                    #str(fig_no)+
                    with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(args.ep_greed)+'_'+'battery'\
                              +'test_large'+'_'+str(args.g_discount)+'_greedy_'+\
                            str(args.greed_style)+\
                            '_'+args.brt,'wb') as f:
                        pk.dump(battery_storage,f)
                    #str(fig_no)+
                    with open(cwd+'/drl_results/seed_'+str(seed)+'_'+str(args.ep_greed)+'_'+'all_states'\
                              +'test_large'+'_'+str(args.g_discount)+'_greedy_'+\
                            str(args.greed_style)+\
                            '_'+args.brt,'wb') as f:
                        pk.dump(state_save,f)
                    
                    with open(cwd+'/drl_results/seed_'+str(seed)+str(args.ep_greed)+'_'+'visit_freq_large'+\
                              '_'+str(args.g_discount)+'_greedy_'+\
                            str(args.greed_style)+\
                            '_'+args.brt,'wb') as f:
                        pk.dump(freq_visits,f)
            
            else:
                if args.dynamic == True:
                    # save data
                    with open(cwd+'/drl_results/RNN/seed_'+str(seed)+'_'\
                              +str(args.ep_greed)+'_'+'reward'\
                              +'test_large'+'_'+str(args.g_discount)\
                            +'_tanh_mse_dynamic'+\
                            '_'+args.brt,'wb') as f:
                        pk.dump(reward_storage,f)
                    #+'_extra'
                    #str(fig_no)+
                    with open(cwd+'/drl_results/RNN/seed_'+str(seed)+'_'\
                              +str(args.ep_greed)+'_'+'battery'\
                              +'test_large'+'_'+str(args.g_discount)\
                            +'_tanh_mse_dynamic'+\
                            '_'+args.brt,'wb') as f:
                        pk.dump(battery_storage,f)
                    #str(fig_no)+
                    with open(cwd+'/drl_results/RNN/seed_'+str(seed)+'_'\
                              +str(args.ep_greed)+'_'+'all_states'\
                              +'test_large'+'_'+str(args.g_discount)\
                            +'_tanh_mse_dynamic'+\
                            '_'+args.brt,'wb') as f:
                        pk.dump(state_save,f)
                    
                    with open(cwd+'/drl_results/RNN/seed_'+str(seed)+'_'\
                              +str(args.ep_greed)+'_'+'visit_freq_large'+\
                              '_'+str(args.g_discount)\
                            +'_tanh_mse_dynamic'+\
                            '_'+args.brt,'wb') as f:
                        pk.dump(freq_visits,f)
                else:
                    # save data
                    # tfolder = 'cap_'+args.cap
                    # tfolder = 'pen_'+args.pen
                    #'+tfolder+'/
                    with open(cwd+'/drl_results/RNN/1e3seed_'+str(seed)+'_'\
                              +str(args.ep_greed)+'_'+'reward'\
                              +'test_large'+'_'+str(args.g_discount)\
                            +'_tanh_mse'+\
                            '_'+args.brt,'wb') as f:
                        pk.dump(reward_storage,f)
                    with open(cwd+'/drl_results/RNN/1e3seed_'+str(seed)+'_'\
                              +str(args.ep_greed)+'_'+\
                              'ml_reward_only'+'test_large'+'_'+str(args.g_discount)\
                            +'_tanh_mse'+\
                            '_'+args.brt,'wb') as f:
                        pk.dump(ml_reward_only_storage,f)                        
                    #+'_extra'
                    #str(fig_no)+
                    with open(cwd+'/drl_results/RNN/1e3seed_'+str(seed)+'_'\
                              +str(args.ep_greed)+'_'+'battery'\
                              +'test_large'+'_'+str(args.g_discount)\
                            +'_tanh_mse'+\
                            '_'+args.brt,'wb') as f:
                        pk.dump(battery_storage,f)
                    #str(fig_no)+
                    with open(cwd+'/drl_results/RNN/1e3seed_'+str(seed)+'_'\
                              +str(args.ep_greed)+'_'+'all_states'\
                              +'test_large'+'_'+str(args.g_discount)\
                            +'_tanh_mse'+\
                            '_'+args.brt,'wb') as f:
                        pk.dump(state_save,f)
                    
                    with open(cwd+'/drl_results/RNN/1e3seed_'+str(seed)+'_'\
                              +str(args.ep_greed)+'_'+'visit_freq_large'+\
                              '_'+str(args.g_discount)\
                            +'_tanh_mse'+\
                            '_'+args.brt,'wb') as f:
                        pk.dump(freq_visits,f)
                    



