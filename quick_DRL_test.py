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
    Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import time

np.random.seed(1)
random.seed(1)
rng = np.random.default_rng(seed=1)

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
parser.add_argument('--ep_greed',type=float,default=0.6,\
                    help='epsilon greedy val')
parser.add_argument('--ep_min',type=float,default=0.005,\
                    help='epsilon minimum')
parser.add_argument('--g_discount',type=float,default=0.7,\
                    help='gamma discount factor')
parser.add_argument('--replay_bs',type=int,default=20,\
                    help='experience replay batch size')
parser.add_argument('--linear',type=bool,default=True,\
                    help='MLP or CNN') ## argpase cannot evaluate booleans OOB - fix later
parser.add_argument('--cnn_range',type=int, default=2,\
                    help='conv1d range')

# ovr parameters
parser.add_argument('--G_timesteps',type=int,default=200,\
                    help='number of swarm movements') #
parser.add_argument('--training',type=int,default=1,\
                    help='training or testing the DRL')
parser.add_argument('--centralized',type=bool,default=True,\
                    help='centralized or decentralized')
parser.add_argument('--DQN_update_period',type=int,default=50,\
                    help='DQN update period')
#8,3,2,4,2,2

args = parser.parse_args()

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
        
        if args.linear == True:
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
        # model.add(Dense(60,activation='relu'))
        # model.add(Dense(100,activation='relu'))
        model.add(Dense(self.action_size,activation='linear')) ##this should be the size of the action space

        model.compile(loss='mse',optimizer=self.optimizer)

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
    def calc_action(self,state,args,ep_greed):
        
        ## with prob epsilon choose random action
        if np.random.rand() <= ep_greed:
            ## randomly select next device cluster + update the cluster holder            
            action_indexes = np.random.randint(0,self.action_size)
        else:
            ## choose action with highest q-value
            
            if args.linear == True:
                state = np.reshape(state,[1,self.input_size])
            else:
                state = np.reshape(state,[1,self.input_size[0],self.input_size[1]])
            
            q_values = self.q_net.predict(state)
            action_indexes = np.argmax(q_values[0])
        
            #print(q_values)
            
        # print('state')
        # print(state)
        # print('end state')
        
        return action_indexes #which index the swarm should go to next
    
    ##########################################################################
    ## retraining procedure (experience replay)
    def retrain_exp_replay(self,batch_size,args=args):
        minibatch = random.sample(self.past_exps,batch_size)
        
        if args.linear == True:
        
            for state,action,reward,next_state in minibatch:
                
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
        
        else: #follows the format of linear
            
            for item in minibatch:
                # print('item printing')
                # print(item)
                # print(item[-1])
                
                # raise NameError('Int')
                
                ## bugged
                if type(item[0][0]) == np.ndarray:
                    item[0][0] = item[0][0].tolist()
                
                if type(item[0][-1]) == np.ndarray:
                    item[0][-1] = item[0][-1].tolist()
                    
                state = [item[0][0], item[0][-1][0] ]
                # print('state check')
                # print(state)
                
                state = np.reshape(state,[1,self.input_size[0],self.input_size[1]])
                
                next_state = [item[-1][0],item[-1][-1]]
                # print('next state check')
                # print(next_state)
                
                next_state = np.reshape(next_state,[1,self.input_size[0],self.input_size[1]])
                
                target = self.q_net.predict(state) #item is [args.cnn_range, args.U_swarms + args.Clusters]
                
                ## terminated training later
                
                t = self.target_network.predict(next_state)
                target[0][item[-1][1]] = item[-1][-2] + self.g_discount * np.amax(t)
                
                self.q_net.fit(state,target,epochs=1,verbose=0)
                

# %% confirmation testing
test_DQN = DQN(args)

test_DQN.q_net.summary()
test_DQN.target_network.summary()

# %% function to calculate rewards

def reward_state_calc(test_DQN,current_state,current_action,current_action_space,\
        cluster_expectations,cluster_limits,min_battery_levels,historical_results):
    
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
    
    ## build reward vector/matrix
    reward_vec = np.zeros(shape=len(test_DQN.U)).tolist()
    for i,j in enumerate(reward_vec):
        if new_positions[i] < 8:
            reward_vec[i] = 1000*5/ historical_data[i][0][-1]
    
    current_reward = 0
    for i,j in enumerate(new_positions): #next_state_set 
        # print(i,j)
        
        ## filter for device cluster or recharge station
        if j < 8: #len(next_state_visits) - len(test_DQN.recharge_points): # it is a device cluster
            battery_status[i] -= cluster_bat_drain[i] #[j] # drain by cluster needs
            
            ## reward function calculated based on elapsed time x cluster factor
            if battery_status[i] > min_battery_levels[i] :#0: #min thresh
                
                # model drift is now just a penalty term 
                # if cluster_expectations[j]*next_state_visits[j] < cluster_limits[j]:
                #     current_reward += cluster_expectations[j]*next_state_visits[j]
                # else:
                #     current_reward += cluster_limits[j]
                
                current_reward += reward_vec[i] #from gradient
                
        else: #it is a recharge station
            battery_status[i] = 1000 #100 #reset to 100%

        #previously was cluster_expectations[j] * next_state_visits[j]
        next_state_visits[j] = 0 # zero out since now it will be visited
    
    
    ## calculate penalty for not visiting certain nodes (25% of their nominal value)
    penalty = 0
    
    for i,j in enumerate(next_state_visits):
        if i < len(next_state_visits) - len(test_DQN.recharge_points): # it is a device cluster
            
            # if j * 0.25 * cluster_expectations[i] > 0.5 * cluster_limits[i]:
            #     penalty += 0.5* cluster_limits[i]
            # else:
            #     penalty += j * 0.25 * cluster_expectations[i]
    
            if j * 0.5* cluster_expectations[i] > 0.5*cluster_limits[i]:
                penalty += cluster_limits[i]
            else:
                penalty += j * 0.5* cluster_expectations[i]    
    
    # check for battery failures (cannot afford to lose any UAVs)
    for i,j in enumerate(battery_status):
        if j < min_battery_levels[i]: #0:
            penalty += 1000 #20000
    
    current_reward -= penalty
    
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
    
    print(next_state_set)

    return current_reward, next_state_set


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
cluster_expectations = 20*np.random.rand(args.Clusters)
cluster_limits = 3*cluster_expectations


cluster_bat_drain = np.array([3,5,5,6,2,1])
init_battery_levels = (1000* np.ones(args.U_swarms)).tolist() #70600
max_battery_levels = deepcopy(init_battery_levels)

min_battery_levels = (360*np.ones(args.U_swarms)).tolist() # initialize full battery

Ts = [180,220,260]
taus1 = [1,2]
taus2 = [1,2]
## TODO: swarm battery plots

# saving some plots for debugging
fig_no = 0
cwd = os.getcwd()
state_set_all = []

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
    for i,j in enumerate(test_DQN.U):
        init_rng_index = np.random.randint(0,len(temp_C_set))
        # init_state_set.append(temp_C_set[init_rng_index]) #populates swarm pos
        init_last_visit[temp_C_set[init_rng_index]] = 0
        
        del temp_C_set[init_rng_index]
    
    # print(init_last_visit)
    # determine drift based on init_last_visit
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
        
        # calculate the reward
        if args.linear == True:
            ep_greed = np.max([args.ep_min, args.ep_greed*(1-10**(-3))**timestep])
            
            if timestep == 0:
                action_set = test_DQN.calc_action(state=init_state_set, \
                                                  args=args,ep_greed =ep_greed)
                # map action_set to a movement
                
                # rewards, state_set = reward_state_calc(test_DQN,init_state_set,action_set,\
                #                 action_space,cluster_expectations,cluster_limits,\
                #                     cluster_bat_drain)
                rewards, state_set = reward_state_calc(test_DQN,init_state_set,action_set,\
                    action_space,cluster_expectations,cluster_limits,\
                    min_battery_levels,historical_results)
                # cluster exepectatiosn + cluster limits are the model drift factors
                
                    
                    
                ## store experiences
                test_DQN.store(init_state_set,action_set,rewards,state_set)
                
            else:
                current_state_set = deepcopy(state_set)
                
                action_set = test_DQN.calc_action(state=current_state_set, \
                                                  args=args,ep_greed =ep_greed)
                
                rewards, state_set = reward_state_calc(test_DQN,current_state_set,action_set,\
                    action_space,cluster_expectations,cluster_limits,\
                    min_battery_levels,historical_results) # cluster_bat_drain)
                
                ## store experiences
                test_DQN.store(current_state_set,action_set,rewards,state_set)
            
            reward_DQN[0,0,timestep] = rewards
            
        else:
            if timestep != 0 and timestep % 2 == 0:
                ep_greed1 = np.max([args.ep_min, args.ep_greed*(1-10**(-3))**(timestep-1)])
                ep_greed2 = np.max([args.ep_min, args.ep_greed*(1-10**(-3))**timestep])
                
                if timestep == 2:
                    # action_set = test_DQN.calc_action(state=init_state_set, \
                                                      # args=args,ep_greed =ep_greed1)
                    # random initial actions
                    action_set = np.random.randint(0,len(action_space))
                    
                    reward1, state_set1 = reward_state_calc(test_DQN,init_state_set,\
                                        action_set, action_space,cluster_expectations)
                    
                    current_state_set = deepcopy(state_set1)
                    
                    # action_set2 = test_DQN.calc_action(state=current_state_set, \
                                                       # args=args,ep_greed = ep_greed2)
                                                       
                    action_set2 = np.random.randint(0,len(action_space))
                    
                    reward2, state_set2 = reward_state_calc(test_DQN,state_set1,\
                                        action_set2, action_space, cluster_expectations)
                        
                    test_DQN.store(init_state_set,action_set,reward1,state_set1,\
                                   action2=action_set2, reward2=reward2, next_state2 = state_set2)
                        
                else:
                    # print(timestep)
                    # print('state_set1 check')
                    # print(state_set1)
                    # print('state_set1 end')
                    prev_state_set = deepcopy(state_set2)                    
                    
                    state_set = [state_set1,state_set2] #needs the previous two instances
                    
                    action_set = test_DQN.calc_action(state=state_set, args=args, \
                                            ep_greed = ep_greed1)
                    
                    reward1, state_set1 = reward_state_calc(test_DQN,state_set2,\
                                        action_set, action_space,cluster_expectations)
                
                    state_set = [state_set2,state_set1] #temporal remap
                    
                    action_set2 = test_DQN.calc_action(state=state_set, \
                                                       args=args,ep_greed = ep_greed2)
                    
                    reward2, state_set2 = reward_state_calc(test_DQN,state_set1,\
                                        action_set2, action_space, cluster_expectations)
                    
                    test_DQN.store(prev_state_set,action_set,reward1,state_set1,\
                                   action2=action_set2, reward2=reward2, next_state2 = state_set2)
                
        
        ## printing check up
        if timestep % 10 == 0:
            #print(state_set)
            if args.linear == True:
                print('timestep='+str(timestep),
                      'reward_DQN ={:.2f}'.format(np.sum(reward_DQN,axis=0)[e][timestep]),
                      'epsilon = {:.2f}'.format(ep_greed)
                      )
            else:
                if timestep != 0 and timestep % 2 == 0:
                    print('timestep='+str(timestep),
                          'reward_DQN1 ={:.2f}'.format(reward1),
                          'reward_DQN2 ={:.2f}'.format(reward2),
                          'epsilon = {:.2f}'.format(ep_greed2)
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
        
        if args.linear == True:
            state_save.append(state_set)
            
            # reward save for plots
            reward_storage.append(rewards)
            
            # battery save for plots
            # print(state_set[-args.U_swarms:])
            # battery_storage.append(state_set[-args.U_swarms:])
            state_set_bat2 = -args.Clusters - args.recharge_points -3
            state_set_bat1 = -args.U_swarms -args.Clusters - args.recharge_points -3
            
            print(state_set[ state_set_bat1: state_set_bat2 ])
            
            battery_storage.append(state_set[ state_set_bat1 : state_set_bat2 ])
            
        else: 
            if timestep != 0 and timestep % 2 == 0:
                reward_storage.append(reward1)
                reward_storage.append(reward2)
        # state_set_all.append(state_set)
        
        
        if timestep % 100 == 0:
            print(test_DQN.q_net.get_weights()[-1])
            
            plt.figure(fig_no) # plot reward change over time - move this to separate file later
            
            plt.plot(reward_storage)
            plt.xlabel('time instance')
            plt.ylabel('reward')
            plt.title('reward over time')    
            
            # save image
            plt.savefig(cwd+'/plots/'+str(fig_no)+'_'+str(args.ep_greed)+'_'+'linear.png')
            
            plt.clf()
            
            # save data
            with open(cwd+'/data/'+str(fig_no)+'_'+str(args.ep_greed)+'_'+'reward'\
                      +'test','wb') as f:
                pk.dump(reward_storage,f)
            
            with open(cwd+'/data/'+str(fig_no)+'_'+str(args.ep_greed)+'_'+'battery'\
                      +'test','wb') as f:
                pk.dump(battery_storage,f)
            
            with open(cwd+'/data/'+str(fig_no)+'_'+str(args.ep_greed)+'_'+'all_states'\
                      +'test','wb') as f:
                pk.dump(state_save,f)
                
            # with open(cwd+'/data/'+str(fig_no)+'_30_epsilon_10000_lr_small_states','wb') as f:
            #     pk.dump(state_set_all,f)
            print(time.time()-start)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            









