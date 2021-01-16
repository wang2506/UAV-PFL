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
from itertools import combinations, permutations
from math import factorial
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Flatten, \
    Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

np.random.seed(1)
random.seed(1)
rng = np.random.default_rng(seed=1)

# %% parser function
parser = argparse.ArgumentParser()

## UAV and device cluster setups
parser.add_argument('--U_swarms',type=int,default=2,\
                    help='number of UAV swarms')
parser.add_argument('--Clusters',type=int,default=6,\
                    help='number of device clusters')
parser.add_argument('--total_UAVs',type=int,default=6,\
                    help='aggregate number of UAVs')
parser.add_argument('--UAV_init_ratio',type=float,default=0.8,\
                    help='initial worker to coorindator ratio')

## RL probs (epsilon and gamma)
parser.add_argument('--ep_greed',type=float,default=0.6,\
                    help='epsilon greedy val')
parser.add_argument('--ep_min',type=float,default=0.005,\
                    help='epsilon minimum')
parser.add_argument('--g_discount',type=float,default=0.7,\
                    help='gamma discount factor')
parser.add_argument('--replay_bs',type=int,default=10,\
                    help='experience replay batch size')
parser.add_argument('--linear',type=bool,default=False,\
                    help='MLP or CNN')
parser.add_argument('--cnn_range',type=int, default=2,\
                    help='conv1d range')
    
# ovr parameters
parser.add_argument('--G_timesteps',type=int,default=10000,\
                    help='number of swarm movements')
parser.add_argument('--training',type=int,default=1,\
                    help='training or testing the DRL')
parser.add_argument('--centralized',type=bool,default=True,\
                    help='centralized or decentralized')
parser.add_argument('--DQN_update_period',type=int,default=50,\
                    help='DQN update period')

args = parser.parse_args()

# %% DQN object 

class DQN:
    def __init__(self,args,optimizer=Adam(learning_rate=0.001)):
        # inits
        self.U = np.arange(0,args.U_swarms,1) #the swarms as an array 
        self.C = np.random.randint(0,10,size=args.Clusters) # measures their difficulty (i.e., need more revisits)
        self.total_UAVs = args.total_UAVs
        
        self.total_time = args.G_timesteps
        ## (swarm 0 position, swarm 0 min energy),... , (latest cluster visit times), ...
        
        self.action_size = factorial(args.Clusters)/factorial(args.Clusters-args.U_swarms) # all possible permutations
        
        #self.ep_greed = args.ep_greed
        self.g_discount = args.g_discount
        self.past_exps = deque(maxlen=300)

        self.optimizer = optimizer
        
        # RL Q-function nets
        if args.linear == True:
            self.input_size = args.U_swarms + args.Clusters 
            
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
        ## calculate the initial UAV swarm allocation
        ## if there cannot be an equal allocation (i.e., all swarms init the same size)
        ## then, run an incremental loop until all swarms populated
        ups_holder = int(self.total_UAVs/args.U_swarms) * np.ones(args.U_swarms)
        if self.total_UAVs % args.U_swarms != 0:
            #avg assign first, then randomly add the remainder in
            while np.sum(ups_holder) != self.total_UAVs:
                #randomly add 1 to a random swarm
                ups_holder[int(rng.random()*self.total_UAVs)] += 1

        self.UAVs_per_swarm = ups_holder
        
        ## allocate workers and coordinators (ratio rounding should favor workers)
        self.coordinators_per_swarm = [int((1-args.UAV_init_ratio)*i) for i in self.UAVs_per_swarm]
        self.workers_per_swarm = [self.UAVs_per_swarm[self.coordinators_per_swarm.index(i)]-i \
                                for i in self.coordinators_per_swarm]
    
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

        model.add(Dense(50,activation='relu',input_shape = [self.input_size]))
        model.add(Dense(50,activation='relu'))
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
        ## last time you visited some cluster
        ## add this into the state
        ## TODO
        
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
        
        else:
            
            for item in minibatch:
                # print(item)
                
                if type(item[0][0]) == list:
                    item[0][0] = np.array(item[0][0])
                
                print(item)
                
                item = tf.convert_to_tensor(item)
                
                print(item)
                
                target = self.q_net.predict(item) #item is [args.cnn_range, args.U_swarms + args.Clusters]
                
                terminated = 0
                t = self.target_network.predict(item)
                target[0][item[0][1]] = reward + self.g_discount * np.amax(t)
                
                self.q_net.fit(item,target,epochs=1,verbose=0)
                

# %% confirmation testing
test_DQN = DQN(args)

test_DQN.q_net.summary()
test_DQN.target_network.summary()

# %% function to calculate rewards

def reward_state_calc(test_DQN,current_state,current_action,current_action_space,\
                      cluster_expectations):
    
    ## ali - add penalty for not visiting nodes; want to ensure visit the nodes not being visited
    
    ## calculate current_state + current_action expected gain  
    ## determine next state, also changes last visits
    next_state_set = list(current_action_space[current_action]) #deepcopy(current_state)
    # print('begin')
    # print(next_state_set)
    
    # update last visits 
    next_state_visits = [i+1 for i in current_state[len(test_DQN.U):]]
    
    # print(current_state)
    
    current_reward = 0
    for i,j in enumerate(next_state_set):
        # print(i,j)
        
        ## reward function calculated based on elapsed time x cluster factor
        current_reward += cluster_expectations[j]*next_state_visits[j]
        
        #previously was cluster_expectations[j] * next_state_visits[j]
        next_state_visits[j] = 0 # zero out since now it will be visited
    
    next_state_set += next_state_visits

    return current_reward, next_state_set


# %% function to calculate action space

## calculate the full action space
## this is static
def action_space_calc(all_clusters,num_swarms=args.U_swarms):
    
    ## calculate all permutations
    action_space = [i for i in permutations(all_clusters,num_swarms)]
    
    return action_space

# %% building the sequence for DRL
episodes = 1
reward_storage = []

if args.centralized == True:
    reward_DQN = np.zeros((1,1,args.G_timesteps))
else:
    print('only centralized is currently supported')

# static action space - as finite swarm movement choices
action_space = action_space_calc(list(range(args.Clusters)))
#cluster_expectations = 100*np.random.rand(args.Clusters) # the distribution change over time
cluster_expectations = 100*np.array([0.005,1.6,0.8,5,0.3,0.02])

# saving some plots for debugging
fig_no = 0
cwd = os.getcwd()
state_set_all = []

## full loop beginning
for e in range(episodes):
    ## calculate and build a set of states for the UAV swarms
    
    ## randomly initialize state
    ## state_set indexed by swarm [swarm0 pos, swarm1 pos, etc., cluster 0 last visit, ...]
    init_state_set = []
    init_last_visit = np.ones(args.Clusters)# building the last visit structure
    
    temp_C_set = list(range(args.Clusters))
    for i,j in enumerate(test_DQN.U):
        init_rng_index = np.random.randint(0,len(temp_C_set))
        init_state_set.append(temp_C_set[init_rng_index])
        init_last_visit[temp_C_set[init_rng_index]] = 0
        
        del temp_C_set[init_rng_index]

    init_state_set += [int(i) for i in init_last_visit]

    ## iterate over the timesteps
    for timestep in range(args.G_timesteps):
        
        # calculate the reward
        if args.linear == True:
            ep_greed = np.max([args.ep_min, args.ep_greed*(1-10**(-3))**timestep])
            
            if timestep == 0:
                action_set = test_DQN.calc_action(state=init_state_set, \
                                                  args=args,ep_greed =ep_greed)
                
                rewards, state_set = reward_state_calc(test_DQN,init_state_set,action_set,\
                                action_space,cluster_expectations)
                
                ## store experiences
                test_DQN.store(init_state_set,action_set,rewards,state_set)
                
            else:                
                current_state_set = deepcopy(state_set)
                
                action_set = test_DQN.calc_action(state=current_state_set, \
                                                  args=args,ep_greed =ep_greed)
                
                rewards, state_set = reward_state_calc(test_DQN,current_state_set,action_set,\
                        action_space,cluster_expectations)
                
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
                        
                    test_DQN.store(init_state_set,action_set,reward1,state_set1,\
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
        if args.linear == True:
            reward_storage.append(rewards)
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
            plt.savefig(cwd+'/plots/'+str(fig_no)+'_30_epsilon_10000_lr_small.png')
            
            plt.clf()
            
            # save data
            with open(cwd+'/data/'+str(fig_no)+'_30_epsilon_10000_lr_small','wb') as f:
                pk.dump(reward_storage,f)
            
            # with open(cwd+'/data/'+str(fig_no)+'_30_epsilon_10000_lr_small_states','wb') as f:
            #     pk.dump(state_set_all,f)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            









