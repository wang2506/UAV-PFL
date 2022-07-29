# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:25:21 2022

@author: ch5b2
"""

import numpy as np
import pickle as pk
import os
import random
import argparse
from copy import deepcopy

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

parser.add_argument('--brt',type=str,default='medium',\
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


# %%
cwd = os.getcwd()
with open(cwd+'/geo_optim_chars/results_all','rb') as f:
    historical_results = pk.load(f)

# 160 per Ts
optim_180 = historical_results[:160]
optim_220 = historical_results[160:320]
optim_260 = historical_results[320:]

# 4 swarms - each with an even split
optim_180_s0 = optim_180[:40] # swarm0; each cluster takes 4, taus2 takes 2, then taus1 remains

# %%
# ## compute the reward for each one
if args.dynamic == True:
    time_drift_min = 2*np.array(range(1,args.Clusters+1)) 
    time_drift_max = deepcopy(time_drift_min) + np.array([13,11,15,3,2,5,5,1])
    cluster_limits = 20*time_drift_max
    cluster_expectations = deepcopy(time_drift_min)
else:
    ## static model drift
    ## reset seed in numpy random
    np.random.seed(args.seed)
    cluster_expectations = 25*np.random.rand(args.Clusters) #20
    cluster_limits = 20*cluster_expectations #3 - vary epsilon + gamma use this one

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
    brt = 8440*4
elif args.brt == 'vhigh3':
    brt = 8440*5
elif args.brt == 'debug' or args.brt == 'debug2':
    brt = -1e8

min_battery_levels = (brt*np.ones(args.U_swarms)).tolist()
Ts = [180,220,260]
taus1 = [1,2]
taus2 = [1,2]

# %% main loop
with open(cwd+'/drl_results/RNN/2e2seed_'+str(args.seed)+'_'\
          +str(args.ep_greed)+'_'+'visit_freq_large'+\
          '_'+str(args.g_discount)\
        +'_tanh_mse'+\
        '_'+args.brt,'rb') as f:
    freq_visits = pk.load(f)

with open(cwd+'/drl_results/RNN/2e2seed_'+str(args.seed)+'_'\
          +str(args.ep_greed)+'_'+'all_states'\
          +'test_large'+'_'+str(args.g_discount)\
        +'_tanh_mse'+\
        '_'+args.brt,'rb') as f:
    state_saves = pk.load(f)

with open(cwd+'/drl_results/RNN/2e2seed_'+str(args.seed)+'_'\
          +str(args.ep_greed)+'_'+'reward'\
          +'test_large'+'_'+str(args.g_discount)\
        +'_tanh_mse'+\
        '_'+args.brt,'rb') as f:
    reward_storage = pk.load(f)

reward_vec = []
ml_only = []
# for timestep in range(args.G_timesteps):
#     ep_greed = np.max([args.ep_min, args.ep_greed*(1-10**(-3))**timestep])

    
    ## determine what was visited

    ## determine ML reward

    ## determine/update model drift

    ## update overall reward calculation




















# %%
# all_brt_freqs = {}
# for ind_b,brt in enumerate(bat_vec):
#     for ind_s,seed in enumerate(seeds):
#         with open(cwd+'/drl_results/RNN/2e2seed_'+str(seed)+'_'\
#                   +str(ep_greed)+'_'+'visit_freq_large'+\
#                   '_'+str(g_discount)\
#                 +'_tanh_mse'+\
#                 '_'+brt,'rb') as f:
#             freq_visits = pk.load(f)
            
#         if ind_s == 0: #3 swarms, 1000 epochs
#             brt_freqs = sum((freq_visits[29000]-freq_visits[28000])[-2:])/1000/len(seeds)/3
#             # brt_freqs = sum((freq_visits[20000]-freq_visits[19000])[-2:])/1000/len(seeds)/3            
#         else:
#             brt_freqs += sum((freq_visits[29000]-freq_visits[28000])[-2:])/1000/len(seeds)/3
#             # brt_freqs = sum((freq_visits[20000]-freq_visits[19000])[-2:])/1000/len(seeds)/3                   
    
#     all_brt_freqs[brt] = brt_freqs











