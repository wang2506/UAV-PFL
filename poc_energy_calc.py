# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 11:40:53 2021

@author: henry
"""

import numpy as np
import os
from Shared_ML_code.fl_parser import ml_parser
from copy import deepcopy

# %% HFL and HNPFL parser
settings = ml_parser()
settings.swarms = 4 # we actually set --swarms to be 4, so change here 
# settings.data_style = 'fmnist'

# %% init settings
np.random.seed(1)
nodes_per_swarm = [np.random.randint(2,4) for i in range(settings.swarms)]
# because reset the seed to 1, the swarm characteristics will be the same

## powers and communication rates
# powers
total_workers = sum(nodes_per_swarm)
total_coordinators = round(total_workers*0.4)
total_uavs = total_workers+total_coordinators
total_devices = total_workers*2
leaders = deepcopy(settings.swarms)

# %% really just need flight energy
uav_tx_powers = [0.1 for i in range(total_uavs)] #20 dbm, 0.1 W
# device_tx_powers = [0.25 for i in range(devices)] #0.25 W - 24dbm, 0.15 was also used for some sims

device_tx_powers = 0.2 + (0.32-0.2)*np.random.rand(total_devices)  # want to do 23dbm (0.2) to 25dbm (0.32)
device_tx_powers = device_tx_powers.tolist()
leader_tx_powers = [0.1 for i in range(leaders)] #20 dbm 0.1W

# constants
carrier_freq = 2 * 1e9
noise_db = 1e-13 #-130 dB, we convert to watts
univ_bandwidth = 10*1e6 #MHz
mu_tx = 4*np.pi * carrier_freq/(3*1e8)
eta_los = 2 #3db 
eta_nlos = 200 #23db
path_loss_alpha = 2
psi_tx = 11.95
beta_tx = 0.14
device_uav_altitude_diff = 30 #121 #meters
device_coord_altitude_diff = 25

dist_device_uav_max = 100 # 3000 #3km
dist_device_uav_min = device_uav_altitude_diff #100

dist_uav_uav_max = 100 #1000 # make 100 m
dist_uav_uav_min = 50 #100 # 50 m

dist_uav_leader_max = 20 #1000 
dist_uav_leader_min = 10 #100

# rates
# devices to uavs
device_tx_rates = np.zeros(shape=(total_devices,total_uavs))
for q in range(total_devices):
    for j in range(total_uavs):
        dist_qj = dist_device_uav_min + (dist_device_uav_max-dist_device_uav_min) \
            *np.random.rand() # randomly determined
        theta_qj = 180/np.pi * np.arcsin(device_uav_altitude_diff / dist_qj )
        
        prob_los = 1/(1+ psi_tx * np.exp(-beta_tx*(theta_qj-psi_tx)) )
        prob_nlos = 1-prob_los
        
        la2g_qj = (mu_tx * dist_qj)**path_loss_alpha *\
            (prob_los*eta_los + prob_nlos * eta_nlos )
        
        device_tx_rates[q,j] = univ_bandwidth *\
            np.log2(1 + (device_tx_powers[q]/la2g_qj) / noise_db )
        
coord_tx_rates = np.zeros(shape=(total_coordinators,total_workers))
for h in range(total_coordinators):
    for j in range(total_workers):
        dist_hj = dist_uav_uav_min + (dist_uav_uav_max-dist_uav_uav_min) \
            * np.random.rand() #randomly determined
        
        la2a_hj = eta_los * (mu_tx * dist_hj)**path_loss_alpha 
        
        coord_tx_rates[h,j] = univ_bandwidth* \
            np.log2( 1 + (uav_tx_powers[total_workers+h]/la2a_hj) / noise_db )

worker_tx_rates = np.zeros(shape=(total_workers,1))
for j in range(total_workers):
    dist_jl = dist_uav_leader_min + (dist_uav_leader_max-dist_uav_leader_min) \
        * np.random.rand() #randomly determined
    
    la2a_jl = eta_los * (mu_tx * dist_jl)**path_loss_alpha 
    
    worker_tx_rates[j] = univ_bandwidth* \
        np.log2( 1 + (uav_tx_powers[j]/la2a_jl) / noise_db )

leader_tx_rates = min(worker_tx_rates) * np.ones(shape=(1,total_workers))

# coord_tx_rates = 5600*np.ones(shape=(coordinators,workers))
# worker_tx_rates = 5600*np.ones(shape=(workers,1))
# # device_tx_rates = 5600*np.ones(shape=(devices,workers+coordinators)) #1000
# # device_tx_rates[0,workers:] = 200000
# # device_tx_rates[1,workers:] = 200000
# leader_tx_rates = 5600*np.ones(shape=(1,workers))

## 
K_s1 = 1
alphas = 1/3*np.ones((total_workers,3))
worker_c = [1e4 for i in range(total_workers)] #1e4
freq_min = 10e6 #0.5*1e9
freq_max = 2.3*1e9

worker_freq = freq_min*np.ones(total_workers)
capacitance = 2e-28 #2e-28 #2e-16 #2e-28 #10*1e-12

rho = 1 #{i:cp.Variable(shape=(devices,coordinators+workers),pos=True) for i in range(K_s1)}
varrho = 1 #{i:cp.Variable(shape=(coordinators,workers),pos=True) for i in range(K_s1)}


## flight energy coeffs
# max_speed_uav = 57 #
min_speed_uav = 10 # km/h
seconds_conversion = 2 #5
air_density = 1.225 
zero_lift_drag_max = 0.0378 #based on sopwith camel 
zero_lift_breakpoint = 0.0269
zero_lift_drag_min = 0.0161 #based on p-51 mustang
wing_area_max = 3 #m^2
wing_area_breakpoint = 1.75
wing_area_min = 0.5
oswald_eff = 0.8 #0.7 to 0.85
speed = 5 #circular rotation, should not be fast
aspect_ratio = 4.5 #2 to 7 
weight_max = 10 #kg
weight_breakpoint = 5.05
weight_min = 0.1 #100g
kg_newton = 9.8 

psi_j = (np.zeros(total_workers)).tolist()
psi_h = (np.zeros(total_coordinators)).tolist() #2 #11.95 #0.5 #0.25 #1 #10

for j in range(total_workers):
    c1 = 0.5 * air_density * (zero_lift_breakpoint +\
        (zero_lift_drag_max-zero_lift_breakpoint)*np.random.rand()) * \
        (wing_area_breakpoint + (wing_area_max - wing_area_breakpoint)*np.random.rand())
    
    c2 = 2 * (weight_breakpoint + (weight_max - weight_breakpoint)*np.random.rand())**2 \
        / (np.pi * oswald_eff * (wing_area_breakpoint + \
        (wing_area_max - wing_area_breakpoint)*np.random.rand()) * air_density**3 )   
    
    psi_j[j] = c1 * (speed**3) + c2/speed

for h in range(total_coordinators):
    c1 = 0.5 * air_density * (zero_lift_drag_min +\
        (zero_lift_breakpoint - zero_lift_drag_min )*np.random.rand()) * \
        (wing_area_min + (wing_area_breakpoint - wing_area_min)*np.random.rand())
    
    c2 = 2 * (weight_min + (weight_breakpoint - weight_min)*np.random.rand())**2 \
        / (np.pi * oswald_eff * (wing_area_min + \
        ( wing_area_breakpoint - wing_area_min)*np.random.rand()) * air_density**3 )   
    
    psi_h[h] = c1 * (speed**3) + c2/speed
    
# psi_m = c1 * (min_speed_uav**3) + c2/speed   # parameter for leader flight to nearest AP
psi_l = psi_j[np.random.randint(0,total_workers)] #+ 2*psi_m*2/tau_s2

# image parameters
img_to_bits =  8e4 #20
params_to_bits = 1e4 #2

# %% HFL and HNPFL settings
if settings.data_style == 'mnist':
    avg_qty = 2500 # MNIST
elif settings.data_style == 'fmnist':
    avg_qty = 3500 # FMNIST
elif settings.data_style == 'cifar10':
    avg_qty = 3500
elif settings.data_style == 'mlradio':
    avg_qty = 4500 

data_qty = np.random.normal(avg_qty,avg_qty/10, size=(total_uavs)).astype(int)

# %% energy calcs per iteration
# calculate the processing energy needed
eng_p = (1e-10 * np.ones(shape=total_workers)).tolist()
eng_p_obj = 1e-10 

for j in range(total_workers):
    eng_p[j] += 0.5*capacitance*worker_c[0]*data_qty[j]* \
        (1/3+1/3+1/3) * np.power(worker_freq[j],2) #*cp.power(worker_freq[i][j],2)
    eng_p_obj += eng_p[j]
    
# calculate tx energy by UAVs
eng_tx_u = (1e-10 * np.ones(shape=total_coordinators)).tolist()
eng_tx_u_obj = 1e-10 

for q in range(total_devices):
    for j in range(total_coordinators):
        for k in range(total_workers):
            eng_tx_u[j] += 0.25*0.25*1000*uav_tx_powers[j]*\
            img_to_bits/coord_tx_rates[j][k]
            
            eng_tx_u_obj += eng_tx_u[j]
            
# calculate worker tx energy
eng_tx_w = np.zeros(shape=total_workers)

for j in range(total_workers):
    eng_tx_w[j] += params_to_bits * uav_tx_powers[j] /worker_tx_rates[j]

# calculate device tx energy
eng_tx_q = 1e-10 #q reps device

for j in range(total_devices):
    for k in range(total_uavs):#coordinators+workers):
        eng_tx_q += 0.25*1000*device_tx_powers[j]*\
            img_to_bits/device_tx_rates[j][k]


# calculate worker and coordinators flight energy
eng_f_j = np.zeros(total_workers)
eng_f_h = np.zeros(total_coordinators)

for j in range(total_workers):
    eng_f_j[j] += seconds_conversion * psi_j[j]

for h in range(total_coordinators):
    eng_f_h[h] += seconds_conversion * psi_h[h]

eng_f_l = seconds_conversion * psi_l

# leader energy computation
eng_tx_l = 0

for l in range(leaders):
    # build vector 
    bit_div_rates = []
    for j in range(total_workers):
        bit_div_rates.append(params_to_bits/leader_tx_rates[0][l])
    eng_tx_l += np.max(bit_div_rates)*leader_tx_powers[l]

energy_iter = eng_p_obj + eng_tx_u_obj + sum(eng_tx_w)*settings.swarms + eng_tx_q + sum(eng_f_j) \
    + sum(eng_f_h) + eng_f_l*settings.swarms + eng_tx_l*settings.swarms

# %% determine the total energy consumptions for HFL and HNPFL 
ratios = [1,2,4,8]
swarm_agg = 1
global_agg = 1

if settings.data_style == 'mnist':
    # acc thresh = 60%
    # s_hfl means fixed s, vary tau_G
    s_hnpfl = np.array([4, 7, 14, 31])*energy_iter
    s_hfl = np.array([8, 14, 25, 34])*energy_iter
    
    g_hnpfl = np.array([4, 8, 14, 25])*energy_iter
    g_hfl = np.array([8, 16, 26, 32])*energy_iter
    
elif settings.data_style == 'fmnist':
    # acc thresh = 45%
    s_hnpfl = np.array([4,6,11,20])*energy_iter
    s_hfl = np.array([9,12,19,31])*energy_iter

    g_hnpfl = np.array([4,7,13,20])*energy_iter
    g_hfl = np.array([9,12,20,32])*energy_iter

elif settings.data_style == 'cifar10':
    # acc thresh = 40%
    s_hnpfl = np.array([6,6,7,12])*energy_iter
    s_hfl = np.array([40,40,40,40])*energy_iter

    g_hnpfl = np.array([6,6,7,12])*energy_iter
    g_hfl = np.array([40,40,40,40])*energy_iter    

# elif settings.data_style == 'mlradio':
    

# %% calculate for swarm aggs, before calculating for global aggs
## treat standard hovering as approximately the same energy cost as travelling to AP





















