# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 16:12:20 2021

@author: henry
"""
import numpy as np
import os

import matplotlib.pyplot as plt

from copy import deepcopy
import cvxpy as cp

import mosek
import pickle as pk

#### this is the cvxpy optimization file
## for lists - first element must be nonzero for dgp to accept
## also, cp.sum, sum, np.sum the underlying wrapper starts with a 0 then +=
np.random.seed(1)

# %% objective function test
T_s = 20

K_s1 = 2 #1
K_s2 = 2#5
tau_s1 = 2
tau_s2 = 2

img_to_bits =  8e4 #20
params_to_bits = 1e4 #2

swarms = 1
leaders = 1 #same as swarms
workers = 5 #2
coordinators = 3 #2
devices = 10 #2 
#5 uavs, 10 device, 10 uavs, 15 devices, 100 iterations

## powers and communication rates
# powers
uav_tx_powers = [0.1 for i in range(workers+coordinators)] #20 dbm, 0.1 W
# device_tx_powers = [0.25 for i in range(devices)] #0.25 W - 24dbm, 0.15 was also used for some sims
device_tx_powers = 0.2 + (0.32-0.2)*np.random.rand(devices)  # want to do 23dbm (0.2) to 25dbm (0.32)
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
device_uav_altitude_diff = 121 #meters

dist_device_uav_max = 3000 #3km
dist_device_uav_min = device_uav_altitude_diff #100

dist_uav_uav_max = 1000 #1km
dist_uav_uav_min = 100 # 50 m

dist_uav_leader_max = 1000
dist_uav_leader_min = 100

# rates
# devices to uavs
device_tx_rates = np.zeros(shape=(devices,workers+coordinators))
for q in range(devices):
    for j in range(workers + coordinators):
        dist_qj = dist_device_uav_min + (dist_device_uav_max-dist_device_uav_min) \
            *np.random.rand() # randomly determined
        theta_qj = 180/np.pi * np.arcsin(device_uav_altitude_diff / dist_qj )
        
        prob_los = 1/(1+ psi_tx * np.exp(-beta_tx*(theta_qj-psi_tx)) )
        prob_nlos = 1-prob_los
        
        la2g_qj = (mu_tx * dist_qj)**path_loss_alpha *\
            (prob_los*eta_los + prob_nlos * eta_nlos )
        
        device_tx_rates[q,j] = univ_bandwidth *\
            np.log2(1 + (device_tx_powers[q]/la2g_qj) / noise_db )
        
coord_tx_rates = np.zeros(shape=(coordinators,workers))
for h in range(coordinators):
    for j in range(workers):
        dist_hj = dist_uav_uav_min + (dist_uav_uav_max-dist_uav_uav_min) \
            * np.random.rand() #randomly determined
        
        la2a_hj = eta_los * (mu_tx * dist_hj)**path_loss_alpha 
        
        coord_tx_rates[h,j] = univ_bandwidth* \
            np.log2( 1 + (uav_tx_powers[workers+h]/la2a_hj) / noise_db )

worker_tx_rates = np.zeros(shape=(workers,1))
for j in range(workers):
    dist_jl = dist_uav_leader_min + (dist_uav_leader_max-dist_uav_leader_min) \
        * np.random.rand() #randomly determined
    
    la2a_jl = eta_los * (mu_tx * dist_jl)**path_loss_alpha 
    
    worker_tx_rates[j] = univ_bandwidth* \
        np.log2( 1 + (uav_tx_powers[j]/la2a_jl) / noise_db )

leader_tx_rates = min(worker_tx_rates) * np.ones(shape=(1,workers))

# coord_tx_rates = 5600*np.ones(shape=(coordinators,workers))
# worker_tx_rates = 5600*np.ones(shape=(workers,1))
# # device_tx_rates = 5600*np.ones(shape=(devices,workers+coordinators)) #1000
# # device_tx_rates[0,workers:] = 200000
# # device_tx_rates[1,workers:] = 200000
# leader_tx_rates = 5600*np.ones(shape=(1,workers))

## 
alphas = {i:cp.Variable(shape=(workers,3),pos=True) for i in range(K_s1)}

worker_c = [1e6 for i in range(workers)] #1e4
freq_min = 0.5*1e9
freq_max = 2.3*1e9
worker_freq = {i:[cp.Variable(pos=True) for i in range(workers)] for i in range(K_s1)}
capacitance = 2e-26 #2e-28 #2e-16 #2e-28 #10*1e-12

rho = {i:cp.Variable(shape=(devices,coordinators+workers),pos=True) for i in range(K_s1)}
varrho = {i:cp.Variable(shape=(coordinators,workers),pos=True) for i in range(K_s1)}

Omega = cp.Variable(pos=True)


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

psi_j = (np.zeros(workers)).tolist()
psi_h = (np.zeros(coordinators)).tolist() #2 #11.95 #0.5 #0.25 #1 #10

for j in range(workers):
    c1 = 0.5 * air_density * (zero_lift_breakpoint +\
        (zero_lift_drag_max-zero_lift_breakpoint)*np.random.rand()) * \
        (wing_area_breakpoint + (wing_area_max - wing_area_breakpoint)*np.random.rand())
    
    c2 = 2 * (weight_breakpoint + (weight_max - weight_breakpoint)*np.random.rand())**2 \
        / (np.pi * oswald_eff * (wing_area_breakpoint + \
        (wing_area_max - wing_area_breakpoint)*np.random.rand()) * air_density**3 )   
    
    psi_j[j] = c1 * (speed**3) + c2/speed

for h in range(coordinators):
    c1 = 0.5 * air_density * (zero_lift_drag_min +\
        (zero_lift_breakpoint - zero_lift_drag_min )*np.random.rand()) * \
        (wing_area_min + (wing_area_breakpoint - wing_area_min)*np.random.rand())
    
    c2 = 2 * (weight_min + (weight_breakpoint - weight_min)*np.random.rand())**2 \
        / (np.pi * oswald_eff * (wing_area_min + \
        ( wing_area_breakpoint - wing_area_min)*np.random.rand()) * air_density**3 )   
    
    psi_h[h] = c1 * (speed**3) + c2/speed
    
# psi_m = c1 * (min_speed_uav**3) + c2/speed   # parameter for leader flight to nearest AP
psi_l = psi_j[np.random.randint(0,workers)] #+ 2*psi_m*2/tau_s2


D_q = {i:[500 for j in range(devices)]  for i in range(K_s1)}

# building D_j
D_j = {i:[] for i in range(K_s1)}

for i in range(K_s1):
    if i == 0:
        for j in range(coordinators+workers):
            temp = [1e-10] #to satisfy log reqs
            for k in range(devices):
                temp.append(rho[i][k,j]*D_q[i][k])
        
            D_j[i].append(cp.sum(temp))
    else:
        # device offloading
        for j in range(coordinators+workers):
            temp = [1e-10]
            for k in range(devices): #devices to all uavs
                temp.append(rho[i][k,j]*D_q[i][k])
                
            D_j[i].append(cp.sum(temp))
        
        # coordinator offloading
        for j in range(workers):
            temp = [1e-10]
            for k in range(coordinators):
                temp.append(varrho[i][k,j]*D_j[i][workers+k])
            D_j[i][j] += cp.sum(temp)
            
            # for k in range(coordinators): # coordinator to workers only
            #     temp.append(varrho[i][workers+k,j]*D_j[i][workers+k]) #*D_j[i-1][workers+k])

B_j = {i:[600 for j in range(workers)] for i in range(K_s1)} 
B_j_coord = {i:[600 for h in range(coordinators)] for i in range(K_s1)}

# %% build objective

for i in range(1,K_s1):
    print('new K_s1 iteration')
    ## theta terms
    # calculate the processing energy needed
    eng_p = (1e-10 * np.ones(shape=workers)).tolist()
    eng_p_obj = 1e-10 
    
    for j in range(workers):
        eng_p[j] += 0.5*capacitance*worker_c[j]*D_j[i][j]* \
            (cp.sum(alphas[i][j,:])) * cp.power(worker_freq[i][j],2) #*cp.power(worker_freq[i][j],2)
        eng_p_obj += eng_p[j]
        
    # calculate tx energy by UAVs
    eng_tx_u = (1e-10 * np.ones(shape=coordinators)).tolist()
    eng_tx_u_obj = 1e-10 
    
    # for i in range(K_s1):
    # for j in range(coordinators):
    #     for k in range(workers):
    #         eng_tx_u += varrho[i][j,k]*D_j[i][workers+j]*uav_tx_powers[workers+j]*\
    #         img_to_bits/coord_tx_rates[j][k]

    for q in range(devices):
        for j in range(coordinators):
            for k in range(workers):
                eng_tx_u[j] += varrho[i][j,k]*rho[i][q,workers+j]*D_q[i][q]*uav_tx_powers[workers+j]*\
                img_to_bits/coord_tx_rates[j][k]
                
                eng_tx_u_obj += eng_tx_u[j]
                
    # calculate worker tx energy 
    eng_tx_w = np.zeros(shape=workers)

    for j in range(workers):
        eng_tx_w[j] += params_to_bits * uav_tx_powers[j] /worker_tx_rates[j]

    # calculate device tx energy
    eng_tx_q = 1e-10 #q reps device

    for j in range(devices):
        for k in range(coordinators+workers):
            eng_tx_q += rho[i][j,k]*D_q[i][j]*device_tx_powers[j]*\
                img_to_bits/device_tx_rates[j][k]


    # calculate worker and coordinators flight energy
    eng_f_j = np.zeros(workers)
    eng_f_h = np.zeros(coordinators)
    
    for j in range(workers):
        eng_f_j[j] += seconds_conversion * psi_j[j]

    for h in range(coordinators):
        eng_f_h[h] += seconds_conversion * psi_h[h]

    eng_f_l = seconds_conversion * psi_l

    # leader energy computation
    eng_tx_l = 0

    for l in range(leaders):
        # build vector 
        bit_div_rates = []
        for j in range(workers):
            bit_div_rates.append(params_to_bits/leader_tx_rates[l])
        eng_tx_l += np.max(bit_div_rates)*leader_tx_powers[l]
    
    ## build constraints
    constraints = []

    # alphas
    for j in range(workers):
        for k in range(3):
            constraints.append(alphas[i][j,k] >= 1e-10)#1e-10)
            constraints.append(alphas[i][j,k] <= 1)

    # freqs
    for j in range(workers):
        constraints.append(worker_freq[i][j] <= freq_max)
        constraints.append(worker_freq[i][j] >= freq_min)


    # offloading vars
    for j in range(devices):
        for k in range(coordinators+workers):
            constraints.append(rho[i][j,k] <= 1)
            constraints.append(rho[i][j,k] >= 1e-10)

        constraints.append(cp.sum(rho[i][j,:]) <= 1)


    for j in range(coordinators):
        for k in range(workers):
            constraints.append(varrho[i][j,k] <= 1)
            constraints.append(varrho[i][j,k] >= 1e-10)

        constraints.append(cp.sum(varrho[i][j,:]) <= 1)
    
    zeta_p = np.zeros(workers).tolist()
    zeta_g_j = np.zeros(workers).tolist()
    zeta_g_h = np.zeros(coordinators).tolist()
    zeta_local = 1000
    
    # implementing zeta constraint
    for j in range(workers):    
        zeta_p[j] = worker_c[j] * (cp.sum(alphas[i][j,:])) * D_j[i][j] / worker_freq[i][j]
        zeta_g_j[j] = 1e-10 #img_to_bits
        
        for q in range(devices):
            zeta_g_j[j] += rho[i][q,j] * D_q[i][q] * img_to_bits/device_tx_rates[q,j]
            
        for h in range(coordinators):
            for q in range(devices):
                zeta_g_j[j] += varrho[i][h,j] * rho[i][q,workers+h] * D_q[i][q] *\
                    img_to_bits/coord_tx_rates[h,j] 
                #* D_j[i][workers+h] *\
                    #img_to_bits/coord_tx_rates[h,j] #device_tx_rates[q,workers+h] produces good results
                    #rho[i][q,workers+h] * D_q[i][q] *
                    
        constraints.append(zeta_p[j] + zeta_g_j[j] <= zeta_local)
    
    for h in range(coordinators):
        zeta_g_h[h] = 1e-10
        for q in range(devices):
            zeta_g_h[h] += rho[i][q,workers+h] * D_q[i][q] * \
                img_to_bits/device_tx_rates[q,workers+h]
        
        constraints.append(zeta_g_h[h] <= zeta_local)
        
    # # data capacity constraints
    for j in range(workers):
        constraints.append(D_j[i][j] <= B_j[i][j])
    for h in range(coordinators):
        constraints.append(D_j[i][workers+h] <= B_j_coord[i][h])
    
    eng_bat_j = 2000 * np.ones(shape=workers)
    eng_bat_h = 2000 * np.ones(shape=coordinators)
    eng_thresh_j = 20 * np.ones(shape=workers)
    eng_thresh_h = 20 * np.ones(shape=coordinators)
    
    eng_bat_l = 2000
    eng_thresh_l = 20 
    
    # energy limits 
    for j in range(workers):
        constraints.append(eng_p[j] + eng_tx_w[j] + eng_f_j[j] \
            <= (eng_bat_j[j] - eng_thresh_j[j])/(K_s1) )

    for h in range(coordinators):
        constraints.append(eng_f_h[h] + eng_tx_u[h] \
            <= (eng_bat_h[h] - eng_thresh_h[h])/( K_s1 ) )
    
    # constraints.append(eng_f_l  + eng_tx_l \
    #         <= (eng_bat_l - eng_thresh_l)/ K_s1)
    

## 1-theta terms
eta_2 = 1e-4 #1e-4
mu_F = 20
grad_fu_scale = 1/(eta_2/2 - 6 *eta_2**2 * mu_F/2) * (3*eta_2**2 *mu_F/2 + eta_2)

B, eta_1, mu = 500, 1e-3, 10 #500
sigma_j_H,sigma_j_G = 50, 50 ##sigma_j_H greatly affects data?
gamma_u_F, gamma_F = 10, 10

## need to approximate delta_u
# delta_u = D_j[]
delta_u_holder = []
# init_delta_u = 50 #1e-10

max_approx_iters = 100 #50 #100 #100 #200 #50
plot_obj = []
plot_energy = []
plot_acc = []

# calc objective fxn value with initial estimate numbers
# eng_p_prior = 0.01*0.5*capacitance*()
alpha_ind_init = 0.9

test_init_rho = 0.05
test_init_varrho = 0.1

sigma_c_H,sigma_c_G = 50, 50
B_cluster = 500

for i in range(1,K_s1):
    for t in range(max_approx_iters):
        delta_u_approx = 1
        delta_u = 1e-10
        
        init_q_j = []
        init_q_j1,init_q_j2,init_q_j3 = [],[],[]
        init_h_j1,init_h_j2,init_h_j3 = [],[],[]
        
        ## varrho and D_j must be considered hereafter
        # else: #i != 0, D_j and varrho active
    
        # init_h_j1 = np.zeros(shape=(devices,coordinators,workers))
        # init_h_j2 = np.zeros(shape=(devices,coordinators,workers))
        # init_h_j3 = np.zeros(shape=(devices,coordinators,workers))
        
        if t == 0:

            # calculate delta_u and delta_u_approx
            for j in range(workers):
                alpha_j1, alpha_j2, alpha_j3 = alpha_ind_init,alpha_ind_init,alpha_ind_init # 0.9,0.9,0.9
                alpha_j = alpha_j1 + alpha_j2 + alpha_j3
                
                for q in range(devices):
                    rho_qj, D_q_approx =  test_init_rho, D_q[i][q]  #1/(workers+coordinators),
                    
                    init_q_j1.append(alpha_j1*rho_qj*D_q_approx)
                    init_q_j2.append(alpha_j2*rho_qj*D_q_approx)
                    init_q_j3.append(alpha_j3*rho_qj*D_q_approx)
                    
                    delta_u += alpha_j*rho_qj*D_q_approx
                
                for h in range(coordinators):
                    varrho_hj = test_init_varrho #1/workers                        
                    
                    for q in range(devices):
                        rho_qh, D_q_approx = test_init_rho, D_q[i][q] #1/(workers+coordinators), 
                        
                        init_h_j1.append(alpha_j1*varrho_hj*rho_qh*D_q_approx)
                        init_h_j2.append(alpha_j2*varrho_hj*rho_qh*D_q_approx)
                        init_h_j3.append(alpha_j3*varrho_hj*rho_qh*D_q_approx)
                    
                        # init_h_j1[q,h,j] = alpha_j*varrho_hj*rho_qh*D_q_approx

                        delta_u += alpha_j*varrho_hj*rho_qh*D_q_approx
            
            # powers_check = 0
            for j in range(workers):
                for q in range(devices):
                    # build true q_j factors
                    delta_u_approx *= (alphas[i][j,0] *rho[i][q,j] * D_q[i][q] *\
                        delta_u/init_q_j1[j*devices+q] ) **(init_q_j1[j*devices+q]/delta_u)
                    delta_u_approx *= (alphas[i][j,1] *rho[i][q,j] * D_q[i][q] *\
                        delta_u/init_q_j2[j*devices+q] ) **(init_q_j2[j*devices+q]/delta_u)
                    delta_u_approx *= (alphas[i][j,2] *rho[i][q,j] * D_q[i][q] *\
                        delta_u/init_q_j3[j*devices+q] ) **(init_q_j3[j*devices+q]/delta_u)  
                    
                    # delta_u_approx *= (alpha_j1 * test_init_rho * D_q[i][q] *\
                    #     delta_u/init_q_j1[j*devices+q] ) **(init_q_j1[j*devices+q]/delta_u)
                    # delta_u_approx *= (alpha_j2 * test_init_rho * D_q[i][q] *\
                    #     delta_u/init_q_j2[j*devices+q] ) **(init_q_j2[j*devices+q]/delta_u)
                    # delta_u_approx *= (alpha_j3 * test_init_rho * D_q[i][q] *\
                    #     delta_u/init_q_j3[j*devices+q] ) **(init_q_j3[j*devices+q]/delta_u)  
                    
                    # powers_check += (init_q_j1[j*devices+q]+\
                    #     init_q_j2[j*devices+q]+init_q_j3[j*devices+q])/delta_u 
                    
                for h in range(coordinators):
                    for q in range(devices):
                        delta_u_approx *= (alphas[i][j,0] *varrho[i][h,j] *rho[i][q,workers+h]*\
                            D_q[i][q] * delta_u/init_h_j1[j*devices*coordinators+h*devices+q] ) \
                            **(init_h_j1[j*devices*coordinators+h*devices+q]/delta_u)
                        delta_u_approx *= (alphas[i][j,1] *varrho[i][h,j] *rho[i][q,workers+h]*\
                            D_q[i][q] * delta_u/init_h_j2[j*devices*coordinators+h*devices+q] ) \
                            **(init_h_j2[j*devices*coordinators+h*devices+q]/delta_u)
                        delta_u_approx *= (alphas[i][j,2] *varrho[i][h,j] *rho[i][q,workers+h]*\
                            D_q[i][q] * delta_u/init_h_j3[j*devices*coordinators+h*devices+q] ) \
                            **(init_h_j3[j*devices*coordinators+h*devices+q]/delta_u)
            
                        # delta_u_approx *= (alpha_j1 * test_init_varrho * test_init_rho *\
                        #     D_q[i][q] * delta_u/init_h_j1[j*devices*coordinators+h*devices+q] ) \
                        #     **(init_h_j1[j*devices*coordinators+h*devices+q]/delta_u)
                        # delta_u_approx *= (alpha_j2 * test_init_varrho * test_init_rho *\
                        #     D_q[i][q] * delta_u/init_h_j2[j*devices*coordinators+h*devices+q] ) \
                        #     **(init_h_j2[j*devices*coordinators+h*devices+q]/delta_u)
                        # delta_u_approx *= (alpha_j3 * test_init_varrho * test_init_rho *\
                        #     D_q[i][q] * delta_u/init_h_j3[j*devices*coordinators+h*devices+q] ) \
                        #     **(init_h_j3[j*devices*coordinators+h*devices+q]/delta_u)
                        
                        # powers_check += (init_h_j1[j*devices*coordinators+h*devices+q]+\
                        #     init_h_j2[j*devices*coordinators+h*devices+q]+\
                        #     init_h_j3[j*devices*coordinators+h*devices+q])/delta_u 
                        
                            # print(delta_u_approx)
                        
                        
            # print('powers_check = ' + str(powers_check))        
            # print(delta_u_approx)
            # print(delta_u)
            
            ## calc sigma
            # first approx D_j
            sigma_j = []
            mismatch_j = []
            
            for j in range(workers):
                sigma_prev, D_j_approx = 10, 1
                D_j_prev = 1e-10 #the total denom
                
                # previous D_j estimate
                for q in range(devices):
                    D_q_approx, rho_qj = D_q[i][q], test_init_rho #1/(workers+coordinators)
                    denom_prev = D_q_approx*rho_qj
                    
                    D_j_prev += denom_prev

                for h in range(coordinators):
                    varrho_hj = test_init_varrho #1/workers
                    
                    for q in range(devices):
                        D_q_approx, rho_qh  = D_q[i][q], test_init_rho #1/(workers+coordinators) 
                        denom_prev = D_q_approx*varrho_hj*rho_qh
                    
                        D_j_prev += denom_prev
                
                # sig_powers_check = 0
                
                # print('dj_prev =' + str(D_j_prev))
                # calc approximation
                for q in range(devices):
                    D_q_approx, rho_qj = D_q[i][q], test_init_rho #1/(workers+coordinators)
                    denom_prev = D_q_approx*rho_qj
                    
                    D_j_approx *= (rho[i][q,j] * D_q[i][q] *\
                        D_j_prev/denom_prev)**(denom_prev/D_j_prev)
                      
                    # D_j_approx *= (rho_qj * D_q_approx *\
                    #     D_j_prev/denom_prev)**(denom_prev/D_j_prev)
                        
                    # sig_powers_check += denom_prev/D_j_prev
                    
                        
                for h in range(coordinators):
                    varrho_hj = test_init_varrho
                    
                    for q in range(devices):
                        D_q_approx, rho_qh  = D_q[i][q], test_init_rho #1/(workers+coordinators)
                        denom_prev = D_q_approx*varrho_hj*rho_qh
                    
                        D_j_approx *= (varrho[i][h,j] * rho[i][q,workers+h] *\
                            D_q[i][q] * D_j_prev/denom_prev) \
                            **(denom_prev/D_j_prev)
                        
                        # D_j_approx *= (varrho_hj * test_init_rho *\
                        #     D_q_approx * D_j_prev/denom_prev) \
                        #     **(denom_prev/D_j_prev)
                            
                        # sig_powers_check += denom_prev/D_j_prev
                
                # print('dj_approx =' + str(D_j_approx))
                # print(sig_powers_check)
                
                sigma_j_pre = 3*B**2*eta_1**2*sigma_j_H*alphas[i][j,0]*alphas[i][j,1]*D_j[i][j] \
                    + 3*eta_1**2 * sigma_j_H * sigma_j_G  \
                    *( alphas[i][j,0] + mu**2 * eta_1**2 * alphas[i][j,1]) \
                    + 12 * sigma_j_G * alphas[i][j,2] * D_j[i][j] \
                    *( alphas[i][j,0] + mu**2 * eta_1**2 * alphas[i][j,1] )
                
                # print(D_j_approx)
                
                
                sigma_j.append(sigma_j_pre/(cp.prod(alphas[i][j,:])*D_j_approx**2))
                # sigma_j.append(1/cp.prod(alphas[i][j,:]))
                
                
                ## mismatch term
                mismatch_pre_factor = 3*B_cluster**2 * eta_1**2 * sigma_c_H * D_j[i][j] \
                    + 3*eta_1**2 * sigma_c_H * sigma_c_G * (1+ mu**2 * eta_1**2) \
                    + 12 * sigma_c_G * D_j[i][j] *(1+mu**2 * eta_1**2)
                
                # D_j_approx doesn't consider the alpha factors, we already have the approx
                mismatch_j.append(mismatch_pre_factor / D_j_approx**2 )
                
        # the t!=0  case
        else:
            for j in range(workers):
                alpha_j1, alpha_j2, alpha_j3 = alphas[i][j,0].value, \
                    alphas[i][j,1].value, alphas[i][j,2].value
                alpha_j = alpha_j1 + alpha_j2 + alpha_j3
                
                for q in range(devices):
                    rho_qj, D_q_approx = rho[i][q,j].value, D_q[i][q]
                    
                    init_q_j1.append(alpha_j1*rho_qj*D_q_approx)
                    init_q_j2.append(alpha_j2*rho_qj*D_q_approx)
                    init_q_j3.append(alpha_j3*rho_qj*D_q_approx)
                    
                    delta_u += alpha_j*rho_qj*D_q_approx
                
                for h in range(coordinators):
                    varrho_hj = varrho[i][h,j].value
                    
                    for q in range(devices):
                        rho_qh, D_q_approx = rho[i][q,workers+h].value, D_q[i][q] 
                    
                        init_h_j1.append(alpha_j1*varrho_hj*rho_qh*D_q_approx)
                        init_h_j2.append(alpha_j2*varrho_hj*rho_qh*D_q_approx)
                        init_h_j3.append(alpha_j3*varrho_hj*rho_qh*D_q_approx)
                        
                        delta_u += alpha_j*varrho_hj*rho_qh*D_q_approx
                
            for j in range(workers):
                for q in range(devices):
                    # build true q_j factors
                    delta_u_approx *= (alphas[i][j,0] *rho[i][q,j] * D_q[i][q] *\
                        delta_u/init_q_j1[j*devices+q] ) **(init_q_j1[j*devices+q]/delta_u)
                    delta_u_approx *= (alphas[i][j,1] *rho[i][q,j] * D_q[i][q] *\
                        delta_u/init_q_j2[j*devices+q] ) **(init_q_j2[j*devices+q]/delta_u)
                    delta_u_approx *= (alphas[i][j,2] *rho[i][q,j] * D_q[i][q] *\
                        delta_u/init_q_j3[j*devices+q] ) **(init_q_j3[j*devices+q]/delta_u)  
        
                for h in range(coordinators):
                    for q in range(devices):
                        delta_u_approx *= (alphas[i][j,0] *varrho[i][h,j] *rho[i][q,workers+h]*\
                            D_q[i][q] * delta_u/init_h_j1[j*devices*coordinators+h*devices+q] ) \
                            **(init_h_j1[j*devices*coordinators+h*devices+q]/delta_u)
                        delta_u_approx *= (alphas[i][j,1] *varrho[i][h,j] *rho[i][q,workers+h]*\
                            D_q[i][q] * delta_u/init_h_j2[j*devices*coordinators+h*devices+q] ) \
                            **(init_h_j2[j*devices*coordinators+h*devices+q]/delta_u)
                        delta_u_approx *= (alphas[i][j,2] *varrho[i][h,j] *rho[i][q,workers+h]*\
                            D_q[i][q] * delta_u/init_h_j3[j*devices*coordinators+h*devices+q] ) \
                            **(init_h_j3[j*devices*coordinators+h*devices+q]/delta_u)
            
            ## calc sigma
            # first approx D_j
            for j in range(workers):
                sigma_prev, D_j_approx = sigma_j[j].value, 1
                D_j_prev = 1e-10
                
                for q in range(devices):
                    D_q_approx, rho_qj = D_q[i][q], rho[i][q,j].value
                    denom_prev = D_q_approx*rho_qj
                    
                    D_j_prev += denom_prev

                for h in range(coordinators):
                    varrho_hj = varrho[i][h,j].value
                    
                    for q in range(devices):
                        D_q_approx, rho_qh = D_q[i][q], rho[i][q,workers+h].value
                        denom_prev = D_q_approx*rho_qh*varrho_hj
                    
                        D_j_prev += denom_prev
                
                
                # calc approximation
                for q in range(devices):
                    D_q_approx, rho_qj = D_q[i][q], rho[i][q,j].value
                    denom_prev = D_q_approx*rho_qj
                    
                    D_j_approx *= (rho[i][q,j] * D_q[i][q] *\
                        D_j_prev/denom_prev)**(denom_prev/D_j_prev)
                        
                for h in range(coordinators):
                    varrho_hj = varrho[i][h,j].value

                    for q in range(devices):
                        D_q_approx, rho_qh = D_q[i][q], rho[i][q,workers+h].value
                        denom_prev = D_q_approx*rho_qh*varrho_hj
                    
                        D_j_approx *= (varrho[i][h,j] * rho[i][q,workers+h] *\
                            D_q[i][q] * D_j_prev/denom_prev) \
                            **(denom_prev/D_j_prev)
                        
                sigma_j_pre = 3*(B**2)*(eta_1**2)*sigma_j_H*alphas[i][j,0] \
                    *alphas[i][j,1]*D_j[i][j] \
                    + 3*eta_1**2 * sigma_j_H *sigma_j_G \
                    *( alphas[i][j,0] + mu**2 * eta_1**2 * alphas[i][j,1]) \
                    + 12 * sigma_j_G * alphas[i][j,2] * D_j[i][j] \
                    *( alphas[i][j,0] + mu**2 * eta_1**2 * alphas[i][j,1] )
                
                sigma_j[j] = sigma_j_pre/(cp.prod(alphas[i][j,:])*D_j_approx**2)
                sigma_j[j] = 1/(cp.prod(alphas[i][j,:])*D_j_approx**2)

                ## mismatch term
                mismatch_pre_factor = 3*B_cluster**2 * eta_1**2 * sigma_c_H * D_j[i][j] \
                    + 3*eta_1**2 * sigma_c_H * sigma_c_G * (1+ mu**2 * eta_1**2) \
                    + 12 * sigma_c_G * D_j[i][j] *(1+mu**2 * eta_1**2)
                
                # D_j_approx doesn't consider the alpha factors, we already have the approx
                mismatch_j[j] = (mismatch_pre_factor / D_j_approx**2 )


        # sum delta_j/delta_u
        delta_diff_sigma = 1e-10 # delta_diff = delta_diff/delta_u_approx
        # print(i)
        for j in range(workers):
            delta_diff_sigma += cp.sum(alphas[i][j,:])*D_j[i][j]*sigma_j[j]/delta_u_approx
        
        # upsilon calc
        upsilon = 1e-10
        upsilon_pt1, upsilon_pt2 = 1e-10, 1e-10
        for j in range(workers):
            upsilon_pt1 += (16*eta_2**2 * cp.sum(alphas[i][j,:])*D_j[i][j]*tau_s1*\
                sigma_j[j]/delta_u_approx + 24 * eta_2**2 * gamma_u_F ) * \
                ( (8+48*eta_2**2 * mu_F**2)**tau_s1 - 1 )/( (8+48*eta_2**2 * mu_F**2) - 1 )
                
            upsilon_pt2 += (16*eta_2**2 * cp.sum(alphas[i][j,:])*\
                D_j[i][j]*tau_s1*tau_s2 * sigma_j[j]/delta_u_approx \
                + 24 * eta_2**2 * gamma_F) * \
                ( (8+48*eta_2**2*mu_F**2)**(tau_s1*tau_s2) - 1) /( (8+48*eta_2**2 *mu_F**2) - 1)    
            
        upsilon = upsilon_pt1 + upsilon_pt2
        
        # mismatch calc
        mismatch = 1e-10
        for j in range(workers):
            mismatch_scale = cp.sum(alphas[i][j,:]) * D_j[i][j] /delta_u_approx
            mismatch += mismatch_scale * mismatch_j[j] 
    
        
        # learning combine
        theta = 0.8
        
        true_objective = (1-theta)*(eng_p_obj + eng_tx_u_obj + eng_tx_q + sum(eng_tx_w) \
            + eng_tx_l + sum(eng_f_j) + sum(eng_f_h) + eng_f_l ) + \
            theta* (grad_fu_scale*(delta_diff_sigma + mu_F**2 * upsilon) \
            + 3 * eta_2**2 * mu_F * gamma_u_F / (eta_2/2 - 6 * eta_2**2 * mu_F/2) \
            + mismatch)
        #     #delta_i/delta_u
        
        # true_objective = (1-theta)*(eng_p_obj + eng_tx_u_obj + eng_tx_q + sum(eng_tx_w) \
        #     + eng_tx_l + sum(eng_f_j) + sum(eng_f_h) + eng_f_l )
        
        # true_objective = theta* (grad_fu_scale*(delta_diff_sigma + mu_F**2 * upsilon) \
        #     + 3 * eta_2**2 * mu_F * gamma_u_F / (eta_2/2 - 6 * eta_2**2 * mu_F/2) \
        #     + mismatch)
            # eng_p + eng_tx_u + eng_tx_w + eng_tx_q +  + sum(eng_f_j)
        # true_objective = grad_fu_scale*delta_diff
        # true_objective = (eng_p + eng_tx_u + eng_tx_w + eng_tx_q + eng_f_j)
        
        # constraints.append(true_objective <= Omega) #1e-10)
        # constraints.append(Omega >= 1e-10)
        
        ## objective function
        # objective_fxn = Omega
        objective_fxn = true_objective

    # %% formulate problem and solve
        # print(alpha_j1,alpha_j2,alpha_j3)    
        
        prob = cp.Problem(cp.Minimize(objective_fxn),constraints)
        prob.solve(gp=True,solver=cp.MOSEK) #, max_iters=10000) #verbose=True,solver=cp.SCS,
        
        print('new iteration')
        # print(prob.value)
        # print(objective_fxn.value)
        # plot_obj.append(np.round(prob.value,5))
        plot_obj.append(prob.value)
        
        temp_energy = (eng_p_obj + eng_tx_u_obj + eng_tx_q + sum(eng_tx_w) \
            + eng_tx_l + sum(eng_f_j) + sum(eng_f_h) + eng_f_l).value
        # plot_energy.append(np.round(temp_energy,5))
        plot_energy.append(temp_energy)
        
        # print(grad_fu_scale)
        print('delta-diff')
        print(delta_diff_sigma.value)
        
        print('delta_u_approx_vals')
        print(delta_u_approx.value)
        
        print('sigmas and data')
        for j in range(workers):
            print(sigma_j[j].value)
            print(D_j[i][j].value)
        
        temp_acc = (grad_fu_scale*(delta_diff_sigma + mu_F**2 * upsilon) \
            + 3 * eta_2**2 * mu_F * gamma_u_F)
        #grad_fu_scale*delta_diff.value
        # plot_acc.append(np.round(temp_acc,5))
        plot_acc.append(temp_acc.value)
        
        # print(varrho[i].value)
        # print(rho[i].value)
        
        # for j in range(workers):
        #     print(alphas[i][j,0].value)
        #     print(alphas[i][j,1].value)
        #     print(alphas[i][j,2].value)
        
        
# %% plotting
    plt.figure(1)
    
    plt.plot(plot_obj)
    plt.title('objective fxn value - iter: ' + str(i))
    
    plt.figure(2)
    plt.plot(plot_energy)
    plt.title('aggregate energy - iter: ' + str(i))
    
    plt.figure(3)
    plt.plot(plot_acc)
    plt.title('gradient result - iter: ' + str(i))


init_learning_estimate = 0
temp_delta_u_approx = 0

for j in range(workers):
    temp_alpha_est = 0.9*3
    temp_df_est = test_init_rho*2*D_q[i][j] + test_init_varrho*test_init_rho*2*D_q[i][j] 
    temp_sig_j_est = 3*eta_1**2*sigma_j_H*B**2/(0.9*temp_df_est) \
        + 3*eta_1**2*sigma_j_H*sigma_j_G*\
        ( 0.9 + mu**2 * eta_1**2 *0.9 )/ (0.9*0.9*0.9*temp_df_est**2) \
        + 12*sigma_j_G*( 0.9 + mu**2 * eta_1**2 *0.9 )/ (0.9*0.9*temp_df_est)
    
    # temp_delta_j = temp_alpha_est*temp_df_est # already manually typed in
    init_learning_estimate += temp_sig_j_est*temp_alpha_est*temp_df_est
    temp_delta_u_approx += temp_alpha_est*temp_df_est


init_learning_estimate *= grad_fu_scale/temp_delta_u_approx

# determine initial upsilon
upsilon_pt1_estimate = 0
upsilon_pt2_estimate = 0

for j in range(workers):
    temp_alpha_est = 0.9*3
    temp_df_est = test_init_rho*2*D_q[i][j] + test_init_varrho*test_init_rho*2*D_q[i][j] 
    temp_sig_j_est = 3*eta_1**2*sigma_j_H*B**2/(0.9*temp_df_est) \
        + 3*eta_1**2*sigma_j_H*sigma_j_G*\
        ( 0.9 + mu**2 * eta_1**2 *0.9 )/ (0.9*0.9*0.9*temp_df_est**2) \
        + 12*sigma_j_G*( 0.9 + mu**2 * eta_1**2 *0.9 )/ (0.9*0.9*temp_df_est)
    
    upsilon_pt1_estimate += (16*eta_2**2 * temp_alpha_est * temp_df_est * tau_s1 *\
            temp_sig_j_est/temp_delta_u_approx + 24 * eta_2**2 * gamma_u_F) *\
            ( (8+48*eta_2**2 *mu_F**2)**tau_s1-1)/ ( (8+48*eta_2**2 * mu_F**2)-1)
    
    upsilon_pt2_estimate += (16*eta_2**2 * temp_alpha_est * temp_df_est * tau_s1 * tau_s2*\
            temp_sig_j_est/temp_delta_u_approx + 24 * eta_2**2 * gamma_F) *\
            ( (8+48*eta_2**2 *mu_F**2)**(tau_s1*tau_s2)-1)/ ( (8+48*eta_2**2 * mu_F**2)-1)

upsilon_estimate = upsilon_pt1_estimate + upsilon_pt2_estimate


# determine initial mismatch
mismatch_estimate = 0
for j in range(workers):
    temp_alpha_est = 0.9*3
    temp_df_est = test_init_rho*2*D_q[i][j] + test_init_varrho*test_init_rho*2*D_q[i][j] 
    
    mismatch_scale = temp_alpha_est * temp_df_est / temp_delta_u_approx
    mismatch_estimate += mismatch_scale * \
        ( 3*eta_1**2 * sigma_c_H * B_cluster**2 * temp_df_est +\
        3 * eta_1**2 * sigma_c_H * sigma_c_G * ( 1 + mu**2 * eta_1**2 ) \
        + 12 * sigma_c_G * temp_df_est * ( 1 + mu**2 * eta_1**2) ) / temp_df_est**2

# calc initial point
plot_acc[0] = init_learning_estimate + grad_fu_scale * upsilon_estimate \
    + 3*eta_2**2 * mu_F *gamma_u_F / (eta_2/2 - 6 * eta_2**2 * mu_F/2)  \
    + mismatch_estimate
plot_obj[0] = plot_acc[0]*theta + (1-theta)*plot_energy[0]

# %% saving
cwd = os.getcwd()
with open(cwd + '/optim_plots/objective_d_'+str(devices)+'_w_'\
    +str(workers)+'_c_'+str(coordinators),'wb') as f:
    pk.dump(plot_obj,f)

with open(cwd + '/optim_plots/gradient_d_'+str(devices)+'_w_'\
    +str(workers)+'_c_'+str(coordinators),'wb') as f:
    pk.dump(plot_acc,f)

with open(cwd + '/optim_plots/energy_d_'+str(devices)+'_w_'\
    +str(workers)+'_c_'+str(coordinators),'wb') as f:
    pk.dump(plot_energy,f)


# %% graveyard
# build the alpha dict
# for i in range(workers):
#     for j in range(4):        
#         if j == 0:
#             alphas[i].append(1e-10) #to satisfy log reqs
#         else:
#             alphas[i].append(cp.Variable(pos=True))

# # rho vec populate
# for i in range(devices):
#     rho[i] = [cp.Variable(pos=True) for j in range(coordinators+workers)]

# # varrho vec populate
# for i in range(coordinators+1):
#     varrho[i] = [cp.Variable(pos=True) for j in range(coordinators+workers)]
    

        # if t == 0 and i == 0:
            
        #     ## calculate delta_u_approx
        #     for j in range(workers):
        #         alpha_j1,alpha_j2,alpha_j3 = 0.9,0.9,0.9
        #         alpha_j = alpha_j1 + alpha_j2 + alpha_j3
                
        #         for q in range(devices):
        #             rho_qj = 1/(workers) #0.95
        #             D_q_approx = D_q[i][q] # as this is currently hard coded
                    
        #             # init_q_j.append(init_i_alpha*init_rho_qj*init_D_q)
        #             init_q_j1.append(alpha_j1*rho_qj*D_q_approx)
        #             init_q_j2.append(alpha_j2*rho_qj*D_q_approx)
        #             init_q_j3.append(alpha_j3*rho_qj*D_q_approx)
                    
        #             delta_u += alpha_j*rho_qj*D_q_approx

        #     for j in range(workers):
        #         for q in range(devices):
        #             delta_u_approx *= (alphas[i][j,0] *rho[i][q,j] * D_q[i][q] *\
        #                 delta_u/init_q_j1[j*devices+q] ) **(init_q_j1[j*devices+q]/delta_u)
        #             delta_u_approx *= (alphas[i][j,1] *rho[i][q,j] * D_q[i][q] *\
        #                 delta_u/init_q_j2[j*devices+q] ) **(init_q_j2[j*devices+q]/delta_u)
        #             delta_u_approx *= (alphas[i][j,2] *rho[i][q,j] * D_q[i][q] *\
        #                 delta_u/init_q_j3[j*devices+q] ) **(init_q_j3[j*devices+q]/delta_u)                
            
        #     ## calc sigma
        #     # first approx D_j
        #     sigma_j = []
        #     for j in range(workers):
        #         sigma_prev, D_j_approx = 10, 1
        #         D_j_prev = 1e-10
                
        #         for q in range(devices):
        #             D_q_approx, rho_qj = D_q[i][q], 1/workers #0.95
        #             denom_prev = D_q_approx*rho_qj
                    
        #             # D_j_approx *= (rho[i][q,j] * D_q[i][q] *\
        #             #     sigma_prev/denom_prev ) **(denom_prev/sigma_prev)
        #             D_j_prev += denom_prev
                
        #         for q in range(devices):
        #             D_q_approx, rho_qj = D_q[i][q], 1/(workers)
        #             denom_prev = D_q_approx*rho_qj
                    
        #             D_j_approx *= (rho[i][q,j] * D_q[i][q] *\
        #                 D_j_prev/denom_prev)**(denom_prev/D_j_prev)
                    
        #         sigma_j_pre = 3*(B**2)*(eta_1**2)*sigma_j_H*alphas[i][j,0]*alphas[i][j,1]*D_j[i][j] \
        #             + 3*eta_1**2 * sigma_j_H * sigma_j_G  \
        #             *( alphas[i][j,0] + mu**2 * eta_1**2 * alphas[i][j,1]) \
        #             + 12 * sigma_j_G * alphas[i][j,2] * D_j[i][j] \
        #             *( alphas[i][j,0] + mu**2 * eta_1**2 * alphas[i][j,1] )
                
        #         sigma_j.append(sigma_j_pre/(cp.prod(alphas[i][j,:])*D_j_approx**2))
                
        # elif t!= 0 and i == 0:
        #     ## update delta_u_approx
        #     for j in range(workers):
        #         alpha_j1, alpha_j2, alpha_j3 = alphas[i][j,0].value, \
        #             alphas[i][j,1].value, alphas[i][j,2].value
        #         alpha_j = alpha_j1 + alpha_j2 + alpha_j3
                
        #         for q in range(devices):
        #             rho_qj = rho[i][q,j].value
        #             D_q_approx = D_q[i][q]
                    
        #             init_q_j1.append(alpha_j1*rho_qj*D_q_approx)
        #             init_q_j2.append(alpha_j2*rho_qj*D_q_approx)
        #             init_q_j3.append(alpha_j3*rho_qj*D_q_approx)
                    
        #             delta_u += alpha_j*rho_qj*D_q_approx
                    
        #     for j in range(workers):
        #         for q in range(devices):
        #             delta_u_approx *= (alphas[i][j,0] *rho[i][q,j] * D_q[i][q] *\
        #                 delta_u/init_q_j1[j*devices+q] ) **(init_q_j1[j*devices+q]/delta_u)
        #             delta_u_approx *= (alphas[i][j,1] *rho[i][q,j] * D_q[i][q] *\
        #                 delta_u/init_q_j2[j*devices+q] ) **(init_q_j2[j*devices+q]/delta_u)
        #             delta_u_approx *= (alphas[i][j,2] *rho[i][q,j] * D_q[i][q] *\
        #                 delta_u/init_q_j3[j*devices+q] ) **(init_q_j3[j*devices+q]/delta_u)   
            
        #     ## update sigma, via updating D_j approx
        #     # first approx D_j
        #     for j in range(workers):
        #         sigma_prev, D_j_approx = sigma_j[j].value, 1 #D_j[i][j].value
        #         D_j_prev = 1e-10
                
        #         for q in range(devices):
        #             D_q_approx, rho_qj = D_q[i][q], rho[i][q,j].value
        #             denom_prev = D_q_approx*rho_qj
                    
        #             D_j_prev += denom_prev
                
        #         for q in range(devices):
        #             D_q_approx, rho_qj = D_q[i][q], rho[i][q,j].value
        #             denom_prev = D_q_approx*rho_qj
                    
        #             D_j_approx *= (rho[i][q,j] * D_q[i][q] *\
        #                 D_j_prev/denom_prev)**(denom_prev/D_j_prev)
                    
        #         sigma_j_pre = 3*B**2*eta_1**2*sigma_j_H*alphas[i][j,0]*alphas[i][j,1]*D_j[i][j] \
        #             + 3*eta_1**2 * sigma_j_H * sigma_j_G  \
        #             *( alphas[i][j,0] + mu**2 * eta_1**2 * alphas[i][j,1]) \
        #             + 12 * sigma_j_G * alphas[i][j,2] * D_j[i][j] \
        #             *( alphas[i][j,0] + mu**2 * eta_1**2 * alphas[i][j,1] )
                
        #         sigma_j[j] = sigma_j_pre/(cp.prod(alphas[i][j,:])*D_j_approx**2)

