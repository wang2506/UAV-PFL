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

import time
import sys

#### this is the cvxpy optimization file
seed = 1 #1,3,5,;6,7,8,;10,13, 16; 17
np.random.seed(seed)
start = time.time()

sys.setrecursionlimit(5000)

# %% plot defns
fig1,ax1 = plt.subplots()
ax1.set_title('objective function - devices 10, workers 2')
ax1.set_xlabel('posynomial approximation iteration')
ax1.set_ylabel('value')
ax1.grid(True)

fig2,ax2 = plt.subplots()
ax2.set_title('aggregate energy - devices 10, workers 2')
ax2.set_xlabel('posynomial approximation iteration')
ax2.set_ylabel('energy value')
ax2.grid(True)

fig3,ax3 = plt.subplots()
ax3.set_title('gradient result - devices 10, workers 2')
ax3.set_xlabel('posynomial approximation iteration')
ax3.set_ylabel('value')
ax3.grid(True)

# %% theta change loop
# 0.4
plot_counter = 0
theta_vec = [0.01] #[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for theta in theta_vec:
    # %% objective function test
    np.random.seed(seed) # prevents randomization from messing up energies
    T_s = 8 #200
    tau_s1 = 2
    tau_s2 = 2
    K_s1 = int(T_s/tau_s1) #17
    K_s2 = int(T_s/(tau_s1*tau_s2)) #5
    
    img_to_bits =  8e4 #20
    params_to_bits = 1e4 #2
    
    swarms = 1 #
    leaders = 1 #same as swarms
    workers = 2 #5 #2-4 #3-5
    coordinators = 2 #3 #2 #1-2
    devices = 10 #2  #9-12

    ## powers and communication rates
    # powers
    uav_tx_powers = [0.1 for i in range(workers+coordinators)] #20 dbm, 0.1 W
    # device_tx_powers = [0.25 for i in range(devices)] #0.25 W - 24dbm, 0.15 was also used for some sims
    device_tx_powers = 0.2 + (0.32-0.2)*np.random.rand(devices)  # want to do 23dbm (0.2) to 25dbm (0.32)
    device_tx_powers = device_tx_powers.tolist()
    leader_tx_powers = [0.1 for i in range(leaders)] #20 dbm 0.1W
    
    # constants
    carrier_freq = 2 * 1e9
    noise_db = 4e-21 #-174 dBm/Hz, we convert to watts
    univ_bandwidth = 2e6 #MHz #10 MHz
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
    
    dist_uav_uav_varrho_emph = np.ones(workers+coordinators)
    dist_uav_uav_varrho_emph[0:workers] = 100
    dist_uav_uav_varrho_emph[workers:] = 20
    
    dist_uav_leader_max = 20 #1000 
    dist_uav_leader_min = 10 #100
    
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
            # dist_hj = dist_uav_uav_min + (dist_uav_uav_max-dist_uav_uav_min) \
            #     * np.random.rand() #randomly determined
            # TODO : note, the above was commented in order to emphasize varrho diffs
            dist_hj = dist_uav_uav_varrho_emph[h+j]
            
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
    
    ## 
    alphas = {i:cp.Variable(shape=(workers,3),pos=True) for i in range(K_s1)}
    
    worker_c = [1e4 for i in range(workers)] #1e4
    freq_min = 10e6 #0.5*1e9
    freq_max = 2.3*1e9
    worker_freq = {i:[cp.Variable(pos=True) for i in range(workers)] for i in range(K_s1)}
    capacitance = 2e-28 #2e-16 #2e-28 #10*1e-12
    
    rho = {i:cp.Variable(shape=(devices,coordinators+workers),pos=True) for i in range(K_s1)}
    varrho = {i:cp.Variable(shape=(coordinators,workers),pos=True) for i in range(K_s1)}
    
    Omega = cp.Variable(pos=True) #unused
    
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
    psi_h = (np.zeros(coordinators)).tolist()
    
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
        
    psi_m = c1 * (min_speed_uav**3) + c2/speed   # parameter for leader flight to nearest AP
    psi_l = psi_j[np.random.randint(0,workers)] #+ 2*psi_m*2/tau_s2

    # building D_j
    D_q = {i:[500 for j in range(devices)]  for i in range(K_s1)}
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
                
    B_j = {i:[600 for j in range(workers)] for i in range(K_s1)} 
    B_j_coord = {i:[600 for h in range(coordinators)] for i in range(K_s1)}
    
    # %% build optimization problem
    
    # %% build constraints + energy terms
    constraints = []
    # build energy terms [first, no loop terms, then loop terms]
    # calculate worker tx energy 
    eng_tx_w = np.zeros(shape=workers)
    for j in range(workers):
        eng_tx_w[j] += K_s1*params_to_bits * uav_tx_powers[j] /worker_tx_rates[j]
    
    # calculate worker and coordinators flight energy
    eng_f_j = np.zeros(workers)
    for j in range(workers):
        eng_f_j[j] += T_s*seconds_conversion * psi_j[j]
    
    eng_f_h = np.zeros(coordinators)
    for h in range(coordinators):
        eng_f_h[h] += T_s*seconds_conversion * psi_h[h]    
    
    # leader energy computation
    eng_f_l = T_s*seconds_conversion * psi_l + 2 *K_s2 *psi_m * dist_device_uav_min
    bit_div_rates = []
    for j in range(workers):
        bit_div_rates.append(params_to_bits/leader_tx_rates[0])
    eng_tx_l = K_s1*np.max(bit_div_rates)*leader_tx_powers[0]
    
    # build loop dependent energy terms
    eng_p = (1e-10 * np.ones(shape=(K_s1,workers))).tolist()
    eng_tx_u = (1e-10 * np.ones(shape=(K_s1,coordinators))).tolist()
    eng_p_obj, eng_tx_u_obj, eng_tx_q = 1e-10,1e-10,1e-10
    for i in range(K_s1):
        # calculate the processing energy needed
        for j in range(workers):
            eng_p[i][j] += tau_s1*0.5*capacitance*worker_c[j]*D_j[i][j]* \
                (cp.sum(alphas[i][j,:])) * cp.power(worker_freq[i][j],2) 
            eng_p_obj += tau_s1*eng_p[j]
        
        # calculate coordinator tx energy
        for q in range(devices):
            for j in range(coordinators):
                for k in range(workers):
                    eng_tx_u[i][j] += varrho[i][j,k]*rho[i][q,workers+j]\
                        *D_q[i][q]*uav_tx_powers[workers+j]*\
                        img_to_bits/coord_tx_rates[j][k]                
                    eng_tx_u_obj += eng_tx_u[j]
        
        # calculate device tx energy
        for j in range(devices):
            for k in range(coordinators+workers):
                eng_tx_q += rho[i][j,k]*D_q[i][j]*device_tx_powers[j]*\
                    img_to_bits/device_tx_rates[j][k]
        
        # alphas
        for j in range(workers):
            for k in range(3):
                constraints.append(alphas[i][j,k] >= 1e-10) 
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
    
    # Add the energy constraints into the constraints structure (also 20,000 bat)
    eng_bat_j = 20000 * np.ones(shape=workers)
    eng_bat_h = 20000 * np.ones(shape=coordinators)
    eng_bat_l = 20000        
    eng_thresh_j = 20 * np.ones(shape=workers)
    eng_thresh_h = 20 * np.ones(shape=coordinators)
    eng_thresh_l = 20 
    
    # energy limit constraints
    for j in range(workers):
        constraints.append(cp.sum([t_eng_p[j] for t_eng_p in eng_p]) \
            + eng_tx_w[j] + eng_f_j[j] <= (eng_bat_j[j] - eng_thresh_j[j]) )

    for h in range(coordinators):
        constraints.append(cp.sum([t_eng_tx_u[h] for t_eng_tx_u in eng_tx_u]) +\
            + eng_f_h[h] <= (eng_bat_h[h] - eng_thresh_h[h]) )
    
    constraints.append(eng_f_l  + eng_tx_l <= (eng_bat_l - eng_thresh_l))
    
    # build timing terms
    zeta_p = np.zeros((K_s1,workers)).tolist()
    zeta_g_j = np.zeros((K_s1,workers)).tolist()
    zeta_g_h = np.zeros((K_s1,coordinators)).tolist()
    zeta_local = 5 #1000
    for i in range(K_s1):
        for j in range(workers):    
            zeta_p[i][j] = tau_s1*worker_c[j] * (cp.sum(alphas[i][j,:])) \
                * D_j[i][j] / worker_freq[i][j]
            
            zeta_g_j[i][j] = 1e-10
            for q in range(devices):
                zeta_g_j[i][j] += rho[i][q,j] * D_q[i][q] * img_to_bits/device_tx_rates[q,j]
            if i != 0: #takes a cycle for the data to get to coordinators
                for h in range(coordinators):
                    for q in range(devices):
                        zeta_g_j[i][j] += varrho[i][h,j] * rho[i][q,workers+h] * D_q[i][q] *\
                            img_to_bits/coord_tx_rates[h,j] 
            
            constraints.append(zeta_p[i][j] + zeta_g_j[i][j] <= zeta_local)
        
        for h in range(coordinators):
            zeta_g_h[i][h] = 1e-10
            for q in range(devices):
                zeta_g_h[i][h] += rho[i][q,workers+h] * D_q[i][q] * \
                    img_to_bits/device_tx_rates[q,workers+h]
            
            constraints.append(zeta_g_h[i][h] <= zeta_local)
    
    ## 1-theta terms
    eta_2 = 1e-4 #1e-4
    mu_F = 20
    grad_fu_scale = 1/(eta_2/2 - 6 *eta_2**2 * mu_F/2) * (3*eta_2**2 *mu_F/2 + eta_2)
    
    B, eta_1, mu = 500, 1e-3, 10 #500
    sigma_j_H,sigma_j_G = 50, 50
    gamma_u_F, gamma_F = 10, 10
    
    ## need to approximate delta_u
    # delta_u = D_j[]
    delta_u_holder = []
    
    max_approx_iters = 2 #5 #10 #50 #100 #100 #200 #50
    # max_approx_iters = 5
    plot_obj = []
    plot_energy = []
    plot_acc = []
    
    # calc objective fxn value with initial estimate numbers
    alpha_ind_init = 0.9    
    test_init_rho = 0.05
    test_init_varrho = 0.1
    
    sigma_c_H,sigma_c_G = 50, 50 #50, 50
    B_cluster = 500
    
    # TODO: this is wrong!!! - the for loops should be flipped!!
    for t in range(max_approx_iters):
        # calc weighted sigma term
        delta_u_approx_vec = [1 for i in range(1,K_s1)]
        delta_u_vec = [1e-10 for i in range(1,K_s1)]
        for ks2 in range(1,K_s2+1):
            # calc delta_j = tau_s1*alpha_j(k)*D_j(k)
            delta_j_vec = []
            for j in range(workers): 
                # calc delta_j
                delta_j_vec.append(tau_s1*cp.sum(alphas[])
                
            #calc delta_u
            
            
            min_ks1, max_ks1 = (ks2-1)*tau_s1, ks2*tau_s1-1 #0 starting index
            for ks1 in range(min_ks1, max_ks1):
                for j in range(workers):
                    #calc sigma 
    
    
    
    for i in range(1,K_s1): #K_s2??
        for t in range(max_approx_iters):
            delta_u_approx = 1
            delta_u = 1e-10
            
            init_q_j = []
            init_q_j1,init_q_j2,init_q_j3 = [],[],[]
            init_h_j1,init_h_j2,init_h_j3 = [],[],[]
            
            ## varrho and D_j must be considered hereafter
            if t == 0:
                # calculate delta_u and delta_u_approx
                for j in range(workers):
                    alpha_j1, alpha_j2, alpha_j3 = alpha_ind_init,alpha_ind_init,alpha_ind_init
                    alpha_j = alpha_j1 + alpha_j2 + alpha_j3
                    
                    # device to worker delta_u
                    for q in range(devices):
                        rho_qj, D_q_approx =  test_init_rho, D_q[i][q]  #1/(workers+coordinators),
                        
                        init_q_j1.append(alpha_j1*rho_qj*D_q_approx)
                        init_q_j2.append(alpha_j2*rho_qj*D_q_approx)
                        init_q_j3.append(alpha_j3*rho_qj*D_q_approx)
                        
                        delta_u += alpha_j*rho_qj*D_q_approx

                    # device to coord delta_u
                    for h in range(coordinators):
                        varrho_hj = test_init_varrho #1/workers                        
                        
                        for q in range(devices):
                            rho_qh, D_q_approx = test_init_rho, D_q[i][q] #1/(workers+coordinators), 
                            
                            init_h_j1.append(alpha_j1*varrho_hj*rho_qh*D_q_approx)
                            init_h_j2.append(alpha_j2*varrho_hj*rho_qh*D_q_approx)
                            init_h_j3.append(alpha_j3*varrho_hj*rho_qh*D_q_approx)
    
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
                    
                    # calc approximation
                    for q in range(devices):
                        D_q_approx, rho_qj = D_q[i][q], test_init_rho #1/(workers+coordinators)
                        denom_prev = D_q_approx*rho_qj
                        
                        D_j_approx *= (rho[i][q,j] * D_q[i][q] *\
                            D_j_prev/denom_prev)**(denom_prev/D_j_prev)
                        
                    for h in range(coordinators):
                        varrho_hj = test_init_varrho
                        
                        for q in range(devices):
                            D_q_approx, rho_qh  = D_q[i][q], test_init_rho #1/(workers+coordinators)
                            denom_prev = D_q_approx*varrho_hj*rho_qh
                        
                            D_j_approx *= (varrho[i][h,j] * rho[i][q,workers+h] *\
                                D_q[i][q] * D_j_prev/denom_prev) \
                                **(denom_prev/D_j_prev)
                    
                    sigma_j_pre = 3*B**2*eta_1**2*sigma_j_H*alphas[i][j,0]*alphas[i][j,1]*D_j[i][j] \
                        + 3*eta_1**2 * sigma_j_H * sigma_j_G  \
                        *( alphas[i][j,0] + mu**2 * eta_1**2 * alphas[i][j,1]) \
                        + 12 * sigma_j_G * alphas[i][j,2] * D_j[i][j] \
                        *( alphas[i][j,0] + mu**2 * eta_1**2 * alphas[i][j,1] )
                    
                    sigma_j.append(sigma_j_pre/(cp.prod(alphas[i][j,:])*D_j_approx**2))
                    
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
            
            # mismatch calc (\widehat{\Xi})
            mismatch = 1e-10
            for j in range(workers):
                mismatch_scale = cp.sum(alphas[i][j,:]) * D_j[i][j] /delta_u_approx
                mismatch += mismatch_scale * mismatch_j[j] 
            
            # learning combine
            true_objective = (1-theta)*(eng_p_obj + eng_tx_u_obj + eng_tx_q) + \
                theta* (grad_fu_scale*(delta_diff_sigma + mu_F**2 * upsilon) \
                + 3 * eta_2**2 * mu_F * gamma_u_F / (eta_2/2 - 6 * eta_2**2 * mu_F/2) \
                + mismatch)
            
            ## objective function
            objective_fxn = true_objective

        # %% formulate problem and solve
            # print(alpha_j1,alpha_j2,alpha_j3)    
            print(time.time()-start)
            
            prob = cp.Problem(cp.Minimize(objective_fxn),constraints)
            prob.solve(gp=True,solver=cp.MOSEK) #, max_iters=10000) #verbose=True,solver=cp.SCS,
            
            print('new iteration')
            # print(prob.value)
            # print(objective_fxn.value)
            # plot_obj.append(np.round(prob.value,5))
            plot_obj.append(prob.value)
            
            temp_energy = (eng_p_obj + eng_tx_u_obj + eng_tx_q).value
            # + sum(eng_f_j) + sum(eng_f_h) + eng_f_l
            # + sum(eng_tx_w)  + eng_tx_l
            
            
            # plot_energy.append(np.round(temp_energy,5))
            plot_energy.append(temp_energy)
            
            # # print(grad_fu_scale)
            # print('delta-diff')
            # print(delta_diff_sigma.value)
            
            # print('delta_u_approx_vals')
            # print(delta_u_approx.value)
            
            # print('sigmas and data')
            # for j in range(workers):
            #     print(sigma_j[j].value)
            #     print(D_j[i][j].value)
            
            temp_acc = (grad_fu_scale*(delta_diff_sigma + mu_F**2 * upsilon) \
                + 3 * eta_2**2 * mu_F * gamma_u_F)
            #grad_fu_scale*delta_diff.value
            # plot_acc.append(np.round(temp_acc,5))
            plot_acc.append(temp_acc.value)
            
            
    # %% plotting    
    init_learning_estimate = 0
    temp_delta_u_approx = 0
    temp_alpha_est = 0.1*3 #0.9*3 ## testing random - manually save
    for j in range(workers):
        # temp_alpha_est = 0.1*3 
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
        # temp_alpha_est = 0.1*3 #0.9*3
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
        # temp_alpha_est = 0.1*3 #0.9*3
        temp_df_est = test_init_rho*2*D_q[i][j] + test_init_varrho*test_init_rho*2*D_q[i][j] 
        
        mismatch_scale = temp_alpha_est * temp_df_est / temp_delta_u_approx
        mismatch_estimate += mismatch_scale * \
            ( 3*eta_1**2 * sigma_c_H * B_cluster**2 * temp_df_est +\
            3 * eta_1**2 * sigma_c_H * sigma_c_G * ( 1 + mu**2 * eta_1**2 ) \
            + 12 * sigma_c_G * temp_df_est * ( 1 + mu**2 * eta_1**2) ) / temp_df_est**2
    
    
    # calc init estimated energy
    const_energies = sum(eng_tx_w) + eng_tx_l # + sum(eng_f_j) + sum(eng_f_h) + eng_f_l
    test_init_rho = 0.4
    eng_p_obj2 = 0 
    eng_p2 = np.zeros(devices).tolist()
    for j in range(workers):
        eng_p2[j] += 0.5*capacitance*worker_c[j]*test_init_rho*devices*D_q[i][j]* \
            temp_alpha_est * 0.5e6**2 #*cp.power(worker_freq[i][j],2)
        eng_p_obj2 += tau_s1*eng_p2[j]
    
    
    eng_tx_u_obj2 = 0
    eng_tx_u2 = np.zeros(coordinators).tolist()
    test_init_varrho = 0.3
    for q in range(devices):
        for j in range(coordinators):
            for k in range(workers):
                eng_tx_u2[j] += test_init_varrho*test_init_rho*D_q[i][q]*uav_tx_powers[workers+j]*\
                img_to_bits/coord_tx_rates[j][k]
                
                eng_tx_u_obj2 += eng_tx_u2[j]
    
    
    eng_tx_q2 = 0 #q reps device
    for j in range(devices):
        for k in range(coordinators+workers):
            eng_tx_q2 += test_init_rho*D_q[i][j]*device_tx_powers[j]*\
                img_to_bits/device_tx_rates[j][k]
    
    const_energies += eng_p_obj2 + eng_tx_u_obj2 + eng_tx_q2
    
    
    # calc initial point
    plot_energy.append(const_energies)
    plot_energy = np.roll(plot_energy,1)
    
    plot_acc.append(init_learning_estimate + grad_fu_scale * upsilon_estimate \
        + 3*eta_2**2 * mu_F *gamma_u_F / (eta_2/2 - 6 * eta_2**2 * mu_F/2)  \
        + mismatch_estimate)
    plot_acc = np.roll(plot_acc,1)
    
    plot_obj.append(plot_acc[0]*theta+(1-theta)*plot_energy[0])
    plot_obj = np.roll(plot_obj,1)
    
    # plot_acc[0] = init_learning_estimate + grad_fu_scale * upsilon_estimate \
        # + 3*eta_2**2 * mu_F *gamma_u_F / (eta_2/2 - 6 * eta_2**2 * mu_F/2)  \
        # + mismatch_estimate
    # plot_obj[0] = plot_acc[0]*theta + (1-theta)*plot_energy[0]
    
    print(time.time() - start)

    print(rho[1].value)
    print(varrho[1].value)
    print(alphas[1].value)
    worker_freq2 = [kk.value for kk in worker_freq[1]]
    print(worker_freq2)
    print('done printing optimization results')


    # plt.figure(1)
    
    ax1.plot(plot_obj)
    # plt.title('objective fxn value - iter: ' + str(i))
    # plt.title('objective fxn - devices: ' + str(devices) + 'workers: ' + str(workers))
    # plt.savefig()
    
    # plt.figure(2)
    ax2.plot(plot_energy)
    # plt.title('aggregate energy - iter: ' + str(i))
    
    # plt.figure(3)
    ax3.plot(plot_acc)
    # print(plot_acc)
    # plt.title('gradient result - iter: ' + str(i))

    # ## store the plot_obj/energy/acc
    # cwd = os.getcwd()
    # subfolder = 'avg' #'greed'
    # with open(cwd+'/geo_optim_chars/'+subfolder+'/147_tau_adjust_seed_'+str(seed)+\
    #           '_default_'+str(theta)+'_obj','wb') as f:
    #     pk.dump(plot_obj,f)

    # with open(cwd+'/geo_optim_chars/'+subfolder+'/147_tau_adjust_seed_'+str(seed)+\
    #           'default_'+str(theta)+'_energy','wb') as f:
    #     pk.dump(plot_energy,f)

    # with open(cwd+'/geo_optim_chars/'+subfolder+'/147_tau_adjust_seed_'+str(seed)+\
    #           'default_'+str(theta)+'_acc','wb') as f:
    #     pk.dump(plot_acc,f)    
    
    # # subfolder = 'greed'
    # # with open(cwd+'/geo_optim_chars/'+subfolder+'/tau_adjust'+\
    # #           '_default_'+str(theta)+'_obj','wb') as f:
    # #     pk.dump(plot_obj,f)

    # # with open(cwd+'/geo_optim_chars/'+subfolder+'/tau_adjust+\
    # #           '_default_'+str(theta)+'_energy','wb') as f:
    # #     pk.dump(plot_energy,f)

    # # with open(cwd+'/geo_optim_chars/'+subfolder+'/tau_adjust+\
    # #           '_default_'+str(theta)+'_acc','wb') as f:
    # #     pk.dump(plot_acc,f)    
    
    # # ## store the data
    # # cwd = os.getcwd()
    # with open(cwd+'/geo_optim_chars/'+subfolder+'/147_tau_adjust_seed_'+str(seed)+\
    #           '_'+str(theta)+'rho','wb') as f:
    #     pk.dump(rho[1].value,f)
    
    # with open(cwd+'/geo_optim_chars/'+subfolder+'/147_tau_adjust_seed_'+str(seed)+\
    #           '_'+str(theta)+'varrho','wb') as f:
    #     pk.dump(varrho[1].value,f)
    
    # with open(cwd+'/geo_optim_chars/'+subfolder+'/147_tau_adjust_seed_'+str(seed)+\
    #           '_'+str(theta)+'alphas','wb') as f:
    #     pk.dump(alphas[1].value,f)
    
    # # worker freqs and D_j
    # worker_freq2 = []
    # for l in worker_freq[1]:
    #     worker_freq2.append(l.value)
    
    # D_j2 = []
    # for l in D_j[1]:
    #     D_j2.append(l.value)
    
    # # with open(cwd+'/geo_optim_chars/'+subfolder+'/seed_'+str(seed)+\
    # #           '_'+str(theta)+'worker_freq','wb') as f:
    # #     pk.dump(worker_freq2,f)

    # with open(cwd+'/geo_optim_chars/'+subfolder+'/147_tau_adjust_seed_'+str(seed)+\
    #           '_'+str(theta)+'worker_freq','wb') as f:
    #     pk.dump(worker_freq2,f)

    # with open(cwd+'/geo_optim_chars/'+subfolder+'/147_tau_adjust_seed_'+str(seed)+\
    #           '_'+str(theta)+'D_j','wb') as f:
    #     pk.dump(D_j2,f)


# %% saving
# cwd = os.getcwd()
# with open(cwd + '/optim_plots/objective_d_'+str(devices)+'_w_'\
#     +str(workers)+'_c_'+str(coordinators),'wb') as f:
#     pk.dump(plot_obj,f)

# with open(cwd + '/optim_plots/gradient_d_'+str(devices)+'_w_'\
#     +str(workers)+'_c_'+str(coordinators),'wb') as f:
#     pk.dump(plot_acc,f)

# with open(cwd + '/optim_plots/energy_d_'+str(devices)+'_w_'\
#     +str(workers)+'_c_'+str(coordinators),'wb') as f:
#     pk.dump(plot_energy,f)


