# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:17:08 2021

@author: henry
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

cwd = os.getcwd()
counter = 1
for ratio in [0.5,1,1.5,2,2.5]:
# with open(cwd+'/data/fl_acc_test_base','rb') as f:
#     fl_acc = pickle.load(f)
    
        
    with open(cwd+'/data/fl_acc_test_noniid_'+str(ratio),'rb') as f:
        fl_acc_noniid = pickle.load(f)    
        
        
    # with open(cwd+'/data/FO_hn_pfl_acc_test_base','rb') as f:
    #     fo_pfl_acc = pickle.load(f)
    # fo_pfl_acc2 = [3*i for i in fo_pfl_acc]
    # fo_pfl_acc = fo_pfl_acc2
    
    
    with open(cwd+'/data/FO_hn_pfl_acc_test_noniid_'+str(ratio),'rb') as f:
        fo_pfl_acc_noniid = pickle.load(f)
    
        
    # with open(cwd+'/data/HF_hn_pfl_acc_test_base','rb') as f:
    #     hf_pfl_acc = pickle.load(f)
        
    with open(cwd+'/data/HF_hn_pfl_acc_test_noniid_'+str(ratio),'rb') as f:
        hf_pfl_acc_noniid = pickle.load(f) 
        
        
    # plt.figure(1)
    # plt.plot(fl_acc,label='Hierarchical FL',marker='^',color='black')
    # plt.plot(fo_pfl_acc,label='FO-HN-PFL',marker='o',color='blue')
    # plt.plot(hf_pfl_acc,label='HF-HN-PFL',marker='x',color='brown') 
        
    
    # plt.title('iid data performance')
    # plt.xlabel('global aggregation cycle number')
    # plt.ylabel('accuracy (%)')
    # plt.legend()
    # plt.savefig('iid_perf.png',dpi=500)
        
    plt.figure(counter)
    counter += 1
    plt.plot(fl_acc_noniid,label='Hierarchical FL',marker='^',color='black')
    plt.plot(fo_pfl_acc_noniid,label='FO-HN-PFL',marker='o',color='blue')
    plt.plot(hf_pfl_acc_noniid,label='HF-HN-PFL',marker='x',color='brown') 
    
    
    plt.title('non-iid data performance, ratio = '+str(ratio))
    plt.xlabel('global aggregation cycle number')
    plt.ylabel('accuracy (%)')
    plt.legend()
    plt.savefig('noniid_perf_'+str(ratio)+'.png',dpi=500)
    
    
    