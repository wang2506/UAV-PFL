# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:42:22 2021

@author: henry
"""
import argparse

# %% parser function
def ml_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_style',type=str,default='mnist',\
                        choices=['mnist','fmnist'],\
                        help='data style: mnist or fashion-mnist')
    parser.add_argument('--nn_style',type=str,default='CNN',\
                        choices=['CNN','MLP'],\
                        help='neural network style: cnn or mlp')
    parser.add_argument('--ratio',type=str,default='global',\
                        choices=['global','swarm'],\
                        help='global or swarm-wide ratio varying')
    
    parser.add_argument('--iid_style',type=str,default='mild',\
                        choices=['extreme','mild','iid'],\
                        help='noniid/iid styles')
    
    parser.add_argument('--online',action='store_true',\
                        help='varying data distributions flag')
    parser.add_argument('-rd','--ratio_dynamic',action='store_true',\
                        help='non-unitary initial ratio flag')
    parser.add_argument('--rd_val',type=int,default=2,\
                        help='non-unitary ratio value')
    
    
    parser.add_argument('--swarms',type=int,default=4,\
                        help='swarms') #4 or 10
    parser.add_argument('--l_nps',type=int,default=2,\
                        help='min nodes per swarm')
    parser.add_argument('--h_nps',type=int,default=4,\
                        help='max nodes per swarm')        
    
    parser.add_argument('--time',type=int,default=48)#40)
    parser.add_argument('--comp',type=str,default='gpu',\
                        choices=['gpu','cpu'],\
                        help='gpu or cpu')        
    parser.add_argument('--gpu_num',type=int,default=0,\
                        help='gpu_num')
    
    parser.add_argument('--seed',type=int,default=1)
        
    args = parser.parse_args()
    args.online = True
    return args