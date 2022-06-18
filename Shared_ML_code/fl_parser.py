# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:42:22 2021

@author: henry
"""
import argparse

# %% parser function
def ml_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_style',type=str,default='cifar10',\
                        choices=['mnist','fmnist','cifar10'],\
                        help='data style: mnist or fashion-mnist')
    parser.add_argument('--nn_style',type=str,default='CNN2',\
                        choices=['CNN','MLP','CNN2'],\
                        help='neural network style: cnn or mlp')
    parser.add_argument('--ratio',type=str,default='swarm',\
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
    
    parser.add_argument('--comp',type=str,default='cpu',\
                        choices=['gpu','cpu'],\
                        help='gpu or cpu')
    
    parser.add_argument('--swarms',type=int,default=10,\
                        help='swarms')
    args = parser.parse_args()
    return args