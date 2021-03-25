# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:42:22 2021

@author: henry
"""
import os
import argparse

cwd = os.getcwd()

# %% parser function
parser = argparse.ArgumentParser()

## RL probs (epsilon and gamma)
parser.add_argument('--ep_greed',type=float,default=0.7,\
                    help='epsilon greedy val') #0.6,0.7,0.8
parser.add_argument('--ep_min',type=float,default=0.005,\
                    help='epsilon minimum')
parser.add_argument('--g_discount',type=float,default=0.7,\
                    help='gamma discount factor') #0.6,0.7,0.8
parser.add_argument('--replay_bs',type=int,default=20,\
                    help='experience replay batch size')
parser.add_argument('--linear',type=bool,default=True,\
                    help='MLP or CNN') ## argpase cannot evaluate booleans OOB - fix later
parser.add_argument('--cnn_range',type=int, default=2,\
                    help='conv1d range')

# ovr parameters
parser.add_argument('--G_timesteps',type=int,default=10000,\
                    help='number of swarm movements') #10,000
parser.add_argument('--training',type=int,default=1,\
                    help='training or testing the DRL')
parser.add_argument('--centralized',type=bool,default=True,\
                    help='centralized or decentralized')
parser.add_argument('--DQN_update_period',type=int,default=20,\
                    help='DQN update period') #50

args = parser.parse_args()
