# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:05:56 2022

@author: ch5b2
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

class RML(Dataset):
    def __init__(self,ldir,train=True):
        in_x = np.load(ldir+'/x_10.npy')
        in_y = np.load(ldir+'/y_10.npy')
        
        # in_x = np.reshape(in_x,(in_x.shape[0],in_x.shape[1]*in_x.shape[2]))\
        in_x = in_x[:,np.newaxis,:,:]
        in_x = np.reshape(in_x,(in_x.shape[0],in_x.shape[1],\
                int(np.sqrt(in_x.shape[2]*in_x.shape[3])),\
                    int(np.sqrt(in_x.shape[2]*in_x.shape[3]))))
        x_train, x_test, y_train, y_test = \
            train_test_split(in_x, in_y, test_size=0.20, random_state=25, shuffle=True)
        
        if train == True:
            self.x_data,self.y_data = x_train,y_train
        else:
            self.x_data,self.y_data = x_test,y_test
        
    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i]

if __name__ == '__main__':
    pwd = os.pardir
    trml = RML(ldir=pwd+'/data/radio_ml/')


