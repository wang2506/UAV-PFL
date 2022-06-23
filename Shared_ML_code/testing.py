# -*- coding: utf-8 -*-
"""
@author: henry
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import numpy as np


class segmentdataset(Dataset):
    def __init__(self,dataset,indexes):
        self.dataset = dataset
        self.indexes = indexes
    def __len__(self):
        return len(self.indexes)
    def __getitem__(self,item):
        image,label = self.dataset[self.indexes[item]]
        return image,label

def test_img(net_g, datatest,bs):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size= bs)
    for idx, (data, target) in enumerate(data_loader):
        data = data.to(torch.device('cuda:1'))
        target = target.to(torch.device('cuda:1'))
        
        # data = data.to(torch.device('cpu'))
        # target = target.to(torch.device('cpu'))
        
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100*correct.item() / len(data_loader.dataset)
    return accuracy, test_loss


def test_img2(net_g, datatest,bs,indexes,device=torch.device('cpu')):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(segmentdataset(datatest,indexes),batch_size=bs,shuffle=True)
    
    for idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        
        ## added this for radioML dataset
        target = target.type(torch.long)
        
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100*correct.item() / len(data_loader.dataset)
    return accuracy, test_loss

