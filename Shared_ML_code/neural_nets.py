# -*- coding: utf-8 -*-
"""
@author: henry
"""

import torch
from torch import nn,autograd
from torch.utils.data import DataLoader,Dataset
import numpy as np
import random
from copy import deepcopy
import torch.nn.functional as F
from Shared_ML_code.mod_optimizer import SGD_FO_PFL, SGD_HF_PFL, SGD_HN_PFL_del, SGD_PFL

class MLP(nn.Module):
    def __init__(self,dim_in,dim_hidden,dim_out):
        super(MLP,self).__init__()
        self.layer_input = nn.Linear(dim_in,dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden,dim_out)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        x = x.view(-1,x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

class MLP2(nn.Module):
    def __init__(self,dim_in,dim_hidden,dim_out):
        super(MLP2,self).__init__()
        self.layer_input = nn.Linear(dim_in,dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden,dim_out)
        #self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.sigmoid(x)

class MLP3(nn.Module):
    def __init__(self,dim_in,dim_hidden,dim_out):
        super(MLP2,self).__init__()
        self.layer_input = nn.Linear(dim_in,dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden,dim_out)
        #self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x#self.sigmoid(x)


class CNN(nn.Module):
    def __init__(self, nchannels,nclasses):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(nchannels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x #F.log_softmax(x, dim=1)
        # return F.log_softmax(x,dim=1)
    
class CNNCIFAR10(nn.Module):
    def __init__(self, nchannels,nclasses):
        super(CNNCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(nchannels, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        
        self.fc1 = nn.Linear(16*5*5,120) #2*2*6*5
        self.fc2 = nn.Linear(120,84) #2*2*7*3 - 7 operations, 3 linear
        self.fc3 = nn.Linear(84,nclasses)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x#F.log_softmax(x, dim=1)


class CNNkaggle(nn.Module):
    def __init__(self,nchannels,nclasses):
        super(CNNkaggle, self).__init__()
        self.conv1 = nn.Conv2d(nchannels,32,kernel_size=(3,3))
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,32,kernel_size=(3,3))
        self.conv3 = nn.Conv2d(32,64,kernel_size=(3,3))
        self.conv4 = nn.Conv2d(64,64,kernel_size=(3,3))
        self.conv5 = nn.Conv2d(64,128,kernel_size=(3,3))
        #self.conv6 = nn.Conv2d(128,128,kernel_size=(3,3))
        
        self.fc1 = nn.Linear(128*1*1,64) 
        self.fc2 = nn.Linear(64,nclasses)
        
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.4)
        
        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(128)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        #x = self.bn1(x)
        x = F.relu(self.conv2(x))
        # x = self.pool(self.bn1(F.relu(self.conv2(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        
        # x = self.bn2(F.relu(self.conv3(x)))
        # x = self.bn2(F.relu(self.conv4(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)
        
        # x = self.bn3(F.relu(self.conv5(x)))
        # x = self.bn3(F.relu(self.conv6(x)))
        x = F.relu(self.conv5(x))
        #x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.dropout3(x)
        # print(x.shape)
        
        x = x.view(-1,128)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print(x.shape)
        return F.softmax(x,dim=1)



class segmentdataset(Dataset):
    def __init__(self,dataset,indexes):
        self.dataset = dataset
        self.indexes = indexes
    def __len__(self):
        return len(self.indexes)
    def __getitem__(self,item):
        image,label = self.dataset[self.indexes[item]]
        return image,label
    
class LocalUpdate(object):
    def __init__(self,device,bs,lr,epochs,dataset=None,indexes=None):
        self.device = device
        self.bs = bs
        self.lr = lr
        self.dataset = dataset
        self.indexes = indexes
        self.epochs = epochs
        self.ldr_train = DataLoader(segmentdataset(dataset,indexes),batch_size=bs,shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()
        
    def train(self,net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(),lr=self.lr, momentum=0.5,weight_decay=1e-4) #l2 penalty
        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            
            for batch_indx,(images,labels) in enumerate(self.ldr_train):
                images,labels = images.to(self.device),labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                loss.backward() #this computes the gradient
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net,net.state_dict(),(sum(batch_loss)/len(batch_loss))


class LocalUpdate_PFL(object):
    def __init__(self,device,bs,lr,epochs,dataset=None,indexes=None):
        self.device = device
        self.bs = bs
        self.lr = lr
        self.dataset = dataset
        self.indexes = indexes
        self.epochs = epochs
        self.ldr_train = DataLoader(segmentdataset(dataset,indexes),batch_size=bs,shuffle=True)
        self.ldr_train2 = DataLoader(segmentdataset(dataset,indexes),batch_size=bs,shuffle=True)
        self.ldr_train3 = DataLoader(segmentdataset(dataset,indexes),batch_size=bs,shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()
        
    def train(self,net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(),lr=self.lr, momentum=0.5,weight_decay=1e-4) #l2 penalty
        optimizer2 = SGD_PFL(net.parameters(),lr=self.lr, momentum=0.5,weight_decay=1e-4)
        
        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            
            # calculate the meta-function of SGD
            temp = deepcopy(net.state_dict())
            
            ## first batch 
            # use for SGD
            for batch_indx,(images,labels) in enumerate(self.ldr_train):
                images,labels = images.to(self.device),labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                loss.retain_grad()
                loss.backward() #this computes the gradient
                optimizer.step()
                #batch_loss.append(loss.item())
            
            
            ## second batch
            # parameters updated via SGD, need to pull and
            # we can calculate the loss on these new params
            temp_SGD_new = deepcopy(net.state_dict())
            
            for batch_indx,(images,labels) in enumerate(self.ldr_train2):
                images,labels = images.to(self.device),labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                loss.retain_grad()
                loss.backward() #this computes the gradient
            
            ## third batch
            # reload old params
            net.load_state_dict(temp)
            
            #### net.state_dict vs. net.parameters
            
            # calc hessian
            # I_mat = torch.eye(n=temp.shape[0],m=temp.shape[1])
            
            for batch_indx,(images,labels) in enumerate(self.ldr_train2):
                images,labels = images.to(self.device),labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss_hess = self.loss_func(log_probs,labels)
                loss_hess.retain_grad()
                loss_hess.backward(create_graph=True) #this computes the gradient
                
                loss_hess2 = loss_hess.grad
                loss_hess2.retain_grad()
                loss_hess2.backward()
            
            # identity fix
            for t1 in loss_hess2.grad.keys():
                loss_hess2.grad = (torch.eye(temp[t1].shape[0],temp[t1].shape[1])\
                                - loss_hess2.grad)
            
            loss.grad = loss.grad*loss_hess2.grad
            
            optimizer2.step()
            
            #epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net,net.state_dict(),(sum(batch_loss)/len(batch_loss))


class LocalUpdate_FO_PFL(object):
    def __init__(self,device,bs,lr1,lr2,epochs,dataset=None,indexes=None):
        self.device = device
        self.bs = bs
        self.lr1 = lr1
        self.lr2 = lr2
        self.dataset = dataset
        self.indexes = indexes
        self.epochs = epochs
        self.ldr_train = DataLoader(segmentdataset(dataset,indexes),batch_size=bs,shuffle=True)
        self.ldr_train2 = DataLoader(segmentdataset(dataset,indexes),batch_size=bs,shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()
        
    def train(self,net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(),lr=self.lr1, momentum=0.5,weight_decay=1e-4) #l2 penalty
        
        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            
            # calculate the meta-function of SGD
            temp = deepcopy(net.state_dict())
            temp_params = [] #temp_params = deepcopy(net.parameters())
            for i,j in enumerate(net.parameters()):
                temp_params.append(deepcopy(j))

            ## first batch 
            # use for SGD
            for batch_indx,(images,labels) in enumerate(self.ldr_train):
                images,labels = images.to(self.device),labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                loss.retain_grad()
                loss.backward() #this computes the gradient
                optimizer.step()
                
            
            ## second batch
            # parameters updated via SGD, need to pull and
            # we can calculate the loss on these new params
            temp_SGD_new = deepcopy(net.state_dict())
            temp_SGD_new_params = [] #temp_SGD_new_params = deepcopy(net.parameters())
            for i,j in enumerate(net.parameters()):
                temp_SGD_new_params.append(j)
            
            # net.load_state_dict(temp)
            
            optimizer2 = SGD_FO_PFL(net.parameters(),deepcopy(temp_params),\
                        lr=self.lr2, momentum=0.5,weight_decay=1e-4)
            
            for batch_indx,(images,labels) in enumerate(self.ldr_train2):
                images,labels = images.to(self.device),labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                loss.retain_grad()
                loss.backward() #this computes the gradient
                batch_loss.append(loss.item())
                
                # net.load_state_dict(temp)
                optimizer2.step()
                
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net,net.state_dict(),(sum(batch_loss)/len(batch_loss))


class LocalUpdate_HF_PFL(object): #MLP 1e-3; CNN 1e-4
    def __init__(self,device,bs,lr1,lr2,epochs,dataset=None,indexes=None,del_acc=1e-3):
        self.device = device
        self.bs = bs
        self.lr1 = lr1
        self.lr2 = lr2
        self.del_acc = del_acc
        self.dataset = dataset
        self.indexes = indexes
        self.epochs = epochs
        self.ldr_train = DataLoader(segmentdataset(dataset,indexes),batch_size=int(bs/3),shuffle=True)
        self.ldr_train2 = DataLoader(segmentdataset(dataset,indexes),batch_size=int(bs/3),shuffle=True)
        self.ldr_train3 = DataLoader(segmentdataset(dataset,indexes),batch_size=int(bs/3),shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()
        
    def train(self,net):
        net.train()
        decay_factor = 1e-5
        
        # optimizer = torch.optim.SGD(net.parameters(),lr=self.lr1, momentum=0.5,weight_decay=1e-4) #l2 penalty
        optimizer = SGD_PFL(net.parameters(),lr=self.lr1, momentum=0.5,weight_decay=1e-4)
        
        # optimizer2 = torch.optim.SGD(net.parameters(),lr=self.lr2, momentum=0.5,weight_decay=1e-4)
        
        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            
            # calculate the meta-function of SGD
            temp = deepcopy(net.state_dict())
            # print('start of LocalUpdate_HF_PFL')
            # print(temp['fc2.bias'])
            
            
            temp_params = [] #temp_params = deepcopy(net.parameters())
            for i,j in enumerate(net.parameters()):
                temp_params.append(deepcopy(j))
                
            ## inner params obtain - step size - eta_1
            # total_loss = 0
            for batch_indx,(images,labels) in enumerate(self.ldr_train):
                images,labels = images.to(self.device),labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                # total_loss += loss.item()
                loss.retain_grad()
                loss.backward() #this computes the gradient
                optimizer.step()
            # print('loss testing')
            # print(total_loss)
            
            # this produces the intermediate parameters - needed inner for all three terms
            temp_w_inner = deepcopy(net.state_dict()) #used to find intermediate loss
            # print(temp_w_inner['fc2.bias'])
            
            temp_w_inner_params = []
            for i,j in enumerate(net.parameters()):
                temp_w_inner_params.append(deepcopy(j))
            
            ## calculate term 1 - the optim2 term on batch 2
            # we use the same optimizer as FO_PFL for the isolated batch 2 term
            optimizer2 = SGD_FO_PFL(net.parameters(),deepcopy(temp_params),\
                        lr=self.lr2/self.bs, momentum=0.5,weight_decay=1e-4)
            
            # total_loss = 0
            for batch_indx,(images,labels) in enumerate(self.ldr_train2):
                images,labels = images.to(self.device),labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                # total_loss += loss.item()
                loss.retain_grad()
                loss.backward() #this computes the gradient
                optimizer2.step()
            # print('loss testing optim2')
            # print(total_loss)
            
            manual_w1 = deepcopy(net.state_dict()) #first of three manual add terms
            # print(manual_w1['fc2.bias'])
            
            manual_params1 = []
            for i,j in enumerate(net.parameters()):
                manual_params1.append(deepcopy(j))
            
            net.load_state_dict(temp_w_inner)
            # print(temp_w_inner['fc2.bias'])
            # print('start of optim_plus')
            ## need to check if load_state_dict also changes net.parameters()
            ### confirmed that this works as I thought
            
            ## del_acc terms both plus and minus optim
            # SGD optim will naturally subtract the lr
            optim_plus = SGD_HN_PFL_del(net.parameters(),deepcopy(temp_params),\
                            del_acc=-self.del_acc,momentum=0.5,weight_decay=1e-4)         
            
            # optim_plus = SGD_HN_PFL_del(net.parameters(),deepcopy(temp_params),\
                            # del_acc=-self.del_acc,momentum=0.5,weight_decay=1e-4)       
            
            # optim plus
            # total_loss_op = 0
            for batch_indx,(images,labels) in enumerate(self.ldr_train2):
                images,labels = images.to(self.device),labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                # print(loss.item())
                # if loss.item() >= 100: #the grad result is so small, as the params are stable
                    # break #need to force out otherwise the torch calc will produce nan's

                # total_loss_op += loss.item()
                loss.retain_grad()
                loss.backward() #this computes the gradient
                
                # print(net.state_dict()['fc2.bias'])
                optim_plus.step()
            
            # print('optim plus printing')
            # print(net.state_dict()['fc2.bias'])
            # print('loss_optim_plus = '+ str(total_loss_op))
            
            # cannot use torch.optim.SGD because this grad updates original params
            optim_plus2 = SGD_HN_PFL_del(net.parameters(),deepcopy(temp_params),\
                            del_acc=-self.lr1*self.lr2/(2*self.del_acc*self.bs),\
                        momentum=0.5,weight_decay=1e-4)
            
            # optim_plus2 = SGD_HN_PFL_del(net.parameters(),deepcopy(temp_params),\
            #                 del_acc=self.lr1/(2*self.del_acc),\
            #             momentum=0.5,weight_decay=1e-4)
            
            for batch_indx,(images,labels) in enumerate(self.ldr_train3):
                images,labels = images.to(self.device),labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                loss.retain_grad()
                loss.backward() #this computes the gradient
                batch_loss.append(loss.item()) #### this is superfluous
                
                optim_plus2.step()
            
            optim_plus_w = deepcopy(net.state_dict())
            # print(optim_plus_w['fc2.bias'])
            
            optim_plus_w_params = []
            for i,j in enumerate(net.parameters()):
                optim_plus_w_params.append(deepcopy(j))
            
            # optim minus
            optim_minus = SGD_HN_PFL_del(net.parameters(),deepcopy(temp_params),\
                            del_acc=self.del_acc,momentum=0.5,weight_decay=1e-4) 
            
            # optim_minus = SGD_HN_PFL_del(net.parameters(),deepcopy(temp_params),\
            #                 del_acc=self.del_acc,momentum=0.5,weight_decay=1e-4)             
            
            # reload to calc optim minus
            net.load_state_dict(temp_w_inner)
            
            for batch_indx,(images,labels) in enumerate(self.ldr_train2):
                images,labels = images.to(self.device),labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                loss.retain_grad()
                loss.backward() #this computes the gradient
                optim_minus.step()
            
            # print('start of optim_minus')
            # print(net.state_dict()['fc2.bias'])
            
            optim_minus2 = SGD_HN_PFL_del(net.parameters(),deepcopy(temp_params),\
                            del_acc=self.lr1*self.lr2/(2*self.del_acc*self.bs),\
                        momentum=0.5,weight_decay=1e-4)
            
            # optim_minus2 = SGD_HN_PFL_del(net.parameters(),deepcopy(temp_params),\
            #                 del_acc=self.lr1/(2*self.del_acc),\
            #             momentum=0.5,weight_decay=1e-4)                
                
            for batch_indx,(images,labels) in enumerate(self.ldr_train3):
                images,labels = images.to(self.device),labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                loss.retain_grad()
                loss.backward() #this computes the gradient
                optim_minus2.step()
                
            optim_minus_w = deepcopy(net.state_dict())
            # print(optim_minus_w['fc2.bias'])
            
            optim_minus_w_params = []
            for i,j in enumerate(net.parameters()):
                optim_minus_w_params.append(deepcopy(j))
            
            # manual_w1, optim_plus_w, optim_minus_w combination
            template_w = deepcopy(temp)
            
            for k_i in template_w.keys():
                template_w[k_i] = manual_w1[k_i] + optim_plus_w[k_i] \
                    + optim_minus_w[k_i] - 2*template_w[k_i]
            
            # print(template_w['fc2.bias'])
            
            net.load_state_dict(template_w)
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net,net.state_dict(),(sum(batch_loss)/len(batch_loss))


def FedAvg(w,node_train_sets):
    total_items = sum([len(i) for i in node_train_sets.values()])
    ratios = [len(i)/total_items for i in node_train_sets.values()]
    w_avg = deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(len(w)):
            if i == 0:
                w_avg[k] = w[i][k]*ratios[i]
            else:
                w_avg[k] += w[i][k]*ratios[i]
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedAvg2(w,data_qty):
    total_items = int(sum(data_qty))
    try:
        ratios = data_qty/total_items
    except TypeError:
        data_qty = np.array(data_qty)
        ratios = data_qty/total_items
    except:
        data_qty = np.array(data_qty)
        ratios = data_qty/total_items
    
    w_avg = deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(len(w)):
            if i == 0:
                w_avg[k] = w[i][k]*float(ratios[i])
            else:
                w_avg[k] += w[i][k]*float(ratios[i])
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FPAvg(w): #basic avg
    w_avg = deepcopy(w[0])
    
    for k in w_avg.keys():
        for i in range(len(w)):
            if i == 0:
                w_avg[k] = w[i][k]*(1/len(w))
            else:
                w_avg[k] += w[i][k]*(1/len(w))
    return w_avg


def FedAvg_keras(w,node_train_sets):
    total_items = sum([len(i) for i in node_train_sets.values()])
    ratios = [len(i)/total_items for i in node_train_sets.values()]
    w_avg = deepcopy(w[0])
    for i,val in enumerate(w):
        if i == 0:
            w_avg = np.array(w[i])*ratios[i]
        else:
            w_avg += np.array(w[i])*ratios[i]
    return w_avg


def FedAvgLoss(local_losses,node_train_sets):
    total_items = sum([len(i) for i in node_train_sets.values()])
    ratios = [len(i)/total_items for i in node_train_sets.values()]
    l_avg = 0
    for index,value in enumerate(local_losses):
        l_avg += value*ratios[index]
        
    return l_avg

def grad_calc(local_w,global_w,sg_w,lr):
    #devices and create a data structure
    devices = len(local_w)
    igrad_diff = {i:[] for i in range(devices)}
    ggrad_diff = []
    
    #calc igrad_diffs per device
    for i in range(devices):
        current_set = deepcopy(local_w[i])
        for key,value in current_set.items():
            igrad_diff[i].append((np.array(value) - np.array(global_w[key]))/lr)
    
    # #calc ggrad_diff
    # for key,value in sg_w.items():
    #     ggrad_diff.append((np.array(value)-np.array(global_w[key]))/lr)
    
    # diff = {i:[] for i in range(devices)}
    
    # for i in range(devices):
    #     for index,value in enumerate(igrad_diff[i]):
    #         temp = np.abs(value - ggrad_diff[index])
    #         diff[i].append(temp)
    
    print(devices)
    #populating the data structure containing deltas
    deltas = {i: [] for i in range(devices)}
    temp = 0
    t = []
    for i in range(devices):
        # calculate l2 norm
        for index,value in enumerate(igrad_diff[i]):
            if type(value == list):
                deltas[i].append(value)
            else:
                deltas[i].append(value.tolist)
                
        for ind,j in enumerate(deltas[i]):
            try:
                deltas[i][ind] = np.concatenate(j)
            except IndexError:
                temp = 1
            except ValueError:
                temp = 1
        t.append(np.concatenate(deltas[i]))

    temp = np.linalg.norm(np.concatenate(t),ord=2)
    temp = temp/devices
    print(temp)
    
    return temp
    
    
    
    
    
    
    